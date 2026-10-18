"""
E3 -- seeded, reproducible hypothesis generation (docs/baseline/issues.md).

Fast tests replace the LLM, the sentence encoder and the ARC loader with
fakes, so the cache-build logic (per-batch seeding, batch-boundary
autosave, resume, provenance guard) runs in seconds on CPU. The fake
generator samples from torch's RNG through the real seeded_rng helper and
draws in batch order, so -- like the real model -- its output depends on
the seed AND on how questions are grouped into batches.

The slow test builds a 5-question ARC-Easy cache twice with the real
Qwen2.5-0.5B-Instruct on CPU (downloads the model and dataset):

    python -m pytest tests/test_cache_determinism.py            # fast only
    QAIR_SLOW=1 python -m pytest tests/test_cache_determinism.py
"""

import hashlib
import os
import subprocess
import sys

import pytest
import torch

import training.dataset as dataset_module
from models.generator import seeded_rng
from training.dataset import QAIRDataset, batch_seed
from config import EMBEDDING_DIM, cache_filename

ARC = "ARC-Easy"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def fake_rows(n=30, filtered_at=5):
    """ARC-shaped rows; one row with an unknown answer key gets filtered."""

    rows = []
    for i in range(n):
        rows.append({
            "question": f"Question {i}: which option is right?",
            "choices": {"text": [f"opt{i}-{c}" for c in "ABCD"],
                        "label": list("ABCD")},
            "answerKey": "Z" if i == filtered_at else "ABCD"[i % 4],
        })
    return rows


class FakeGenerator:

    calls = 0
    raise_on_call = None
    prompt_tag = "v1"

    def __init__(self, *args, **kwargs):
        pass

    def provenance(self):
        return {
            "generator": "fake/generator",
            "generator_revision": "fake-rev",
            "gen_config": {"temperature": 0.7},
            "prompt_sha256": hashlib.sha256(self.prompt_tag.encode()).hexdigest(),
            "device": "cpu",
            "device_name": "cpu",
            "torch_version": str(torch.__version__),
            "transformers_version": "fake",
        }

    def generate_batch_with_flags(self, questions, options_list, modes, seed=None):
        type(self).calls += 1
        if type(self).raise_on_call == type(self).calls:
            raise KeyboardInterrupt("simulated interruption")
        assert seed is not None, "the dataset must always pass a seed"
        out = []
        with seeded_rng(seed, "cpu"):
            for q, options in zip(questions, options_list):
                hyps = [
                    f"{q[:12]}|{m}|{oi}|{torch.randint(0, 10 ** 9, (1,)).item()}"
                    for m in modes for oi in range(len(options))
                ]
                out.append({
                    "hypotheses": hyps,
                    "is_fallback": [False] * len(hyps),
                    "polarity": [1.0 if m == "support" else -1.0
                                 for m in modes for _ in options],
                    "hyp_option": [oi for _ in modes for oi in range(len(options))],
                    "modes": [m for m in modes for _ in options],
                })
        return out

    def score_options(self, questions, options_list):
        return [[-float(i) for i in range(len(o))] for o in options_list]

    def fallback(self, question, option, mode):
        return "fallback"


class FakeEncoder:

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        vecs = []
        for t in texts:
            g = torch.Generator().manual_seed(
                int(hashlib.sha256(t.encode()).hexdigest()[:15], 16)
            )
            vecs.append(torch.nn.functional.normalize(
                torch.randn(EMBEDDING_DIM, generator=g), dim=0))
        return torch.stack(vecs)


@pytest.fixture
def fakes(monkeypatch):
    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    monkeypatch.setattr(dataset_module, "load_arc", lambda cfg: {"train": fake_rows()})
    return FakeGenerator


def build(cache_dir, **kwargs):
    return QAIRDataset(split="train", cache_dir=str(cache_dir), arc_config=ARC, **kwargs)


def header(cache_dir):
    # weights_only=True, as QAIRDataset itself loads under torch >= 2.6: the
    # header must stay loadable without unpickling arbitrary objects.
    return torch.load(os.path.join(cache_dir, cache_filename("train", ARC)),
                      map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Fast tests
# ---------------------------------------------------------------------------

def test_batch_seed_is_stable_across_processes():
    expected = int.from_bytes(hashlib.sha256(b"0:5").digest()[:8], "big") & (2 ** 63 - 1)
    assert batch_seed(0, 5) == expected
    assert batch_seed(0, 5) != batch_seed(0, 6) != batch_seed(1, 5)
    # A fresh interpreter with a different str-hash salt gives the same seed.
    code = "from training.dataset import batch_seed; print(batch_seed(0, 5))"
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=True, cwd=os.getcwd())
    assert int(out.stdout.strip().splitlines()[-1]) == expected


def test_seeded_rng_restores_caller_state():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    with seeded_rng(7, "cpu"):
        torch.rand(5)
    assert torch.equal(torch.rand(3), expected)


def test_same_seed_gives_identical_cache(fakes, tmp_path):
    a = build(tmp_path / "a", max_samples=20, gen_seed=0)
    b = build(tmp_path / "b", max_samples=20, gen_seed=0)
    assert [s["hypotheses"] for s in a.samples] == [s["hypotheses"] for s in b.samples]
    ha, hb = header(tmp_path / "a"), header(tmp_path / "b")
    assert ha["hypotheses_sha256"] == hb["hypotheses_sha256"]
    assert ha["provenance"]["gen_seed"] == 0
    assert ha["version"] == 46


def test_different_seed_gives_different_cache(fakes, tmp_path):
    build(tmp_path / "a", max_samples=20, gen_seed=0)
    build(tmp_path / "b", max_samples=20, gen_seed=1)
    assert header(tmp_path / "a")["hypotheses_sha256"] != \
        header(tmp_path / "b")["hypotheses_sha256"]


def test_resumed_build_equals_uninterrupted(fakes, tmp_path):
    full = build(tmp_path / "full", gen_seed=0)
    assert header(tmp_path / "full")["complete"] is True

    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = 3          # dies inside the third batch
    with pytest.raises(KeyboardInterrupt):
        build(tmp_path / "resumed", gen_seed=0, autosave_every=8)
    partial = header(tmp_path / "resumed")
    # Batches: rows 0-4 (flushed by the filtered row 5), then rows 6-13.
    # The autosave fires after the second batch, i.e. at a batch boundary.
    assert partial["complete"] is False
    assert (len(partial["samples"]), partial["raw_index"]) == (13, 14)

    # What a user does after a crash: the lock is stale, take over.
    os.remove(os.path.join(tmp_path / "resumed", cache_filename("train", ARC) + ".lock"))
    FakeGenerator.raise_on_call = None
    resumed = build(tmp_path / "resumed", gen_seed=0, autosave_every=8)

    assert [s["hypotheses"] for s in resumed.samples] == \
        [s["hypotheses"] for s in full.samples]
    final = header(tmp_path / "resumed")
    assert final["hypotheses_sha256"] == header(tmp_path / "full")["hypotheses_sha256"]
    assert len(final["provenance"]["segments"]) == 2


@pytest.mark.parametrize("change", ["seed", "prompt"])
def test_resume_with_different_settings_raises(fakes, tmp_path, change):
    FakeGenerator.raise_on_call = 3
    with pytest.raises(KeyboardInterrupt):
        build(tmp_path / "c", gen_seed=0, autosave_every=8)
    lock = os.path.join(tmp_path / "c", cache_filename("train", ARC) + ".lock")
    os.remove(lock)
    FakeGenerator.raise_on_call = None
    if change == "prompt":
        FakeGenerator.prompt_tag = "v2"
    key = "gen_seed" if change == "seed" else "prompt_sha256"
    with pytest.raises(RuntimeError, match=key):
        build(tmp_path / "c", gen_seed=1 if change == "seed" else 0)
    assert not os.path.exists(lock)          # the refused build released its lock


def test_cache_without_provenance_is_never_resumed(fakes, tmp_path):
    good = build(tmp_path / "src", max_samples=8, gen_seed=0)
    os.makedirs(tmp_path / "old")
    torch.save(
        {"samples": good.samples, "raw_index": 8, "complete": False, "version": 45},
        os.path.join(tmp_path / "old", cache_filename("train", ARC)),
    )
    # Reading within its cap works ...
    assert len(build(tmp_path / "old", max_samples=8).samples) == 8
    # ... resuming does not.
    with pytest.raises(RuntimeError, match="without provenance"):
        build(tmp_path / "old", max_samples=16)


# ---------------------------------------------------------------------------
# Slow test: the real generator
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("QAIR_SLOW") != "1",
                    reason="set QAIR_SLOW=1 (downloads Qwen2.5-0.5B, MiniLM, ARC-Easy)")
def test_real_qwen_same_seed_same_hash(monkeypatch, tmp_path):
    import models.encoder
    import models.generator

    # CPU only: the guarantee is per device, and this machine would pick MPS.
    monkeypatch.setattr(models.generator, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(models.encoder, "resolve_device", lambda: "cpu")

    a = build(tmp_path / "a", max_samples=5, gen_seed=0)
    b = build(tmp_path / "b", max_samples=5, gen_seed=0)
    ha, hb = header(tmp_path / "a"), header(tmp_path / "b")

    print(f"\nhypotheses_sha256 a = {ha['hypotheses_sha256']}")
    print(f"hypotheses_sha256 b = {hb['hypotheses_sha256']}")
    print(f"provenance = {ha['provenance']}")
    print(f"first hypothesis: {a.samples[0]['hypotheses'][0]!r}")

    assert len(a.samples) == 5
    assert ha["hypotheses_sha256"] == hb["hypotheses_sha256"]
    assert ha["provenance"]["generator"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert ha["provenance"]["segments"][0]["device"] == "cpu"
