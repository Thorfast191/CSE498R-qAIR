"""
A6 -- cache edge cases that silently changed state or data
(docs/baseline/issues.md).

    python -m pytest tests/test_cache_edge_cases.py
"""

import hashlib
import os

import pytest
import torch

import training.dataset as dataset_module
from config import cache_filename
from tests.test_cache_determinism import ARC, FakeEncoder, FakeGenerator, fake_rows
from training.dataset import QAIRDataset


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cache_file(cache_dir):
    return os.path.join(cache_dir, cache_filename("train", ARC))


def ds(cache_dir, split="train", **kwargs):
    return QAIRDataset(split=split, cache_dir=str(cache_dir), arc_config=ARC, **kwargs)


def _must_not_be_called(*args, **kwargs):
    raise AssertionError("a complete cache must never start a build")


@pytest.fixture
def fakes(monkeypatch):
    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    monkeypatch.setattr(dataset_module, "load_arc",
                        lambda cfg: {"train": fake_rows(), "validation": fake_rows(6)})
    return monkeypatch


@pytest.mark.parametrize("allow_build", [False, True])
def test_complete_cache_is_final(fakes, tmp_path, allow_build, capsys):
    ds(tmp_path, allow_build=True)                    # complete: 29 of 30 rows
    path = cache_file(tmp_path)
    digest = sha256(path)
    fakes.setattr(dataset_module, "HypothesisGenerator", _must_not_be_called)
    fakes.setattr(dataset_module, "load_arc", _must_not_be_called)

    out = ds(tmp_path, max_samples=40, allow_build=allow_build)

    assert len(out.samples) == 29
    assert "has only 29 samples; returning all 29" in capsys.readouterr().out
    assert sha256(path) == digest
    assert torch.load(path, map_location="cpu", weights_only=True)["complete"] is True
    assert len(ds(tmp_path, max_samples=10).samples) == 10


def test_missing_split_raises_instead_of_substituting(fakes, tmp_path):
    with pytest.raises(ValueError) as err:
        ds(tmp_path, split="test", max_samples=5, allow_build=True)
    msg = str(err.value)
    assert "'test'" in msg and "['train', 'validation']" in msg
    assert not os.path.exists(os.path.join(tmp_path, cache_filename("test", ARC)))
    assert not os.path.exists(os.path.join(tmp_path, cache_filename("test", ARC) + ".lock"))


class FlakyScorer(FakeGenerator):
    """Batch scoring fails; per-question scoring fails only for one question."""

    bad = "Question 3:"

    def score_options(self, questions, options_list):
        if len(questions) > 1:
            raise RuntimeError("simulated batch scoring failure")
        if questions[0].startswith(self.bad):
            raise RuntimeError("simulated single-item failure")
        return [[-1.0 - i for i in range(len(o))] for o in options_list]


def test_scoring_failure_is_flagged_per_question(fakes, tmp_path, capsys):
    fakes.setattr(dataset_module, "HypothesisGenerator", FlakyScorer)

    out = ds(tmp_path, max_samples=8, allow_build=True)

    failed = [s for s in out.samples if s["score_failed"]]
    assert [s["question"][:11] for s in failed] == ["Question 3:"]
    assert torch.equal(failed[0]["llm_logprob"], torch.zeros(4))
    ok = [s for s in out.samples if not s["score_failed"]]
    assert len(ok) == 7 and all(s["llm_logprob"][0] == -1.0 for s in ok)
    assert "score_failed 1/8" in capsys.readouterr().out


def test_cache_without_score_flag_reads_as_not_failed(fakes, tmp_path):
    built = ds(tmp_path / "src", max_samples=8, allow_build=True)
    old = [{k: v for k, v in s.items() if k != "score_failed"} for s in built.samples]
    os.makedirs(tmp_path / "old")
    torch.save({"samples": old, "raw_index": 9, "complete": True, "version": 45},
               cache_file(tmp_path / "old"))

    out = ds(tmp_path / "old")

    assert len(out.samples) == 8
    assert all(s["score_failed"] is False for s in out.samples)
