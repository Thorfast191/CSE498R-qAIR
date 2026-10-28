"""
E8 -- scripts/smoke_timing.py builds a small cache in a separate directory,
trains 1 epoch and reports seconds per stage (docs/baseline/issues.md E8).

    python -m pytest tests/test_smoke_timing.py

Runs the real script with the fake generator / encoder of
tests/test_cache_determinism.py and the real model on CPU; needs pennylane.
"""

import json
import os
import sys

import pytest

import training.dataset as dataset_module
from tests.test_cache_determinism import ARC, FakeEncoder, FakeGenerator, fake_rows

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import smoke_timing  # noqa: E402

N_TRAIN, N_VAL = 4, 2


@pytest.fixture
def fakes(monkeypatch):
    pytest.importorskip("pennylane")

    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    rows = fake_rows()
    monkeypatch.setattr(dataset_module, "load_arc",
                        lambda cfg: {"train": rows, "validation": rows})

    import training.runner as runner_module
    monkeypatch.setattr(runner_module, "device", "cpu")


def smoke(tmp_path, **kwargs):
    return smoke_timing.main([
        "--cache-dir", str(tmp_path / "smoke_cache"),
        "--ckpt-dir", str(tmp_path / "smoke_ckpt"),
        "--train", str(N_TRAIN), "--val", str(N_VAL), "--arc-config", ARC,
        *sum(([f"--{k}", str(v)] for k, v in kwargs.items()), []),
    ])


def test_report_has_every_stage(tmp_path, fakes, capsys):
    smoke(tmp_path)

    with open(tmp_path / "smoke_ckpt" / "smoke_timing.json") as f:
        report = json.load(f)

    assert report["questions"] == {"train": N_TRAIN, "validation": N_VAL}
    assert set(report["seconds"]) == set(smoke_timing.STAGES)
    assert all(report["seconds"][s] > 0 for s in smoke_timing.STAGES)
    assert report["calls"]["generation"] >= 2 and report["calls"]["embedding"] >= 2
    assert report["calls"]["eval"] == 1 and report["train_steps"] == 1
    assert report["quantum_device"] == "torch.statevector"   # E12 default
    assert report["projection_DERIVED"]["full_split_sizes"] == \
        {"train": 2251, "validation": 570, "test": 2376}
    assert report["projection_DERIVED"]["train_steps_per_epoch"] == 141
    assert "SMOKE TIMING" in capsys.readouterr().out


def test_wrappers_are_removed_afterwards(tmp_path, fakes):
    import training.train as train_module

    originals = (FakeGenerator.generate_batch_with_flags, FakeGenerator.score_options,
                 FakeGenerator.__init__, FakeEncoder.encode, FakeEncoder.__init__,
                 train_module.evaluate, train_module.Trainer.train,
                 train_module.Trainer.save_checkpoint, train_module.Trainer.save_history)
    smoke(tmp_path)

    assert originals == (
        FakeGenerator.generate_batch_with_flags, FakeGenerator.score_options,
        FakeGenerator.__init__, FakeEncoder.encode, FakeEncoder.__init__,
        train_module.evaluate, train_module.Trainer.train,
        train_module.Trainer.save_checkpoint, train_module.Trainer.save_history)


def test_refuses_a_directory_that_holds_a_cache(tmp_path, fakes):
    smoke(tmp_path)                                   # first run builds it
    FakeGenerator.calls = 0

    with pytest.raises(smoke_timing.SmokeDirectoryError, match="already holds a cache"):
        smoke(tmp_path)
    assert FakeGenerator.calls == 0                   # nothing generated


def test_refuses_the_real_cache_directory(tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(smoke_timing, "CACHE_DIR", str(tmp_path / "smoke_cache"))

    with pytest.raises(smoke_timing.SmokeDirectoryError, match="config.CACHE_DIR"):
        smoke(tmp_path)
