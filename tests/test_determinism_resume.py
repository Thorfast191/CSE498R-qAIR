"""
E10 -- training runs are deterministic (strict by default, warn-only on
request, recorded in the provenance).
A11 -- a resumed run equals an uninterrupted one: every RNG stream is
checkpointed and restored. (docs/baseline/issues.md)

    python -m pytest tests/test_determinism_resume.py

Real model on CPU with the fake cache from test_checkpoint_provenance;
needs pennylane. Validation is real (not scripted), so histories compare
every metric, bit for bit.
"""

import os
import random

import numpy as np
import pytest
import torch

import training.train as train_module
from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import (  # noqa: F401  (fixture)
    latest, provenance, save, smoke_cache, tiny_trainer,
)
from training.checkpoint import (
    CheckpointMismatchError, load_checkpoint_file, load_or_resume, run_key,
)
from training.seed import capture_rng_state, restore_rng_state, set_seed

RUN = run_key("qair", ARC, 42)


class Interrupted(Exception):
    """Stands in for a Colab disconnect in the middle of a run."""


def interrupt_validation_at(monkeypatch, call):
    """Real validation, except that call number `call` (0-based) raises."""

    real = train_module.evaluate
    calls = {"n": 0}

    def evaluate(model, loader, device):
        if calls["n"] == call:
            raise Interrupted()
        calls["n"] += 1
        return real(model, loader, device)

    monkeypatch.setattr(train_module, "evaluate", evaluate)
    return lambda: monkeypatch.setattr(train_module, "evaluate", real)


def train(ckpt_dir, cache_dir, epochs=4, **kwargs):
    from training.runner import run_training

    return run_training(
        cache_dir=str(cache_dir), ckpt_dir=str(ckpt_dir), train_samples=16,
        val_samples=8, epochs=epochs, patience=None, seed=42,
        with_test=False, arc_config=ARC, **kwargs,
    )


def weights(path):
    return load_checkpoint_file(path)["model"]


def assert_same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def assert_same_history(a, b):
    """Every metric equal, including the DoD §2.2/§2.7 diagnostics and W0."""
    from training.diagnostics import INITIAL_WEIGHTS

    assert a.keys() == b.keys()
    for k in a:
        if k == INITIAL_WEIGHTS:
            assert_same_weights(a[k], b[k])
        else:
            assert a[k] == b[k], k


# ---------------------------------------------------------------------------
# E10: deterministic mode
# ---------------------------------------------------------------------------

def test_strict_and_warn_only_modes(monkeypatch):
    set_seed(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark

    set_seed(0, deterministic=True, warn_only=True)
    assert torch.is_deterministic_algorithms_warn_only_enabled()

    set_seed(0, deterministic=True)


def test_cublas_workspace_set_only_if_unset(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_seed(0, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    set_seed(0, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_same_seed_gives_identical_runs(tmp_path, smoke_cache):
    from training.evaluate import evaluate
    from training.runner import build_loaders
    from evaluation.input_ablation import build_model_from_config

    first = train(tmp_path / "a", smoke_cache, epochs=2)
    second = train(tmp_path / "b", smoke_cache, epochs=2)

    assert_same_history(first, second)

    path_a = latest(tmp_path / "a", RUN)
    path_b = latest(tmp_path / "b", RUN)
    assert_same_weights(weights(path_a), weights(path_b))

    _, val_loader, _ = build_loaders(str(smoke_cache), 16, 8, seed=42,
                                     with_test=False, arc_config=ARC)
    records = []
    for path in (path_a, path_b):
        ckpt = load_checkpoint_file(path)
        model = build_model_from_config(ckpt["provenance"]["model_config"], "cpu")
        model.load_state_dict(ckpt["model"])
        records.append(evaluate(model, val_loader, "cpu", return_records=True)[1])

    assert records[0] == records[1]


def test_warn_only_is_recorded_and_compared(tmp_path, smoke_cache):
    train(tmp_path, smoke_cache, epochs=1)
    tc = load_checkpoint_file(latest(tmp_path, RUN))["provenance"]["train_config"]
    assert (tc["deterministic"], tc["deterministic_warn_only"]) == (True, False)

    with pytest.raises(CheckpointMismatchError, match="deterministic_warn_only"):
        train(tmp_path, smoke_cache, epochs=1, deterministic_warn_only=True)


# ---------------------------------------------------------------------------
# A11: RNG state
# ---------------------------------------------------------------------------

def draws(generator):
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(3).tolist(),
        torch.randperm(10, generator=generator).tolist(),
    )


def test_rng_state_round_trips_under_weights_only(tmp_path):
    g = torch.Generator().manual_seed(7)
    set_seed(3, deterministic=True)
    draws(g)

    path = tmp_path / "rng.pt"
    torch.save(capture_rng_state(g), path)
    expected = draws(g)

    draws(g)
    restore_rng_state(load_checkpoint_file(str(path)), g)

    assert draws(g) == expected


def test_checkpoint_without_rng_state_is_refused(tmp_path):
    prov = provenance()
    save(tiny_trainer(tmp_path, prov), 0, 0.5)
    path = latest(tmp_path)
    ckpt = load_checkpoint_file(path)
    del ckpt["rng"]
    torch.save(ckpt, path)

    with pytest.raises(CheckpointMismatchError, match="rng"):
        load_or_resume(tiny_trainer(tmp_path, prov), str(tmp_path),
                       "qair_ARC-Easy_s42", epochs=8, provenance=prov)


def test_resumed_run_equals_uninterrupted(tmp_path, smoke_cache, monkeypatch):
    whole = train(tmp_path / "whole", smoke_cache)

    restore_validation = interrupt_validation_at(monkeypatch, call=2)
    with pytest.raises(Interrupted):
        train(tmp_path / "resumed", smoke_cache)
    restore_validation()
    resumed = train(tmp_path / "resumed", smoke_cache)

    assert len(whole["loss"]) == 4
    assert_same_history(resumed, whole)
    assert_same_weights(weights(latest(tmp_path / "whole", RUN)),
                        weights(latest(tmp_path / "resumed", RUN)))


def test_grid_arm_does_not_depend_on_earlier_arms(tmp_path, smoke_cache):
    from training.ablations import run_ablation_suite

    def grid(ckpt_dir, configs):
        return run_ablation_suite(
            cache_dir=str(smoke_cache), ckpt_dir=str(ckpt_dir), epochs=2,
            seeds=[42], with_test=False, arc_config=ARC, configs=configs,
            train_samples=16, val_samples=8,
        )

    after_a1 = grid(tmp_path / "both", ["A1_baseline", "A2_validator"])
    alone = grid(tmp_path / "alone", ["A2_validator"])

    assert after_a1["A2_validator"][0]["history"] == alone["A2_validator"][0]["history"]


# ---------------------------------------------------------------------------
# E12: the circuit engine is part of the run
# ---------------------------------------------------------------------------

def interrupted_in_epoch_2(tmp_path, smoke_cache, monkeypatch):
    """A 2-epoch run that disconnects during its second validation."""

    restore = interrupt_validation_at(monkeypatch, 1)
    with pytest.raises(Interrupted):
        train(tmp_path, smoke_cache, epochs=2)
    restore()


def test_resume_under_the_other_circuit_engine_refuses(tmp_path, smoke_cache, monkeypatch):
    import config

    monkeypatch.setattr(config, "QUANTUM_SIMULATOR", "pennylane")
    interrupted_in_epoch_2(tmp_path, smoke_cache, monkeypatch)
    saved = load_checkpoint_file(latest(tmp_path, RUN))["provenance"]
    assert saved["quantum_simulator"] == "pennylane"

    monkeypatch.setattr(config, "QUANTUM_SIMULATOR", "torch")
    with pytest.raises(CheckpointMismatchError) as refused:
        train(tmp_path, smoke_cache, epochs=2)

    mismatches = [line for line in str(refused.value).splitlines() if "checkpoint=" in line]
    assert mismatches == ["  quantum_simulator: checkpoint='pennylane'  this run='torch'"]


def test_resume_under_the_same_engine_continues(tmp_path, smoke_cache, monkeypatch):
    interrupted_in_epoch_2(tmp_path, smoke_cache, monkeypatch)
    resumed = train(tmp_path, smoke_cache, epochs=2)

    assert resumed["provenance"]["quantum_simulator"] == "torch"
    assert resumed["manifest"]["quantum_simulator"] == "torch"
    assert len(resumed["loss"]) == 2
