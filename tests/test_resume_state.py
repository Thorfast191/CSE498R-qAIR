"""
A4 -- a resumed run keeps its history and early-stopping count.
A10 -- an early-stopped run stays finished when it is run again.
(docs/baseline/issues.md)

    python -m pytest tests/test_resume_state.py

Validation accuracy is scripted (training.train.evaluate is replaced), so
when early stopping fires depends only on the bookkeeping under test, not
on training noise. Real model on CPU, fake cache; needs pennylane.
"""

import os
from collections import defaultdict

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

RUN = run_key("qair", ARC, 42)


class Interrupted(Exception):
    """Stands in for a Colab disconnect in the middle of a run."""


class ScriptedValidation:
    """Returns accs[i] on the i-th validation call; raises on interrupt_at."""

    def __init__(self, monkeypatch, accs, interrupt_at=None):
        self.accs = list(accs)
        self.calls = 0
        self.interrupt_at = interrupt_at
        monkeypatch.setattr(train_module, "evaluate", self)

    def __call__(self, model, loader, device):
        if self.calls == self.interrupt_at:
            raise Interrupted()
        metrics = defaultdict(float)
        metrics["acc"] = self.accs[self.calls]
        self.calls += 1
        return metrics


def train(ckpt_dir, cache_dir, epochs, patience):
    from training.runner import run_training

    return run_training(
        cache_dir=str(cache_dir), ckpt_dir=str(ckpt_dir), train_samples=16,
        val_samples=8, epochs=epochs, patience=patience, seed=42,
        with_test=False, arc_config=ARC,
    )


def interrupted_then_resumed(monkeypatch, ckpt_dir, cache_dir, accs,
                             after, epochs, patience):
    script = ScriptedValidation(monkeypatch, accs, interrupt_at=after)
    with pytest.raises(Interrupted):
        train(ckpt_dir, cache_dir, epochs, patience)

    script.interrupt_at = None
    return train(ckpt_dir, cache_dir, epochs, patience)


# ---------------------------------------------------------------------------
# A4
# ---------------------------------------------------------------------------

def test_resumed_history_has_every_epoch(tmp_path, smoke_cache, monkeypatch):
    accs = [0.5, 0.625, 0.375, 0.75]

    ScriptedValidation(monkeypatch, accs)
    whole = train(tmp_path / "whole", smoke_cache, epochs=4, patience=None)

    resumed = interrupted_then_resumed(
        monkeypatch, tmp_path / "resumed", smoke_cache, accs,
        after=2, epochs=4, patience=None,
    )

    for history in (whole, resumed):
        assert history["acc"] == accs
        assert len(history["loss"]) == 4
        assert history["best_acc"] == 0.75

    exported = load_checkpoint_file(
        os.path.join(tmp_path / "resumed", f"{RUN}_history.pt"))
    assert exported["acc"] == accs


def test_early_stopping_count_survives_resume(tmp_path, smoke_cache, monkeypatch):
    # patience 2: best at epoch 1, no improvement at epochs 2 and 3 -> stop
    # after 3 epochs. The interruption falls between epochs 2 and 3, so a
    # count reset on resume would train a 4th epoch.
    accs = [0.5, 0.375, 0.375, 0.375]

    ScriptedValidation(monkeypatch, accs)
    whole = train(tmp_path / "whole", smoke_cache, epochs=4, patience=2)

    resumed = interrupted_then_resumed(
        monkeypatch, tmp_path / "resumed", smoke_cache, accs,
        after=2, epochs=4, patience=2,
    )

    assert whole["acc"] == resumed["acc"] == [0.5, 0.375, 0.375]

    ckpt = load_checkpoint_file(latest(tmp_path / "resumed", RUN))
    assert (ckpt["epoch"], ckpt["epochs_no_improve"], ckpt["stopped_early"]) == (2, 2, True)


def test_checkpoint_without_resume_state_is_refused(tmp_path):
    prov = provenance()
    save(tiny_trainer(tmp_path, prov), 0, 0.5)
    path = latest(tmp_path)
    ckpt = load_checkpoint_file(path)
    del ckpt["history"]
    torch.save(ckpt, path)

    with pytest.raises(CheckpointMismatchError, match="lacks the resume state"):
        load_or_resume(tiny_trainer(tmp_path, prov), str(tmp_path),
                       "qair_ARC-Easy_s42", epochs=8, provenance=prov)


# ---------------------------------------------------------------------------
# A10
# ---------------------------------------------------------------------------

def test_early_stopped_run_is_not_trained_again(tmp_path, smoke_cache,
                                                monkeypatch, capsys):
    ScriptedValidation(monkeypatch, [0.5, 0.375, 0.375])
    first = train(tmp_path, smoke_cache, epochs=6, patience=2)
    assert len(first["acc"]) == 3
    before = load_checkpoint_file(latest(tmp_path, RUN))

    rerun_script = ScriptedValidation(monkeypatch, [], interrupt_at=0)
    again = train(tmp_path, smoke_cache, epochs=6, patience=2)

    assert rerun_script.calls == 0
    assert again["acc"] == first["acc"]
    assert f"[SKIP] {RUN} already finished (stopped early)" in capsys.readouterr().out
    assert load_checkpoint_file(latest(tmp_path, RUN))["epoch"] == before["epoch"]


def test_grid_skips_early_stopped_arm(tmp_path, smoke_cache, monkeypatch):
    from training.ablations import run_ablation_suite

    def grid():
        return run_ablation_suite(
            cache_dir=str(smoke_cache), ckpt_dir=str(tmp_path), epochs=6,
            patience=2, seeds=[42], with_test=False, arc_config=ARC,
            configs=["A1_baseline"], train_samples=16, val_samples=8,
        )

    ScriptedValidation(monkeypatch, [0.5, 0.375, 0.375])
    grid()

    rerun_script = ScriptedValidation(monkeypatch, [], interrupt_at=0)
    entry = grid()["A1_baseline"][0]

    assert rerun_script.calls == 0
    assert entry["history"]["acc"] == [0.5, 0.375, 0.375]
    assert entry["best_val_acc"] == 0.5
