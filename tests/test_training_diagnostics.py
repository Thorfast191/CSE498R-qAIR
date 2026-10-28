"""
DoD §2.2 (per-epoch, per-module gradient norms) and §2.7 (per-epoch
parameter drift) -- training/diagnostics.py, logged by training/train.py.

    python -m pytest tests/test_training_diagnostics.py

The resumed-equals-uninterrupted check for these keys is in
tests/test_determinism_resume.py (assert_same_history).
"""

import math
import os

import pytest
import torch
import torch.nn as nn

from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)
from tests.test_determinism_resume import RUN, train
from training.checkpoint import load_checkpoint_file
from training.diagnostics import (
    EPOCH_KEYS,
    INITIAL_SCALARS,
    INITIAL_WEIGHTS,
    ROOT,
    TOTAL,
    GradNormTracker,
    parameter_drift,
    start_of_run,
)

EPOCHS = 2


# ---------------------------------------------------------------------------
# Unit level: a tiny model with hand-set gradients
# ---------------------------------------------------------------------------

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 1, bias=False)
        self.b = nn.Linear(1, 1, bias=False)
        self.phase = nn.Parameter(torch.tensor(math.pi))


def set_grads(model, a, b, phase):
    model.a.weight.grad = torch.tensor([a])
    model.b.weight.grad = None if b is None else torch.tensor([[b]])
    model.phase.grad = torch.tensor(phase)


def test_grad_norms_per_module_mean_max_and_clipping():
    model = Tiny()
    tracker = GradNormTracker(model, max_norm=2.0)

    set_grads(model, [3.0, 4.0], None, 0.5)
    first = tracker.measure()
    assert first == pytest.approx({"a": 5.0, "b": 0.0, ROOT: 0.5})
    tracker.add({**first, TOTAL: 5.1}, skipped=False)

    set_grads(model, [0.0, 1.0], 2.0, 0.0)
    tracker.add({**tracker.measure(), TOTAL: 1.5}, skipped=False)

    tracker.add({"a": math.inf, "b": 0.0, ROOT: 0.0, TOTAL: math.inf}, skipped=True)

    grad_norm, clipped, skipped = tracker.summary()

    assert grad_norm["a"] == pytest.approx({"mean": 3.0, "max": 5.0})
    assert grad_norm["b"] == pytest.approx({"mean": 1.0, "max": 2.0})
    assert grad_norm[TOTAL] == pytest.approx({"mean": 3.3, "max": 5.1})
    assert clipped == 0.5          # 5.1 > 2.0, 1.5 is not
    assert skipped == 1            # the inf step is counted, not averaged


def test_drift_and_relative_weight_change_from_the_run_start():
    model = Tiny()
    with torch.no_grad():
        model.a.weight.copy_(torch.tensor([[3.0, 4.0]]))
        model.b.weight.fill_(2.0)

    history = {}
    start_of_run(history, model)
    assert set(EPOCH_KEYS) <= set(history)

    with torch.no_grad():
        model.a.weight.add_(torch.tensor([[0.0, 0.5]]))
        model.phase.sub_(0.25)

    scalars, drift, change = parameter_drift(model, history)

    assert scalars == pytest.approx({"phase": math.pi - 0.25})
    assert drift == pytest.approx({"phase": -0.25})
    assert change == pytest.approx({"a": 0.1, "b": 0.0, ROOT: 0.25 / math.pi})


def test_relative_change_of_a_zero_initialised_module_is_floored():
    from training.diagnostics import NORM_FLOOR, describe_drift, floored_modules

    model = Tiny()
    with torch.no_grad():
        model.a.weight.zero_()                     # starts at exactly zero
        model.b.weight.fill_(2.0)

    history = {}
    start_of_run(history, model)
    with torch.no_grad():
        model.a.weight.add_(torch.tensor([[0.003, 0.004]]))   # ||W - W0|| = 0.005

    _, _, change = parameter_drift(model, history)

    assert change["a"] == pytest.approx(0.005 / NORM_FLOOR)   # 5, not ~5e9
    assert change["b"] == 0.0
    assert floored_modules(model, history) == {"a"}
    line = describe_drift({}, change, floored_modules(model, history))
    assert "a* 5" in line and "b 0" in line


def test_start_of_run_keeps_a_resumed_runs_w0():
    model = Tiny()
    history = {}
    start_of_run(history, model)
    w0 = {k: v.clone() for k, v in history[INITIAL_WEIGHTS].items()}

    with torch.no_grad():
        model.phase.fill_(0.0)
    start_of_run(history, model)          # a resume calls it again

    assert all(torch.equal(w0[k], history[INITIAL_WEIGHTS][k]) for k in w0)
    assert history[INITIAL_SCALARS] == {"phase": pytest.approx(math.pi)}


# ---------------------------------------------------------------------------
# Real training run on the fake cache
# ---------------------------------------------------------------------------

@pytest.fixture
def run(tmp_path, smoke_cache):
    ckpt_dir = tmp_path / "diagnostics"
    history = train(ckpt_dir, smoke_cache, epochs=EPOCHS)
    return ckpt_dir, history


def test_every_epoch_logs_every_key(run):
    _, history = run
    modules = {name for name, _ in _model(run).named_children()} | {ROOT}

    for key in EPOCH_KEYS:
        assert len(history[key]) == EPOCHS, key

    for grad_norm, change in zip(history["grad_norm"], history["weight_change"]):
        assert set(grad_norm) == modules | {TOTAL}
        assert set(change) == modules
        assert all(v["mean"] >= 0 and v["max"] >= v["mean"] for v in grad_norm.values())

    assert all(0.0 <= c <= 1.0 for c in history["clipped_fraction"])
    assert history["skipped_steps"] == [0] * EPOCHS        # no AMP on CPU


def test_logged_values_match_the_checkpoint(run):
    ckpt_dir, history = run
    final = load_checkpoint_file(os.path.join(str(ckpt_dir), f"{RUN}_latest.pt"))["model"]
    w0 = history[INITIAL_WEIGHTS]

    for name, value in history["scalars"][-1].items():
        assert value == pytest.approx(final[name].item(), abs=0)
        assert history["scalar_drift"][-1][name] == pytest.approx(
            final[name].item() - history[INITIAL_SCALARS][name], abs=0)

    in_proj = [k for k in w0 if k.startswith("in_proj.")]
    diff = sum((final[k] - w0[k]).pow(2).sum() for k in in_proj).sqrt()
    norm = sum(w0[k].pow(2).sum() for k in in_proj).sqrt()
    assert history["weight_change"][-1]["in_proj"] == pytest.approx((diff / norm).item())
    assert history["weight_change"][-1]["in_proj"] > 0


def test_module_norms_are_measured_before_clipping(tmp_path, smoke_cache, monkeypatch):
    """Per step, the module norms combine to the unclipped total norm."""

    steps = []
    real_add = GradNormTracker.add

    def add(self, norms, skipped):
        steps.append(norms)
        real_add(self, norms, skipped)

    monkeypatch.setattr(GradNormTracker, "add", add)
    history = train(tmp_path, smoke_cache, epochs=1)

    assert steps and history["clipped_fraction"][0] > 0      # clipping happened
    for norms in steps:
        modules = math.sqrt(sum(v ** 2 for k, v in norms.items() if k != TOTAL))
        assert modules == pytest.approx(norms[TOTAL], rel=1e-4)


def test_lines_are_printed_every_epoch(tmp_path, smoke_cache, capsys):
    train(tmp_path, smoke_cache, epochs=EPOCHS)
    out = capsys.readouterr().out

    for prefix in ("Grad norm     :", "Scalar drift  :", "Weight change :"):
        assert out.count(prefix) == EPOCHS, prefix


def test_grid_results_leave_out_w0_but_the_run_keeps_it(tmp_path, smoke_cache):
    from training.ablations import run_ablation_suite

    results = run_ablation_suite(
        cache_dir=str(smoke_cache), ckpt_dir=str(tmp_path), epochs=1, seeds=[42],
        with_test=False, arc_config=ARC, configs=["A1_baseline"],
        train_samples=16, val_samples=8,
    )
    entry = results["A1_baseline"][0]["history"]
    saved = load_checkpoint_file(
        os.path.join(str(tmp_path), f"A1_baseline_{ARC}_s42_history.pt"))

    assert INITIAL_WEIGHTS not in entry and "grad_norm" in entry
    assert INITIAL_WEIGHTS in saved


def _model(run):
    from evaluation.input_ablation import build_model_from_config

    ckpt_dir, _ = run
    ckpt = load_checkpoint_file(os.path.join(str(ckpt_dir), f"{RUN}_latest.pt"))
    return build_model_from_config(ckpt["provenance"]["model_config"], "cpu")
