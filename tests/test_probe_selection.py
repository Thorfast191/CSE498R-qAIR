"""
B7 -- the probe "data ceiling" chooses its epoch count on a holdout of the
train split and evaluates validation exactly once
(docs/baseline/issues.md B7).

    python -m pytest tests/test_probe_selection.py
"""

import pytest
import torch

from evaluation import baselines
from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)

N_FEAT = 1


def problem(n_train=40, n_val=12, n_opt=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_train + n_val, n_opt, N_FEAT * baselines.D, generator=g)
    y = torch.randint(0, n_opt, (n_train + n_val,), generator=g)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_holdout_split_is_fixed_and_disjoint():
    fit, hold = baselines.train_holdout_split(40)
    fit2, hold2 = baselines.train_holdout_split(40)

    assert torch.equal(fit, fit2) and torch.equal(hold, hold2)
    assert len(hold) == 8 and set(fit.tolist()).isdisjoint(hold.tolist())
    assert sorted(fit.tolist() + hold.tolist()) == list(range(40))


def test_validation_labels_do_not_change_the_chosen_epochs():
    Xtr, ytr, Xva, yva = problem()
    a = baselines.fit_probe(N_FEAT, 0, Xtr, ytr, Xva, yva, epochs=8, lr=3e-3)
    b = baselines.fit_probe(N_FEAT, 0, Xtr, ytr, Xva, (yva + 1) % 4, epochs=8, lr=3e-3)

    assert a["epochs"] == b["epochs"] and a["holdout_acc"] == b["holdout_acc"]
    assert 1 <= a["epochs"] <= 8


def test_validation_is_evaluated_exactly_once(monkeypatch):
    Xtr, ytr, Xva, yva = problem()
    real = baselines._accuracy
    seen = []

    def accuracy(model, X, y):
        seen.append(X is Xva)
        return real(model, X, y)

    monkeypatch.setattr(baselines, "_accuracy", accuracy)
    baselines.fit_probe(N_FEAT, 0, Xtr, ytr, Xva, yva, epochs=5, lr=3e-3)

    assert seen.count(True) == 1 and seen[-1] is True
    assert len(seen) == 5 + 1                    # one holdout pass per epoch, then validation


def test_final_probe_trains_on_all_of_train_for_the_chosen_epochs(monkeypatch):
    Xtr, ytr, Xva, yva = problem()
    real = baselines._probe_epoch
    sizes = []

    def epoch(model, opt, X, y):
        sizes.append(len(y))
        return real(model, opt, X, y)

    monkeypatch.setattr(baselines, "_probe_epoch", epoch)
    result = baselines.fit_probe(N_FEAT, 0, Xtr, ytr, Xva, yva, epochs=6, lr=3e-3)

    assert sizes[:6] == [32] * 6
    assert sizes[6:] == [40] * result["epochs"]


def test_run_probes_reports_every_feature_set(smoke_cache, capsys):
    out = baselines.run_probes(str(smoke_cache), epochs=3, arc_config=ARC)

    assert set(out) == set(baselines.FEATURES)
    assert all(set(r) == {"val_acc", "epochs", "holdout_acc"} for r in out.values())
    assert "validation evaluated once" in capsys.readouterr().out


def test_first_best_holdout_epoch_is_chosen(monkeypatch):
    Xtr, ytr, Xva, yva = problem()
    scripted = iter([0.1, 0.5, 0.3, 0.5, 0.2, 0.9])        # 5 holdout passes, then validation

    monkeypatch.setattr(baselines, "_accuracy", lambda model, X, y: next(scripted))
    result = baselines.fit_probe(N_FEAT, 0, Xtr, ytr, Xva, yva, epochs=5, lr=3e-3)

    assert (result["epochs"], result["holdout_acc"], result["val_acc"]) == (2, 0.5, 0.9)
