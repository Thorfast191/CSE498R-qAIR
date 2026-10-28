"""
B6 -- paired item-level statistics: per-seed McNemar, an item-level paired
bootstrap that averages each item over the seeds, Holm within the two
PD-14 families, and the PD-11 verdict (docs/baseline/issues.md B6).

    python -m pytest tests/test_stats_b6.py
"""

import json
import sys

import pytest
import torch

from evaluation import stats

SEEDS = [0, 1, 2]


def test_holm_step_down():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert stats.holm({"x": 0.7, "y": 0.6}) == pytest.approx({"y": 1.0, "x": 1.0})


def test_identical_arms_and_a_clear_win():
    recs = {s: [1, 0, 1, 1, 0, 0, 1, 0] * 10 for s in SEEDS}

    d, lo, hi, p = stats.paired_bootstrap(recs, recs, SEEDS, n_boot=500)
    assert (d, lo, hi) == (0.0, 0.0, 0.0) and p == 1.0

    better = {s: [1] * 80 for s in SEEDS}
    d, lo, hi, p = stats.paired_bootstrap(better, recs, SEEDS, n_boot=500)
    assert d == pytest.approx(0.5) and lo > 0 and p < 0.01


def test_items_not_cells_are_resampled():
    """Averaging over seeds first: a seed-specific flip changes d by 1/(n*S) per cell."""

    base = {s: [0] * 4 for s in SEEDS}
    arm = {0: [1, 0, 0, 0], 1: [1, 0, 0, 0], 2: [0, 0, 0, 0]}

    d, _, _, _ = stats.paired_bootstrap(arm, base, SEEDS, n_boot=200)
    assert d == pytest.approx((2 / 3) / 4)


def test_deterministic_baseline_repeats_and_lengths_are_checked():
    arm = {s: [1, 0, 1, 1] for s in SEEDS}
    assert stats.as_matrix([1, 1, 0, 0], SEEDS).shape == (4, 3)

    with pytest.raises(ValueError, match="equal shapes"):
        stats.paired_bootstrap(arm, [1, 0, 1], SEEDS)
    with pytest.raises(ValueError, match="missing for seeds"):
        stats.paired_bootstrap({0: [1], 1: [1]}, [1], SEEDS)


def test_per_seed_mcnemar_and_direction():
    base = [1, 0, 0, 0, 1, 0, 0, 0]
    arm = {0: [1, 1, 1, 0, 1, 0, 0, 0], 1: [1, 1, 0, 0, 1, 0, 0, 0], 2: [0, 0, 0, 0, 1, 0, 0, 0]}

    r = stats.compare(arm, base, SEEDS, n_boot=200)

    for i, s in enumerate(SEEDS):
        assert r["per_seed"][s]["mcnemar"] == stats.mcnemar(base, arm[s])
    assert r["same_direction"] is False                     # seed 2 is worse


def make_result(p, d, same):
    return {"p": p, "p_holm": p, "diff": d, "same_direction": same}


def test_pd11_needs_every_condition():
    win = {key: make_result(0.01, 0.05, True) for key in stats.PRIMARY_FAMILY}
    assert stats.pd11_verdict(win)

    for bad in (make_result(0.2, 0.05, True), make_result(0.01, -0.05, True),
                make_result(0.01, 0.05, False)):
        assert not stats.pd11_verdict({**win, stats.PRIMARY_FAMILY[1]: bad})
    assert not stats.pd11_verdict({stats.PRIMARY_FAMILY[0]: win[stats.PRIMARY_FAMILY[0]]})

    raw_only = dict(make_result(0.03, 0.05, True), p_holm=0.06)    # passes raw, not Holm
    assert not stats.pd11_verdict({**win, stats.PRIMARY_FAMILY[0]: raw_only})


def test_families_are_the_pd14_ones():
    assert stats.PRIMARY_FAMILY == (("F0_full_with_llm_prior", "llm"),
                                    ("F0_full_with_llm_prior", "letter"))
    assert len(stats.SECONDARY_FAMILY) == 6
    assert not set(stats.PRIMARY_FAMILY) & set(stats.SECONDARY_FAMILY)


def simulated_arm(g, q, kappa):
    fixed = (torch.rand(q.shape, generator=g) < q).float()
    out = {}
    for s in SEEDS:
        fresh = (torch.rand(q.shape, generator=g) < q).float()
        keep = (torch.rand(q.shape, generator=g) < kappa).float()
        out[s] = (keep * fixed + (1 - keep) * fresh).tolist()
    return out


def test_pooling_over_rejects_and_the_item_bootstrap_does_not():
    """Dossier B6 verification: identical arms, item-driven seed correlation."""

    g = torch.Generator().manual_seed(1)
    sims, pooled, boot = 1000, 0, 0
    for _ in range(sims):
        q = 0.2 + 0.7 * torch.rand(200, generator=g)
        a, b = simulated_arm(g, q, 0.8), simulated_arm(g, q, 0.8)
        pooled += stats.mcnemar(sum(a.values(), []), sum(b.values(), []))[2] < 0.05
        boot += stats.paired_bootstrap(a, b, SEEDS, n_boot=500)[3] < 0.05

    assert pooled / sims > 0.12
    assert 0.02 <= boot / sims <= 0.08


def grid(accs, n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {arm: [{"seed": s, "test_records": (torch.rand(n, generator=g) < acc).int().tolist()}
                  for s in SEEDS] for arm, acc in accs.items()}


def test_success_report_runs_both_families(capsys):
    results = grid({"F0_full_with_llm_prior": 0.9, "A3_persistent": 0.6,
                    "P0_classical_collapse": 0.6, "P1_phase_zero": 0.6})
    g = torch.Generator().manual_seed(9)
    baselines = {m: (torch.rand(60, generator=g) < 0.4).int().tolist() for m in ("llm", "letter")}

    report = stats.success_report(results, baselines, n_boot=500)
    out = capsys.readouterr().out

    assert set(report["primary"]) == set(stats.PRIMARY_FAMILY)
    assert set(report["secondary"]) == set(stats.SECONDARY_FAMILY)
    assert report["pd11"] is True
    assert "Holm over 2" in out and "Holm over 6" in out
    raw = {k: r["p"] for k, r in report["primary"].items()}
    assert {k: r["p_holm"] for k, r in report["primary"].items()} == stats.holm(raw)


def test_cli_reads_baseline_files(tmp_path, monkeypatch, capsys):
    results = grid({"F0_full_with_llm_prior": 0.7})
    path = tmp_path / "ablation_results.pt"
    torch.save({"format": 2, "results": results, "allow_mixed": False, "sources": []}, path)
    for mode in ("llm", "letter"):
        (tmp_path / f"{mode}_test.json").write_text(json.dumps(
            {"mode": mode, "split": "test", "records": [1, 0] * 30}))

    monkeypatch.setattr(sys, "argv", ["stats", "--results", str(path),
                                      "--baselines-dir", str(tmp_path)])
    stats.main()

    out = capsys.readouterr().out
    assert "PRIMARY family" in out and "PD-11 success criterion met" in out


def test_validation_records_are_refused_as_test(tmp_path):
    (tmp_path / "llm_test.json").write_text(json.dumps({"split": "validation", "records": []}))
    with pytest.raises(ValueError, match="not 'test'"):
        stats.load_baseline_records(str(tmp_path))
