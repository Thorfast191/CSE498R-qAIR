"""
Statistical reporting for ablation comparisons.

ARC-Easy's validation split has 570 examples, so a single run's accuracy
carries a standard error of roughly

    sqrt(0.33 * 0.67 / 570) = 0.020

i.e. +/- 2.0 points at one sigma, +/- 3.9 at two. Most of the gaps this
project's ablation grid produces are smaller than that. Reporting a
single number per config -- as every result here did before the audit --
cannot distinguish an effect from seed noise.

Provides:
  * bootstrap_ci      -- accuracy with a percentile bootstrap CI
  * mcnemar           -- exact paired test between two arms (one seed)
  * paired_bootstrap  -- item-level paired bootstrap over seeds (B6)
  * holm              -- Holm step-down adjustment within a family
  * compare_arms      -- exploratory arm-vs-arm table over ablation_results.pt
  * success_report    -- the PD-11 / PD-14 families against the LLM baselines

Paired tests matter here: two ablation arms see the SAME examples, so
McNemar on the disagreement cells is far more powerful than comparing
two independent proportions.

B6: records are never pooled across seeds. Treating (item, seed) cells as
independent underestimates the variance by 1 + (S - 1) rho (dossier B6),
so McNemar is run per seed, and the bootstrap resamples ITEMS, averaging
each item over the seeds inside every resample.

PD-14 (provisional, docs/baseline/supervisor_decisions.md): two Holm
families. Primary (the PD-11 success criterion): F0 vs {LLM option
likelihood, LLM letter pick}, Holm over 2. Secondary: {A3 (v46), P0, P1} x
{the same two baselines}, Holm over 6, reported but not used for the
main claim.
"""

import argparse
import json
import math
import os

import torch

from training.results import load_combined, mixing_note


def bootstrap_ci(records, n_boot=10000, alpha=0.05, seed=0):
    """
    records: iterable of 0/1 per-example correctness.
    Returns (accuracy, lo, hi) at the (1-alpha) level.
    """

    x = torch.tensor(records, dtype=torch.float)

    n = x.numel()

    g = torch.Generator().manual_seed(seed)

    idx = torch.randint(0, n, (n_boot, n), generator=g)

    boots = x[idx].mean(dim=1)

    lo = torch.quantile(boots, alpha / 2).item()
    hi = torch.quantile(boots, 1 - alpha / 2).item()

    return x.mean().item(), lo, hi


def mcnemar(records_a, records_b):
    """
    Exact McNemar test on paired per-example correctness.

    b = A right, B wrong;  c = A wrong, B right.
    Under H0 the split of the (b + c) discordant pairs is Binomial(n, 0.5).
    Returns (b, c, two_sided_p).
    """

    a = torch.tensor(records_a, dtype=torch.bool)
    b_ = torch.tensor(records_b, dtype=torch.bool)

    if a.numel() != b_.numel():
        raise ValueError(
            f"paired test needs equal-length records, got "
            f"{a.numel()} and {b_.numel()}"
        )

    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())

    n = b + c

    if n == 0:
        return b, c, 1.0

    k = min(b, c)

    # Exact two-sided binomial tail.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)

    return b, c, min(1.0, 2 * tail)


# The comparison families (PD-14). Baselines are evaluation/baselines.py modes.
PRIMARY_ARM = "F0_full_with_llm_prior"
LLM_BASELINES = ("llm", "letter")
BASELINE_LABELS = {"llm": "LLM option likelihood", "letter": "LLM letter pick"}
PRIMARY_FAMILY = tuple((PRIMARY_ARM, b) for b in LLM_BASELINES)
SECONDARY_FAMILY = tuple((arm, b) for arm in ("A3_persistent", "P0_classical_collapse",
                                              "P1_phase_zero") for b in LLM_BASELINES)

N_BOOT = 10000
BOOT_SEED = 0


def seed_records(runs):
    """{seed: test_records} of one arm's grid entries."""

    return {r["seed"]: r["test_records"] for r in runs if "test_records" in r}


def as_matrix(by_seed, seeds):
    """(items, seeds) float matrix; a deterministic baseline (a plain list) repeats."""

    if isinstance(by_seed, dict):
        missing = [s for s in seeds if s not in by_seed]
        if missing:
            raise ValueError(f"records missing for seeds {missing}")
        cols = [by_seed[s] for s in seeds]
    else:
        cols = [by_seed] * len(seeds)

    lengths = {len(c) for c in cols}
    if len(lengths) != 1:
        raise ValueError(f"paired records need equal lengths, got {sorted(lengths)}")
    return torch.tensor(cols, dtype=torch.float).T


def paired_bootstrap(arm, base, seeds, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    """
    B6. d = mean over items of (mean over seeds of arm_is - base_is).
    Items are resampled with replacement; each resampled item keeps all
    its seeds. Returns (d, lo, hi, p): the percentile CI and the two-sided
    p of the centred (shift) bootstrap, p = (1 + #{|d* - d| >= |d|}) /
    (n_boot + 1). Fixed before any result (PD-14).
    """

    a, b = as_matrix(arm, seeds), as_matrix(base, seeds)
    if a.shape != b.shape:
        raise ValueError(f"paired records need equal shapes, got {tuple(a.shape)} "
                         f"and {tuple(b.shape)}")

    per_item = (a - b).mean(dim=1)
    n = per_item.numel()
    d = per_item.mean().item()

    g = torch.Generator().manual_seed(seed)
    boots = per_item[torch.randint(0, n, (n_boot, n), generator=g)].mean(dim=1)

    lo = torch.quantile(boots, alpha / 2).item()
    hi = torch.quantile(boots, 1 - alpha / 2).item()
    p = (1 + int(((boots - d).abs() >= abs(d) - 1e-12).sum())) / (n_boot + 1)
    return d, lo, hi, min(1.0, p)


def holm(pvalues):
    """Holm step-down adjusted p-values: {key: p} -> {key: p_adj}."""

    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, adjusted = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def compare(arm, base, seeds, n_boot=N_BOOT):
    """One paired comparison: bootstrap over items plus per-seed McNemar."""

    a, b = as_matrix(arm, seeds), as_matrix(base, seeds)
    d, lo, hi, p = paired_bootstrap(arm, base, seeds, n_boot=n_boot)
    per_seed = {s: {"acc_arm": a[:, i].mean().item(), "acc_base": b[:, i].mean().item(),
                    "mcnemar": mcnemar(b[:, i].tolist(), a[:, i].tolist())}
                for i, s in enumerate(seeds)}
    return {"diff": d, "ci": (lo, hi), "p": p, "per_seed": per_seed,
            "acc_arm": a.mean().item(), "acc_base": b.mean().item(),
            "same_direction": all(v["acc_arm"] > v["acc_base"] for v in per_seed.values())}


def run_family(family, arms, baselines, seeds, n_boot=N_BOOT):
    """Comparisons of one family with Holm-adjusted p (missing pairs skipped)."""

    rows = {(arm, base): compare(arms[arm], baselines[base], seeds, n_boot)
            for arm, base in family if arm in arms and base in baselines}
    for key, p_adj in holm({k: r["p"] for k, r in rows.items()}).items():
        rows[key]["p_holm"] = p_adj
    return rows


def pd11_verdict(primary, alpha=0.05):
    """PD-11: F0 beats BOTH LLM baselines (Holm p < alpha, d > 0, all seeds)."""

    complete = len(primary) == len(PRIMARY_FAMILY)
    return complete and all(r["p_holm"] < alpha and r["diff"] > 0 and r["same_direction"]
                            for r in primary.values())


def print_family(title, rows):
    print(f"\n{title}")
    print(f"{'arm':<26s} {'vs':<22s} {'acc':>7s} {'base':>7s} {'diff':>8s} "
          f"{'95% CI':>18s} {'p':>8s} {'p Holm':>8s} {'all seeds':>9s}  per-seed McNemar p")
    for (arm, base), r in rows.items():
        mc = " ".join(f"s{s}:{v['mcnemar'][2]:.3g}" for s, v in r["per_seed"].items())
        print(f"{arm:<26s} {BASELINE_LABELS.get(base, base):<22s} {r['acc_arm']:>7.4f} "
              f"{r['acc_base']:>7.4f} {r['diff']:>+8.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] "
              f"{r['p']:>8.4f} {r['p_holm']:>8.4f} {str(r['same_direction']):>9s}  {mc}")


def success_report(results, baselines, n_boot=N_BOOT):
    """
    results: {arm: [grid entries]}; baselines: {mode: per-item test records}.
    Prints both PD-14 families and the PD-11 verdict; returns them.
    """

    arms = {arm: seed_records(runs) for arm, runs in results.items()}
    arms = {arm: recs for arm, recs in arms.items() if recs}
    if PRIMARY_ARM not in arms:
        raise ValueError(f"{PRIMARY_ARM} has no test records")
    seeds = sorted(arms[PRIMARY_ARM])

    primary = run_family(PRIMARY_FAMILY, arms, baselines, seeds, n_boot)
    secondary = run_family(SECONDARY_FAMILY, arms, baselines, seeds, n_boot)
    verdict = pd11_verdict(primary)

    print("=" * 120)
    print(f"Test accuracy, seeds {seeds}; item-level paired bootstrap ({n_boot} resamples, "
          f"seeds averaged per item), Holm within each family (PD-14)")
    print("=" * 120)
    print_family("PRIMARY family (PD-11 success criterion, Holm over "
                 f"{len(PRIMARY_FAMILY)})", primary)
    print_family("SECONDARY family (reported, not used for the main claim, Holm over "
                 f"{len(SECONDARY_FAMILY)})", secondary)
    print(f"\nPD-11 success criterion met: {verdict}  (F0 beats both LLM baselines: "
          f"Holm-adjusted p < 0.05, positive difference, and higher accuracy in every seed)")
    return {"primary": primary, "secondary": secondary, "pd11": verdict}


def compare_arms(results, baseline=None, alpha=0.05, n_boot=N_BOOT):
    """
    Exploratory arm-vs-arm table (no family, no correction): each arm
    against `baseline` with the item-level paired bootstrap and per-seed
    McNemar. Seeds are never pooled (B6).
    """

    arms = {name: seed_records(runs) for name, runs in results.items()}
    arms = {name: recs for name, recs in arms.items() if recs}

    if not arms:
        print("No test_records found. Re-run the suite with with_test=True.")
        return

    if baseline is None:
        baseline = "A1_baseline" if "A1_baseline" in arms else next(iter(arms))
    seeds = sorted(arms[baseline])

    print("\n" + "=" * 100)
    print(f"Exploratory: test accuracy vs {baseline}, seeds {seeds}, item-level paired "
          f"bootstrap and per-seed McNemar (uncorrected)")
    print("=" * 100)
    print(f"{'config':<34s} {'acc':>7s} {'diff':>8s} {'95% CI':>18s} {'p':>8s}  per-seed McNemar p")

    for name, recs in arms.items():
        if name == baseline:
            acc = as_matrix(recs, seeds).mean().item()
            print(f"{name:<34s} {acc:>7.4f} {'--':>8s}")
            continue
        r = compare(recs, arms[baseline], seeds, n_boot)
        mc = " ".join(f"s{s}:{v['mcnemar'][2]:.3g}" for s, v in r["per_seed"].items())
        star = " *" if r["p"] < alpha else ""
        print(f"{name:<34s} {r['acc_arm']:>7.4f} {r['diff']:>+8.4f} "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] {r['p']:>8.4f}{star}  {mc}")

    print("-" * 100)
    print("Exploratory, uncorrected. The pre-registered claims are in success_report "
          "(--baselines-dir), with Holm within the PD-14 families.")


def load_baseline_records(baselines_dir, split="test"):
    """{mode: records} from evaluation/baselines.py --out-dir files."""

    out = {}
    for mode in LLM_BASELINES:
        path = os.path.join(baselines_dir, f"{mode}_{split}.json")
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            if saved["split"] != split:
                raise ValueError(f"{path} holds split {saved['split']!r}, not {split!r}")
            out[mode] = saved["records"]
    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="ckpt/ablation_results.pt")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Reference arm of the exploratory table.")
    parser.add_argument("--baselines-dir", type=str, default=None,
                        help="evaluation/baselines.py --out-dir with llm_test.json and "
                             "letter_test.json: run the PD-11 / PD-14 families.")
    args = parser.parse_args()

    combined = load_combined(args.results)

    print(mixing_note(combined))

    if args.baselines_dir:
        success_report(combined["results"], load_baseline_records(args.baselines_dir))

    compare_arms(combined["results"], baseline=args.baseline)


if __name__ == "__main__":
    main()
