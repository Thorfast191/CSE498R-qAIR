"""
Step 7 -- the Tier 0 ARC-Easy runs: v46/A3, F0, P0, P1 x seeds 0/1/2 under
the PD-7 / PD-10 protocol (at most 20 epochs, patience 5, deterministic),
plus the LLM baselines on test, in Colab sessions of at most ~4 h on a T4.

    python scripts/step7_runs.py verify-cache --cache-dir C --manifest M
    python scripts/step7_runs.py status --ckpt-dir D --baselines-dir B
    python scripts/step7_runs.py train --session 1 --cache-dir C --ckpt-dir D --manifest M
    python scripts/step7_runs.py baselines --cache-dir C --out-dir B --manifest M

Sessions 1-4 each train one arm under the three seeds, the primary arm F0
first (PD-11); session 5 runs the baselines. Every run goes through
`main.py --mode ablation`, which resumes an interrupted run from its last
checkpoint (A3/A4/A11) and evaluates test once, from the
validation-selected checkpoint (B3). A run with a result file is finished
and never started again, so its test split is not evaluated twice.

Before anything trains, the cache files are hashed and compared with the
canonical cache-build manifest (commit 44fc12a); any difference stops.
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import config  # noqa: E402
from config import cache_filename  # noqa: E402
from training.checkpoint import load_checkpoint_file, run_key  # noqa: E402
from training.manifest import file_sha256  # noqa: E402
from training.results import result_path  # noqa: E402

ARC_CONFIG = "ARC-Easy"
SEEDS = (0, 1, 2)

# Session -> arm. F0 is the PD-11 primary arm, so it runs first.
SESSIONS = {
    1: "F0_full_with_llm_prior",
    2: "A3_persistent",
    3: "P0_classical_collapse",
    4: "P1_phase_zero",
}
BASELINE_SESSION = 5
BASELINE_MODES = ("llm", "letter", "vote")

# Canonical Tier 0 cache: commit 44fc12a, session 20261028_135552_44fc12a.
CANONICAL_SHA256_PREFIX = {
    "train": "4bc0031289a03e4f",
    "validation": "0760535c445ad6dc",
    "test": "5faa6d62c82612d6",
}

# The pre-registered protocol (PD-7 / PD-10) and the E12 engine.
PROTOCOL = {
    "EPOCHS": 20,
    "PATIENCE": 5,
    "DETERMINISTIC": True,
    "DETERMINISTIC_WARN_ONLY": False,
    "QUANTUM_SIMULATOR": "torch",
}


class Step7Error(RuntimeError):
    """The cache or the configuration is not the pre-registered one."""


def check_protocol(cfg=config):
    wrong = {k: getattr(cfg, k, None) for k, v in PROTOCOL.items() if getattr(cfg, k, None) != v}
    if wrong:
        raise Step7Error(f"config differs from the Step 7 protocol {PROTOCOL}: {wrong}")


def verify_cache(cache_dir, manifest_path, arc_config=ARC_CONFIG):
    """{split: sha256} of the cache files, after checking them against the manifest."""

    with open(manifest_path) as f:
        manifest = json.load(f)

    hashes, problems = {}, []
    for split, prefix in CANONICAL_SHA256_PREFIX.items():
        recorded = manifest["caches"][split]["file_sha256"]
        if not recorded.startswith(prefix):
            problems.append(f"{split}: the manifest records {recorded[:16]}, "
                            f"not the canonical {prefix}")
            continue

        path = os.path.join(cache_dir, cache_filename(split, arc_config))
        if not os.path.exists(path):
            problems.append(f"{split}: {path} does not exist")
            continue

        hashes[split] = file_sha256(path)
        if hashes[split] != recorded:
            problems.append(f"{split}: {path} hashes to {hashes[split][:16]}, "
                            f"the manifest records {recorded[:16]}")

    if problems:
        raise Step7Error("the cache is not the canonical Tier 0 cache:\n  "
                         + "\n  ".join(problems))
    return hashes


def run_status(ckpt_dir, arm, seed, epochs=PROTOCOL["EPOCHS"]):
    """One run's state, read from its result file or last checkpoint."""

    run = run_key(arm, ARC_CONFIG, seed)
    row = {"arm": arm, "seed": seed, "run": run, "epochs_done": None,
           "stopped_early": None, "best_val_acc": None, "test_acc": None}

    if os.path.exists(result_path(ckpt_dir, run)):
        entry = load_checkpoint_file(result_path(ckpt_dir, run))["entry"]
        history = entry.get("history", {})
        test_from = entry.get("test_from") or {}
        row.update(state="finished", best_val_acc=entry.get("best_val_acc"),
                   test_acc=entry.get("test_acc"), epochs_done=len(history.get("loss", [])),
                   test_epoch=test_from.get("epoch"))
        return row

    latest = os.path.join(ckpt_dir, f"{run}_latest.pt")
    if not os.path.exists(latest):
        row["state"] = "not started"
        return row

    ckpt = load_checkpoint_file(latest)
    done = ckpt["epoch"] + 1
    trained = ckpt.get("stopped_early") or done >= epochs
    row.update(state="trained, test pending" if trained else "in progress",
               epochs_done=done, stopped_early=bool(ckpt.get("stopped_early")),
               best_val_acc=ckpt.get("best_acc"))
    return row


def baseline_status(out_dir):
    """{mode: accuracy or None} of the test baseline files."""

    found = {}
    for mode in BASELINE_MODES:
        path = os.path.join(out_dir, f"{mode}_test.json")
        if os.path.exists(path):
            with open(path) as f:
                found[mode] = json.load(f)["accuracy"]
        else:
            found[mode] = None
    return found


def status(ckpt_dir, baselines_dir):
    rows = [dict(run_status(ckpt_dir, arm, seed), session=session)
            for session, arm in SESSIONS.items() for seed in SEEDS]
    baselines = baseline_status(baselines_dir)

    pending = [r["session"] for r in rows if r["state"] != "finished"]
    if pending:
        next_session = min(pending)
    elif any(v is None for v in baselines.values()):
        next_session = BASELINE_SESSION
    else:
        next_session = None
    return rows, baselines, next_session


def format_status(rows, baselines, next_session):
    def num(x):
        return "-" if x is None else f"{x:.4f}"

    lines = [f"{'session':<8s} {'arm':<24s} {'seed':>4s}  {'state':<22s} "
             f"{'epochs':>8s} {'best val':>8s} {'test':>8s}"]
    for r in rows:
        epochs = "-" if r["epochs_done"] is None else str(r["epochs_done"])
        if r["state"] == "finished" and r.get("test_epoch") is not None:
            epochs += f" (b{r['test_epoch']})"
        lines.append(f"{r['session']:<8d} {r['arm']:<24s} {r['seed']:>4d}  {r['state']:<22s} "
                     f"{epochs:>8s} {num(r['best_val_acc']):>8s} {num(r['test_acc']):>8s}")
    lines.append(f"{BASELINE_SESSION:<8d} baselines on test: " + ", ".join(
        f"{m} {'not run' if a is None else f'{a:.4f}'}" for m, a in baselines.items()))
    lines.append("epochs (bN): epochs trained, and the epoch whose checkpoint test was read from.")
    if next_session is None:
        lines.append("ALL DONE: run the stats cell.")
    else:
        lines.append(f"NEXT: session {next_session}.")
    return "\n".join(lines)


def train_command(arm, seed, cache_dir, ckpt_dir, python=sys.executable):
    return [python, "main.py", "--mode", "ablation", "--configs", arm,
            "--seeds", str(seed), "--arc-config", ARC_CONFIG,
            "--cache-dir", cache_dir, "--ckpt-dir", ckpt_dir]


def baseline_command(mode, cache_dir, out_dir, python=sys.executable):
    return [python, "-m", "evaluation.baselines", "--mode", mode, "--split", "test",
            "--cache-dir", cache_dir, "--arc-config", ARC_CONFIG, "--out-dir", out_dir]


def call(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def train_session(session, cache_dir, ckpt_dir, manifest_path, runner=call):
    """Trains the session's unfinished runs, resuming any interrupted one."""

    if session not in SESSIONS:
        raise Step7Error(f"training sessions are {sorted(SESSIONS)}; "
                         f"session {BASELINE_SESSION} is the baselines")
    check_protocol()
    verify_cache(cache_dir, manifest_path)

    arm = SESSIONS[session]
    for seed in SEEDS:
        row = run_status(ckpt_dir, arm, seed)
        if row["state"] == "finished":
            print(f"[DONE] {row['run']}: test {row['test_acc']}; not run again.")
            continue
        runner(train_command(arm, seed, cache_dir, ckpt_dir))


def run_baselines(cache_dir, out_dir, manifest_path, runner=call):
    """Each test baseline once; a baseline whose file exists is skipped."""

    verify_cache(cache_dir, manifest_path)

    for mode, acc in baseline_status(out_dir).items():
        if acc is not None:
            print(f"[DONE] {mode}_test.json: {acc:.4f}; not run again.")
            continue
        runner(baseline_command(mode, cache_dir, out_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-cache")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--manifest", required=True)

    p = sub.add_parser("status")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--baselines-dir", required=True)

    p = sub.add_parser("train")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--manifest", required=True)

    p = sub.add_parser("baselines")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)

    args = parser.parse_args(argv)

    if args.command == "verify-cache":
        for split, sha in verify_cache(args.cache_dir, args.manifest).items():
            print(f"[CACHE OK] {split:<10s} {sha}")
    elif args.command == "status":
        os.makedirs(args.ckpt_dir, exist_ok=True)
        print(format_status(*status(args.ckpt_dir, args.baselines_dir)))
    elif args.command == "train":
        os.makedirs(args.ckpt_dir, exist_ok=True)
        train_session(args.session, args.cache_dir, args.ckpt_dir, args.manifest)
    elif args.command == "baselines":
        run_baselines(args.cache_dir, args.out_dir, args.manifest)
        for mode, acc in baseline_status(args.out_dir).items():
            print(f"[BASELINE test] {mode:<7s} {acc:.4f}")


if __name__ == "__main__":
    main()
