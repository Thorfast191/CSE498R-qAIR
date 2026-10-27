"""
E8 -- timed smoke run: build a small ARC-Easy cache in a SEPARATE, new
directory and train 1 epoch on it, reporting seconds per stage, so the
cost assumptions in docs/baseline/improvement_plan.md §0.2 can be replaced
by measurements.

    python scripts/smoke_timing.py --cache-dir /content/smoke_cache \\
        --ckpt-dir /content/smoke_ckpt

Stages, timed by wrapping the real calls (the code under test is not
changed):
  generation  HypothesisGenerator.generate_batch_with_flags
  scoring     HypothesisGenerator.score_options      (LLM option log-likelihood)
  embedding   HypothesisEncoder.encode
  model_load  generator + encoder construction
  train_steps Trainer.train minus evaluation and checkpoint/history writes
  eval        validation passes
  checkpoint_io

Refuses the configured real cache directory and any directory that
already holds a cache, so it can never touch or extend the real cache.
The report (JSON) also projects a full ARC-Easy cache build and one
20-epoch run from the measured rates; those projections are [DERIVED],
not measured.
"""

import argparse
import functools
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import training.dataset as dataset_module  # noqa: E402
import training.train as train_module  # noqa: E402
from config import ARC_SPLIT_SIZES, BATCH_SIZE, CACHE_DIR, GEN_SEED  # noqa: E402

STAGES = ("generation", "scoring", "embedding", "model_load",
          "train_steps", "eval", "checkpoint_io")

PROJECTED_EPOCHS = 20      # PD-7


class SmokeDirectoryError(RuntimeError):
    """The smoke run would read or extend a real cache."""


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    """Accumulates wall-clock seconds and call counts per stage."""

    def __init__(self):
        self.seconds = {stage: 0.0 for stage in STAGES}
        self.calls = {stage: 0 for stage in STAGES}
        self._patches = []

    def wrap(self, owner, attribute, stage):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            synchronize()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                synchronize()
                self.seconds[stage] += time.perf_counter() - start
                self.calls[stage] += 1

        setattr(owner, attribute, timed)
        self._patches.append((owner, attribute, original))
        return timed

    def restore(self):
        for owner, attribute, original in reversed(self._patches):
            setattr(owner, attribute, original)
        self._patches.clear()


def check_smoke_directory(cache_dir):
    real = os.path.realpath(CACHE_DIR)

    if os.path.realpath(cache_dir) == real:
        raise SmokeDirectoryError(
            f"{cache_dir} is the configured cache directory (config.CACHE_DIR); "
            f"the smoke run needs a separate, new directory."
        )

    existing = glob.glob(os.path.join(cache_dir, "*.pt"))
    if existing:
        raise SmokeDirectoryError(
            f"{cache_dir} already holds a cache ({os.path.basename(existing[0])}"
            f"{' ...' if len(existing) > 1 else ''}); the smoke run would read or "
            f"extend it. Use a new, empty directory."
        )


def build_cache(timer, cache_dir, arc_config, n_train, n_val, gen_seed):
    generator = dataset_module.HypothesisGenerator
    encoder = dataset_module.HypothesisEncoder

    timer.wrap(generator, "generate_batch_with_flags", "generation")
    timer.wrap(generator, "score_options", "scoring")
    timer.wrap(generator, "__init__", "model_load")
    timer.wrap(encoder, "encode", "embedding")
    timer.wrap(encoder, "__init__", "model_load")

    sizes = {}
    try:
        for split, n in (("train", n_train), ("validation", n_val)):
            ds = dataset_module.QAIRDataset(
                split=split, max_samples=n, cache_dir=cache_dir,
                arc_config=arc_config, gen_seed=gen_seed, allow_build=True,
            )
            sizes[split] = len(ds)
    finally:
        timer.restore()

    return sizes


def train_one_epoch(timer, cache_dir, ckpt_dir, arc_config, n_train, n_val):
    from training.runner import run_training

    trainer = train_module.Trainer

    train_seconds = {"total": 0.0}
    original_train = trainer.train

    @functools.wraps(original_train)
    def timed_train(self, *args, **kwargs):
        synchronize()
        start = time.perf_counter()
        try:
            return original_train(self, *args, **kwargs)
        finally:
            synchronize()
            train_seconds["total"] += time.perf_counter() - start

    timer.wrap(train_module, "evaluate", "eval")
    timer.wrap(trainer, "save_checkpoint", "checkpoint_io")
    timer.wrap(trainer, "save_history", "checkpoint_io")
    trainer.train = timed_train

    try:
        history = run_training(
            cache_dir=cache_dir, ckpt_dir=ckpt_dir, train_samples=n_train,
            val_samples=n_val, epochs=1, patience=None, with_test=False,
            arc_config=arc_config,
        )
    finally:
        trainer.train = original_train
        timer.restore()

    timer.seconds["train_steps"] = (
        train_seconds["total"] - timer.seconds["eval"] - timer.seconds["checkpoint_io"]
    )
    timer.calls["train_steps"] = math.ceil(n_train / BATCH_SIZE)

    return history


def report(timer, sizes, arc_config, history):
    questions = sum(sizes.values())
    steps = timer.calls["train_steps"]

    per_question = {
        stage: timer.seconds[stage] / questions
        for stage in ("generation", "scoring", "embedding")
    }
    per_step = timer.seconds["train_steps"] / steps
    per_eval_question = timer.seconds["eval"] / sizes["validation"]

    full = ARC_SPLIT_SIZES.get(arc_config, {})
    full_questions = sum(full.values())
    full_steps = math.ceil(full.get("train", 0) / BATCH_SIZE)
    epoch_seconds = full_steps * per_step + full.get("validation", 0) * per_eval_question

    manifest = history.get("manifest", {})

    return {
        "arc_config": arc_config,
        "device": manifest.get("device"),
        "device_name": manifest.get("device_name"),
        "quantum_device": manifest.get("quantum_device"),
        "git_commit": manifest.get("git_commit"),
        "dirty": manifest.get("dirty"),
        "questions": sizes,
        "batch_size": BATCH_SIZE,
        "train_steps": steps,
        "seconds": timer.seconds,
        "calls": timer.calls,
        "per_question_seconds": per_question,
        "per_train_step_seconds": per_step,
        "per_eval_question_seconds": per_eval_question,
        "projection_DERIVED": {
            "note": "measured rates x full split sizes; model loading excluded",
            "full_split_sizes": full,
            "cache_build_hours": full_questions * sum(per_question.values()) / 3600,
            "train_steps_per_epoch": full_steps,
            "epoch_minutes": epoch_seconds / 60,
            f"run_{PROJECTED_EPOCHS}_epochs_hours": PROJECTED_EPOCHS * epoch_seconds / 3600,
        },
    }


def run_smoke(cache_dir, ckpt_dir, n_train=20, n_val=10, arc_config="ARC-Easy",
              gen_seed=GEN_SEED, out=None):
    check_smoke_directory(cache_dir)

    timer = StageTimer()
    sizes = build_cache(timer, cache_dir, arc_config, n_train, n_val, gen_seed)
    history = train_one_epoch(timer, cache_dir, ckpt_dir, arc_config, n_train, n_val)

    result = report(timer, sizes, arc_config, history)

    out = out or os.path.join(ckpt_dir, "smoke_timing.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    return result, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", required=True,
                        help="New, empty directory for the smoke cache.")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--arc-config", default="ARC-Easy",
                        choices=["ARC-Challenge", "ARC-Easy", "merged"])
    parser.add_argument("--gen-seed", type=int, default=GEN_SEED)
    parser.add_argument("--out", default=None,
                        help="Report path (default: <ckpt-dir>/smoke_timing.json).")
    args = parser.parse_args(argv)

    result, out = run_smoke(args.cache_dir, args.ckpt_dir, args.train, args.val,
                            args.arc_config, args.gen_seed, args.out)

    print("\n" + "=" * 66)
    print(f"SMOKE TIMING ({result['device_name']}, quantum {result['quantum_device']})")
    print("=" * 66)
    for stage in STAGES:
        print(f"{stage:<14s} {result['seconds'][stage]:9.2f} s  "
              f"({result['calls'][stage]} calls)")
    for stage, value in result["per_question_seconds"].items():
        print(f"{stage + ' / question':<26s} {value:8.3f} s")
    print(f"{'train step':<26s} {result['per_train_step_seconds']:8.3f} s")
    print(f"{'eval / question':<26s} {result['per_eval_question_seconds']:8.3f} s")
    projection = result["projection_DERIVED"]
    print(f"[DERIVED] full cache build ~{projection['cache_build_hours']:.2f} h; "
          f"one epoch ~{projection['epoch_minutes']:.1f} min; "
          f"{PROJECTED_EPOCHS} epochs ~"
          f"{projection[f'run_{PROJECTED_EPOCHS}_epochs_hours']:.2f} h")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
