import argparse

from config import (
    CACHE_DIR, CKPT_DIR, SEED, SEEDS, ARC_CONFIG, DETERMINISTIC,
    DETERMINISTIC_WARN_ONLY, EPOCHS,
)
from training.seed import set_seed
from training.runner import run_training
from training.ablations import run_ablation_suite, ABLATIONS


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "ablation"]
    )

    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Seed for --mode train. Ablations sweep config.SEEDS.",
    )

    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS),
        help="Seeds for --mode ablation (default: config.SEEDS).",
    )

    parser.add_argument(
        "--configs", type=str, nargs="+", default=None,
        choices=list(ABLATIONS),
        help="Subset of the ablation grid to run (default: all).",
    )

    parser.add_argument(
        "--arc-config", type=str, default=ARC_CONFIG,
        choices=["ARC-Challenge", "ARC-Easy", "merged"],
    )

    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Epoch cap (default config.EPOCHS = 20, PD-7). The capstone "
             "configuration: --arc-config ARC-Challenge --epochs 8 "
             "--train-samples 80 --val-samples 20.",
    )

    parser.add_argument(
        "--train-samples", type=int, default=None,
        help="Use only the first N train questions of the cache "
             "(default: all). The capstone used 80.",
    )

    parser.add_argument(
        "--val-samples", type=int, default=None,
        help="Use only the first N validation questions of the cache "
             "(default: all). The capstone used 20.",
    )

    parser.add_argument(
        "--cache-dir", type=str, default=CACHE_DIR,
        help="Directory of the hypothesis caches (default config.CACHE_DIR).",
    )

    parser.add_argument(
        "--ckpt-dir", type=str, default=CKPT_DIR,
        help="Directory for checkpoints and result files (default "
             "config.CKPT_DIR). A run resumes from the files here.",
    )

    parser.add_argument(
        "--no-test", action="store_true",
        help="Skip held-out test evaluation (validation only).",
    )

    parser.add_argument(
        "--deterministic-warn-only", action="store_true",
        default=DETERMINISTIC_WARN_ONLY,
        help="Warn instead of raising when an op has no deterministic "
             "implementation (recorded in the checkpoint provenance).",
    )

    parser.add_argument(
        "--ignore-pin-mismatch", action="store_true",
        help="Train even if installed versions differ from requirements.txt "
             "(recorded in the provenance and printed with every result).",
    )

    parser.add_argument(
        "--allow-mixed", action="store_true",
        help="--mode ablation: build ablation_results.pt even if the result "
             "files come from different code or caches (recorded in the file "
             "and printed with the results).",
    )

    args = parser.parse_args()

    # set_seed lived in training/seed.py for the project's whole history
    # and was never called from anywhere, so no run was reproducible.
    set_seed(args.seed, deterministic=DETERMINISTIC,
             warn_only=args.deterministic_warn_only)

    if args.mode == "train":

        run_training(
            cache_dir=args.cache_dir,
            ckpt_dir=args.ckpt_dir,
            seed=args.seed,
            epochs=args.epochs,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            with_test=not args.no_test,
            arc_config=args.arc_config,
            deterministic_warn_only=args.deterministic_warn_only,
            ignore_pin_mismatch=args.ignore_pin_mismatch,
        )

    elif args.mode == "ablation":

        run_ablation_suite(
            cache_dir=args.cache_dir,
            ckpt_dir=args.ckpt_dir,
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            with_test=not args.no_test,
            arc_config=args.arc_config,
            configs=args.configs,
            deterministic_warn_only=args.deterministic_warn_only,
            ignore_pin_mismatch=args.ignore_pin_mismatch,
            allow_mixed=args.allow_mixed,
        )


if __name__ == "__main__":

    main()
