"""
E6 -- write the manifest of a finished cache build (Step 6, Section B):
commit, environment, and for every split the cache file's sha256, header
and generation provenance.

    python scripts/cache_manifest.py --cache-dir <cache> --out <manifest.json>

Reads the caches only; an incomplete cache raises (training/dataset.py).
The cache-quality blocks the loader prints are suppressed here, so this
step never shows test-split numbers; the gate runs on train and
validation only (docs/HANDOFF.md, Step 6 decision 3).
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ARC_CONFIG  # noqa: E402
from training.dataset import QAIRDataset  # noqa: E402
from training.manifest import build_cache_manifest, write_manifest  # noqa: E402


def load_quietly(split, cache_dir, arc_config):
    with contextlib.redirect_stdout(io.StringIO()):
        return QAIRDataset(split=split, cache_dir=cache_dir, arc_config=arc_config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--arc-config", default=ARC_CONFIG,
                        choices=["ARC-Challenge", "ARC-Easy", "merged"])
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    datasets = {split: load_quietly(split, args.cache_dir, args.arc_config)
                for split in args.splits}
    manifest = build_cache_manifest(datasets, args.arc_config)
    write_manifest(args.out, manifest)

    print(f"commit {manifest['git_commit']}  dirty {manifest['dirty']}")
    for split, cache in manifest["caches"].items():
        header = cache["header"] or {}
        print(f"{split:<11s} n={cache['n_samples']:<5d} complete={header.get('complete')}  "
              f"gen_seed={header.get('gen_seed')}  sha256={cache['file_sha256'][:16]}...")
    print(f"Manifest: {args.out}")
    return manifest


if __name__ == "__main__":
    main()
