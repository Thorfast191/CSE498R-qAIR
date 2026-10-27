"""
E5 -- one result file per (arm, seed), and a combined file rebuilt from
all of them, so a second grid invocation (more seeds, another subset of
arms, a Colab session that died half way) adds to the results instead of
overwriting them.

The combined file is refused when its result files come from different
code (fingerprint + uncommitted diff) or different cache files: the A8
failure, where stale runs were mixed into one table. --allow-mixed
overrides; the override and the differing sources are recorded in the
combined file and printed with the results.
"""

import glob
import os

import torch

from training.checkpoint import load_checkpoint_file

RESULT_SUFFIX = "_result.pt"
COMBINED_NAME = "ablation_results.pt"
COMBINED_FORMAT = 2


class MixedResultsError(RuntimeError):
    """Result files in one directory come from different code or caches."""


def result_path(ckpt_dir, run_name):
    return os.path.join(ckpt_dir, f"{run_name}{RESULT_SUFFIX}")


def save_result(ckpt_dir, run_name, name, entry):
    torch.save({"arm": name, "run": run_name, "entry": entry},
               result_path(ckpt_dir, run_name))


def source_of(entry):
    """The code and cache files a result was produced from."""

    manifest = entry.get("history", {}).get("manifest") or {}
    caches = manifest.get("caches") or {}

    return {
        "code_sha256": manifest.get("code_sha256"),
        "code_diff_sha256": manifest.get("code_diff_sha256"),
        "caches": {split: c["file_sha256"] for split, c in sorted(caches.items())},
    }


def combine_results(ckpt_dir, allow_mixed=False):
    """
    {"format", "results": {arm: [entries by seed]}, "allow_mixed",
     "sources": [{source, runs}]} from every result file in ckpt_dir.
    """

    groups = []
    results = {}

    for path in sorted(glob.glob(os.path.join(ckpt_dir, f"*{RESULT_SUFFIX}"))):

        saved = load_checkpoint_file(path)
        entry = saved["entry"]
        source = source_of(entry)

        group = next((g for g in groups if g["source"] == source), None)
        if group is None:
            group = {"source": source, "runs": []}
            groups.append(group)
        group["runs"].append(saved["run"])

        results.setdefault(saved["arm"], []).append(entry)

    for runs in results.values():
        runs.sort(key=lambda e: e["seed"])

    if len(groups) > 1 and not allow_mixed:
        raise MixedResultsError(
            f"result files in {ckpt_dir} come from {len(groups)} different "
            f"code versions or caches:\n" + describe_sources(groups)
            + "\nMove the stale *_result.pt files out, or pass --allow-mixed "
              "(recorded in the combined file and printed with the results)."
        )

    return {
        "format": COMBINED_FORMAT,
        "results": results,
        "allow_mixed": bool(allow_mixed),
        "sources": groups,
    }


def describe_sources(groups):
    lines = []

    for i, g in enumerate(groups):
        code = g["source"]["code_sha256"]
        dirty = " +uncommitted diff" if g["source"]["code_diff_sha256"] else ""
        caches = ", ".join(f"{s} {h[:12]}" for s, h in g["source"]["caches"].items())
        lines.append(
            f"  source {i}: code {code[:12] if code else None}{dirty}; "
            f"caches {caches or None}; runs {', '.join(g['runs'])}"
        )

    return "\n".join(lines)


def mixing_note(combined):
    """The line printed with every combined result."""

    if combined["format"] == 1:
        return "[RESULTS] v46-format file: code version and caches unknown"

    if len(combined["sources"]) <= 1:
        return "[RESULTS] one code version and cache set"

    return ("[RESULTS MIXED -- --allow-mixed] these results come from "
            f"{len(combined['sources'])} code versions or caches:\n"
            + describe_sources(combined["sources"]))


def load_combined(path):
    """
    The combined file in any format. The v46 grid saved the bare
    {arm: [entries]} dict (format 1): no sources are known.
    """

    loaded = load_checkpoint_file(path)

    if loaded.get("format") == COMBINED_FORMAT:
        return loaded

    return {"format": 1, "results": loaded, "allow_mixed": None, "sources": []}
