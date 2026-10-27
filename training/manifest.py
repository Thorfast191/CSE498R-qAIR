"""
E6 -- one manifest per run: everything needed to trace a reported number
back to the code, data, environment and configuration that produced it.

Written as {run}_manifest.json (plain JSON, readable without torch) when a
run starts, and stored in the run's history. It is a record: which fields
must match for a resume is decided by the provenance
(training/checkpoint.py), not here. It holds no timestamp, so two
same-seed runs (and a resumed one) keep bit-identical histories (E10).
"""

import hashlib
import json
import os

import torch

from training.dataset import header_path

MANIFEST_VERSION = 1

DATASET = "allenai/ai2_arc"


def file_sha256(path, chunk=1 << 20):

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)

    return digest.hexdigest()


def cache_record(dataset):
    """The cache file a split was read from: path, hash, header, provenance."""

    path = dataset.cache_path
    header = None

    if os.path.exists(header_path(path)):
        with open(header_path(path)) as f:
            header = json.load(f)

    return {
        "path": os.path.abspath(path),
        "file_sha256": file_sha256(path),
        "n_samples": len(dataset),
        "hypotheses_sha256": dataset.hypotheses_sha256(),
        "header": header,
        # gen_seed, generator + revision, gen_config, prompt_sha256 and the
        # device / library segments the cache was built in.
        "generation": dataset.provenance,
    }


def device_name(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return str(device)


def build_manifest(provenance, run_name, loaders, model, device):
    """
    loaders: {"train": ..., "validation": ..., "test": ... or None}.
    """

    caches = {split: cache_record(loader.dataset)
              for split, loader in loaders.items() if loader is not None}

    train_config = provenance["train_config"]

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run": run_name,
        "arm": provenance["arm"],
        "seed": provenance["seed"],
        "git_commit": provenance["git_commit"],
        "dirty": provenance["code_diff_sha256"] is not None,
        "code_sha256": provenance["code_sha256"],
        "code_diff_sha256": provenance["code_diff_sha256"],
        "dataset": {
            "name": DATASET,
            "config": provenance["arc_config"],
            "split_sizes": {split: c["n_samples"] for split, c in caches.items()},
        },
        "caches": caches,
        "deterministic": train_config["deterministic"],
        "deterministic_warn_only": train_config["deterministic_warn_only"],
        "cublas_workspace_config": provenance["cublas_workspace_config"],
        "device": str(device),
        "device_name": device_name(device),
        # Only quantum arms have a circuit; the classical-control layer
        # and the no-bottleneck arms record None.
        "quantum_device": getattr(getattr(model, "quantum", None), "device_name", None),
        "environment": provenance["environment"],
        "environment_info": provenance["environment_info"],
        "pin_check": provenance["pin_check"],
        "model_config": provenance["model_config"],
        "train_config": train_config,
        "parameter_counts": provenance["parameter_counts"],
    }

    return json.loads(json.dumps(manifest))


def write_manifest(path, manifest):

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
