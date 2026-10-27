"""
E6 -- every run writes a manifest (docs/baseline/issues.md E6).

    python -m pytest tests/test_manifest.py

Real model on CPU with the fake cache from test_checkpoint_provenance;
needs pennylane.
"""

import hashlib
import json
import os
import subprocess

import pytest

from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import REPO_ROOT, smoke_cache  # noqa: F401
from training.checkpoint import code_provenance, run_key
from training.dataset import header_path, safe_load_cache

RUN = run_key("qair", ARC, 42)

REQUIRED = {
    "git_commit", "dirty", "code_sha256", "code_diff_sha256", "caches",
    "dataset", "seed", "deterministic", "deterministic_warn_only", "device",
    "device_name", "quantum_device", "environment", "environment_info",
    "pin_check", "model_config", "train_config", "parameter_counts",
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_manifest(ckpt_dir, run):
    with open(os.path.join(str(ckpt_dir), f"{run}_manifest.json")) as f:
        return json.load(f)


@pytest.fixture
def trained(tmp_path, smoke_cache):
    from training.runner import run_training

    history = run_training(
        cache_dir=str(smoke_cache), ckpt_dir=str(tmp_path), train_samples=16,
        val_samples=8, epochs=1, seed=42, with_test=False, arc_config=ARC,
    )
    return read_manifest(tmp_path, RUN), history


def test_manifest_has_every_field(trained):
    manifest, history = trained

    assert REQUIRED <= set(manifest)
    assert history["manifest"] == manifest
    assert manifest["seed"] == 42 and manifest["deterministic"] is True
    assert manifest["dataset"] == {
        "name": "allenai/ai2_arc", "config": ARC,
        "split_sizes": {"train": 16, "validation": 8},
    }
    assert manifest["quantum_device"] == "lightning.qubit"
    assert manifest["model_config"] == history["provenance"]["model_config"]
    assert len(manifest["environment_info"]["packages_sha256"]) == 64


def test_commit_and_dirty_flag_match_git(trained):
    manifest, _ = trained
    head = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()

    assert manifest["git_commit"] == head
    assert manifest["dirty"] == (manifest["code_diff_sha256"] is not None)


def test_cache_hash_and_header_match_the_files(trained):
    manifest, history = trained
    used = {"train": "train_hypotheses_sha256", "validation": "val_hypotheses_sha256"}

    for split, record in manifest["caches"].items():
        path = record["path"]
        loaded = safe_load_cache(path)

        assert record["file_sha256"] == sha256(path), split
        # The samples the run used (max_samples may cut the file short).
        assert record["hypotheses_sha256"] == history["provenance"][used[split]]
        assert record["generation"] == loaded["provenance"]

        with open(header_path(path)) as f:          # _save always writes it
            assert record["header"] == json.load(f)

        generation = record["generation"]
        for key in ("gen_seed", "generator", "generator_revision",
                    "gen_config", "prompt_sha256"):
            assert key in generation, key


def test_dirty_flag_follows_uncommitted_code(tmp_path):
    from training.manifest import build_manifest

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                       capture_output=True)

    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")

    clean = code_provenance(str(tmp_path))
    (tmp_path / "a.py").write_text("x = 2\n")
    dirty = code_provenance(str(tmp_path))

    def manifest(code):
        provenance = {
            **code, "arm": "qair", "seed": 42, "arc_config": ARC,
            "train_config": {"deterministic": True, "deterministic_warn_only": False},
            "cublas_workspace_config": None, "environment": {},
            "environment_info": {}, "pin_check": {}, "model_config": {},
            "parameter_counts": {},
        }
        return build_manifest(provenance, RUN, {}, model=None, device="cpu")

    assert manifest(clean)["dirty"] is False
    assert manifest(dirty)["dirty"] is True
    assert manifest(dirty)["git_commit"] == clean["git_commit"]


def test_grid_arm_writes_a_manifest(tmp_path, smoke_cache):
    from training.ablations import run_ablation_suite

    results = run_ablation_suite(
        cache_dir=str(smoke_cache), ckpt_dir=str(tmp_path), epochs=1, seeds=[42],
        with_test=False, arc_config=ARC, configs=["A1_baseline"],
        train_samples=16, val_samples=8,
    )
    manifest = read_manifest(tmp_path, run_key("A1_baseline", ARC, 42))

    assert manifest["arm"] == "A1_baseline"
    assert manifest["quantum_device"] is None          # no circuit in A1
    assert results["A1_baseline"][0]["history"]["manifest"] == manifest
