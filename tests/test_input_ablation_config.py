"""
A5 -- input_ablation rebuilds the exact architecture a checkpoint was
trained with, or refuses (docs/baseline/issues.md).

    python -m pytest tests/test_input_ablation_config.py

Real model on CPU with the fake cache from test_checkpoint_provenance;
needs pennylane.
"""

import os

import pytest
import torch

from config import MODEL_DIM, N_QUBITS
from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import REPO_ROOT, smoke_cache  # noqa: F401
from training.checkpoint import load_checkpoint_file, run_key


def best(ckpt_dir, run):
    return os.path.join(str(ckpt_dir), f"{run}_best.pt")


def train_qair(ckpt_dir, cache_dir):
    from training.runner import run_training

    run_training(
        cache_dir=str(cache_dir), ckpt_dir=str(ckpt_dir), train_samples=16,
        val_samples=8, epochs=2, seed=42, with_test=False, arc_config=ARC,
    )
    return best(ckpt_dir, run_key("qair", ARC, 42))


def train_arm(ckpt_dir, cache_dir, arm):
    from training.ablations import run_ablation_suite

    run_ablation_suite(
        cache_dir=str(cache_dir), ckpt_dir=str(ckpt_dir), epochs=2,
        seeds=[42], with_test=False, arc_config=ARC, configs=[arm],
        train_samples=16, val_samples=8,
    )
    return best(ckpt_dir, run_key(arm, ARC, 42))


def ablation_inputs(ckpt_path, cache_dir, config_path=None, **overrides):
    from evaluation.input_ablation import load_for_ablation

    kwargs = dict(split="validation", cache_dir=str(cache_dir), arc_config=None,
                  max_samples=None, device="cpu")
    kwargs.update(overrides)
    return load_for_ablation(ckpt_path, config_path, **kwargs)


# ---------------------------------------------------------------------------
# The dossier's prediction: "intact" equals the logged validation accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", [
    "qair",                               # --mode train, quantum, LLM prior on
    "A1_baseline",                        # grid, no bottleneck, prior off
    "A3c_classical_control_persistent",   # grid, backend="classical_control"
])
def test_intact_accuracy_equals_logged_validation_accuracy(tmp_path, smoke_cache, arm):
    from evaluation.input_ablation import input_ablation

    path = (train_qair(tmp_path, smoke_cache) if arm == "qair"
            else train_arm(tmp_path, smoke_cache, arm))
    assert os.path.exists(path), "no best checkpoint: validation acc stayed 0"

    model, loader = ablation_inputs(path, smoke_cache)

    assert len(loader.dataset) == 8
    assert input_ablation(model, loader, "cpu")["intact"] == \
        load_checkpoint_file(path)["best_acc"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_format1_checkpoint_without_config_raises(tmp_path):
    from evaluation.input_ablation import ModelConfigError, resolve_model_config

    path = str(tmp_path / "A3_persistent_s42_best.pt")

    with pytest.raises(ModelConfigError, match="no default"):
        resolve_model_config({"model": {}}, path)


def test_incomplete_config_raises():
    from evaluation.input_ablation import ModelConfigError, build_model_from_config

    # The dict the old code silently assumed when no config file existed.
    old_default = {"use_quantum": True, "use_validator": True, "persistent_steps": 3}

    with pytest.raises(ModelConfigError, match="lacks"):
        build_model_from_config(old_default, "cpu")


def test_contradicting_config_raises(tmp_path, smoke_cache):
    from evaluation.input_ablation import ModelConfigError

    path = train_arm(tmp_path, smoke_cache, "A1_baseline")
    wrong = dict(load_checkpoint_file(path)["provenance"]["model_config"],
                 use_llm_prior=True)
    config_path = tmp_path / "wrong_config.pt"
    torch.save(wrong, config_path)

    with pytest.raises(ModelConfigError, match="contradicts"):
        ablation_inputs(path, smoke_cache, config_path=str(config_path))


@pytest.mark.parametrize("override, match", [
    ({"max_samples": 7}, "validation data differs"),
    ({"arc_config": "ARC-Challenge"}, "was trained on ARC-Easy"),
])
def test_other_validation_data_raises(tmp_path, smoke_cache, override, match):
    from evaluation.input_ablation import ModelConfigError

    path = train_arm(tmp_path, smoke_cache, "A1_baseline")

    with pytest.raises(ModelConfigError, match=match):
        ablation_inputs(path, smoke_cache, **override)


# ---------------------------------------------------------------------------
# v46 checkpoints still build (rule 9)
# ---------------------------------------------------------------------------

def v46_file(name):
    path = os.path.join(REPO_ROOT, "ckpt", name)
    if not os.path.exists(path):
        pytest.skip(f"{path} not on this machine (checkpoints are untracked)")
    return path


def test_v46_main_checkpoint_builds_from_its_config_file():
    pytest.importorskip("pennylane")
    from evaluation.input_ablation import build_model_from_config, resolve_model_config

    path = v46_file("qair_v46_best.pt")
    ckpt = load_checkpoint_file(path)
    cfg = resolve_model_config(ckpt, path)

    assert cfg == load_checkpoint_file(v46_file("qair_v46_config.pt"))
    build_model_from_config(cfg, "cpu").load_state_dict(ckpt["model"])


def test_v46_grid_checkpoint_builds_from_explicit_config(tmp_path):
    pytest.importorskip("pennylane")
    from evaluation.input_ablation import build_model_from_config, resolve_model_config
    from training.ablations import ABLATIONS, resolve_config

    path = v46_file("A3_persistent_s42_best.pt")
    config_path = tmp_path / "A3_persistent_config.pt"
    torch.save({**resolve_config(ABLATIONS["A3_persistent"]),
                "n_qubits": N_QUBITS, "model_dim": MODEL_DIM}, config_path)

    ckpt = load_checkpoint_file(path)
    cfg = resolve_model_config(ckpt, path, str(config_path))

    assert cfg["use_llm_prior"] is False
    build_model_from_config(cfg, "cpu").load_state_dict(ckpt["model"])
