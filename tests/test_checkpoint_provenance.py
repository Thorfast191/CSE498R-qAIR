"""
A3 + A8 -- checkpoints are keyed by arm, dataset and seed, carry their
provenance, and are never resumed or evaluated by a run with different
provenance (docs/baseline/issues.md).

    python -m pytest tests/test_checkpoint_provenance.py

The two real-model tests at the end need pennylane (requirements.txt pins)
and are skipped without it.
"""

import hashlib
import os
import pickle
import subprocess

import pytest
import torch

import training.checkpoint as checkpoint_module
import training.dataset as dataset_module
from tests.test_cache_determinism import ARC, FakeEncoder, FakeGenerator, fake_rows
from training.checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointMismatchError,
    check_provenance,
    code_provenance,
    load_best,
    load_checkpoint_file,
    load_or_resume,
    run_key,
)
from training.dataset import QAIRDataset
from training.train import Trainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def provenance(**overrides):
    base = {
        "arm": "qair",
        "arc_config": ARC,
        "seed": 42,
        "model_config": {"use_quantum": True, "persistent_steps": 3},
        "train_config": {"epochs": 2, "batch_size": 16},
        "cache_version": 46,
        "n_train": 16,
        "n_val": 8,
        "train_hypotheses_sha256": "a" * 64,
        "val_hypotheses_sha256": "b" * 64,
        "git_commit": "c" * 40,
        "code_sha256": "d" * 64,
        "code_diff_sha256": None,
        "environment": {"torch": "2.14.1", "cuda": None,
                        "pennylane": "0.42.3", "pennylane-lightning": "0.42.0"},
        "environment_info": {"python": "3.12.12", "packages_sha256": "1" * 64},
        "pin_check": {"mismatches": [], "overridden": False},
        "device": "cpu",
    }
    base.update(overrides)
    return base


def tiny_trainer(ckpt_dir, prov, name="qair_ARC-Easy_s42", seed=0):
    torch.manual_seed(seed)
    return Trainer(
        model=torch.nn.Linear(4, 2), train_loader=None, val_loader=None,
        device="cpu", ckpt_dir=str(ckpt_dir), name=name, provenance=prov,
    )


def save(trainer, epoch, best_acc, best=False):
    trainer.save_checkpoint(epoch, best_acc, history={"acc": []},
                            epochs_no_improve=0, best=best)


def latest(ckpt_dir, name="qair_ARC-Easy_s42"):
    return os.path.join(ckpt_dir, f"{name}_latest.pt")


def best(ckpt_dir, name="qair_ARC-Easy_s42"):
    return os.path.join(ckpt_dir, f"{name}_best.pt")


# ---------------------------------------------------------------------------
# Run key and checkpoint format
# ---------------------------------------------------------------------------

def test_run_key_separates_arm_dataset_and_seed():
    keys = {run_key(a, d, s) for a in ("qair", "A3_persistent")
            for d in ("ARC-Easy", "ARC-Challenge") for s in (42, 43)}
    assert len(keys) == 8
    assert run_key("qair", "ARC-Easy", 42) == "qair_ARC-Easy_s42"


def test_checkpoint_round_trips_under_weights_only(tmp_path):
    prov = provenance()
    save(tiny_trainer(tmp_path, prov), 0, 0.5)

    ckpt = load_checkpoint_file(latest(tmp_path))

    assert ckpt["format"] == CHECKPOINT_FORMAT == 2
    assert ckpt["provenance"] == prov


def test_resume_with_matching_provenance(tmp_path):
    prov = provenance()
    saved = tiny_trainer(tmp_path, prov, seed=1)
    save(saved, 3, 0.5)

    fresh = tiny_trainer(tmp_path, prov, seed=2)
    state = load_or_resume(
        fresh, str(tmp_path), "qair_ARC-Easy_s42", epochs=8, provenance=prov,
    )

    assert (state["start_epoch"], state["best_acc"]) == (4, 0.5)
    assert torch.equal(fresh.model.weight, saved.model.weight)


@pytest.mark.parametrize("key, value", [
    ("seed", 43),
    ("model_config", {"use_quantum": True, "persistent_steps": 5}),
    ("train_config", {"epochs": 30, "batch_size": 16}),
    ("train_hypotheses_sha256", "e" * 64),
    ("code_sha256", "f" * 64),
    ("code_diff_sha256", "0" * 64),
    ("environment", {"torch": "2.15.0", "cuda": None,
                     "pennylane": "0.42.3", "pennylane-lightning": "0.42.0"}),
    ("environment", {"torch": "2.14.1", "cuda": "12.4",
                     "pennylane": "0.42.3", "pennylane-lightning": "0.42.0"}),
    ("environment", {"torch": "2.14.1", "cuda": None,
                     "pennylane": "0.43.0", "pennylane-lightning": "0.42.0"}),
    ("environment", {"torch": "2.14.1", "cuda": None,
                     "pennylane": "0.42.3", "pennylane-lightning": "0.43.0"}),
])
def test_resume_refuses_any_mismatch_and_leaves_file(tmp_path, key, value):
    save(tiny_trainer(tmp_path, provenance()), 3, 0.5)
    before = sha256(latest(tmp_path))

    with pytest.raises(CheckpointMismatchError) as err:
        load_or_resume(
            tiny_trainer(tmp_path, provenance()), str(tmp_path),
            "qair_ARC-Easy_s42", epochs=8, provenance=provenance(**{key: value}),
        )

    assert key in str(err.value) and repr(value) in str(err.value)
    assert sha256(latest(tmp_path)) == before


def test_informational_fields_do_not_block_resume(tmp_path):
    save(tiny_trainer(tmp_path, provenance()), 0, 0.1)
    current = provenance(
        git_commit="9" * 40, device="mps",
        environment_info={"python": "3.13.0", "packages_sha256": "2" * 64},
        pin_check={"mismatches": [{"package": "datasets"}], "overridden": True},
    )

    state = load_or_resume(
        tiny_trainer(tmp_path, current), str(tmp_path),
        "qair_ARC-Easy_s42", epochs=8, provenance=current,
    )

    assert state["start_epoch"] == 1


def test_old_format_checkpoint_is_refused_and_left(tmp_path):
    v46_style = {
        "epoch": 7, "best_acc": 0.15,
        "model": torch.nn.Linear(4, 2).state_dict(),
        "optimizer": None, "scheduler": None, "scaler": None,
    }
    torch.save(v46_style, latest(tmp_path))
    before = sha256(latest(tmp_path))

    with pytest.raises(CheckpointMismatchError, match="no provenance"):
        load_or_resume(
            tiny_trainer(tmp_path, provenance()), str(tmp_path),
            "qair_ARC-Easy_s42", epochs=8, provenance=provenance(),
        )

    assert sha256(latest(tmp_path)) == before


def test_resume_refuses_when_code_fingerprint_unknown(tmp_path):
    save(tiny_trainer(tmp_path, provenance()), 0, 0.1)

    with pytest.raises(CheckpointMismatchError, match="git is unavailable"):
        load_or_resume(
            tiny_trainer(tmp_path, provenance()), str(tmp_path),
            "qair_ARC-Easy_s42", epochs=8, provenance=provenance(code_sha256=None),
        )


def test_stale_best_is_refused_at_start_and_at_evaluation(tmp_path):
    other = provenance(seed=43)
    save(tiny_trainer(tmp_path, other), 2, 0.9, best=True)

    with pytest.raises(CheckpointMismatchError, match="seed"):
        load_or_resume(
            tiny_trainer(tmp_path, provenance()), str(tmp_path),
            "qair_ARC-Easy_s42", epochs=8, provenance=provenance(),
        )

    with pytest.raises(CheckpointMismatchError, match="seed"):
        load_best(torch.nn.Linear(4, 2), str(tmp_path), "qair_ARC-Easy_s42",
                  provenance(), "cpu")

    assert load_best(torch.nn.Linear(4, 2), str(tmp_path), "absent",
                     provenance(), "cpu") is False


class _CreatesFile:

    marker = None

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_loader_never_unpickles_code(tmp_path):
    marker = tmp_path / "pwned"
    payload = _CreatesFile()
    payload.marker = str(marker)
    path = tmp_path / "evil.pt"
    torch.save({"model": payload}, path)

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint_file(str(path))

    assert not marker.exists()


@pytest.mark.parametrize("filename", [
    "qair_v46_best.pt", "qair_v46_config.pt", "qair_v46_history.pt",
    "A3_persistent_s42_latest.pt", "ablation_results.pt",
])
def test_v46_files_still_load_for_evaluation(filename):
    path = os.path.join(REPO_ROOT, "ckpt", filename)
    if not os.path.exists(path):
        pytest.skip(f"{path} not on this machine (checkpoints are untracked)")

    assert isinstance(load_checkpoint_file(path), dict)


# ---------------------------------------------------------------------------
# Code fingerprint
# ---------------------------------------------------------------------------

def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    (root / "model.py").write_text("x = 1\n")
    (root / "requirements.txt").write_text("torch\n")
    (root / "notes.md").write_text("draft\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "initial")
    return root


def test_docs_only_commit_still_resumes(tmp_path, repo):
    saved = provenance(**code_provenance(str(repo)))
    save(tiny_trainer(tmp_path, saved), 1, 0.3)

    (repo / "notes.md").write_text("final\n")
    git(repo, "commit", "-q", "-am", "docs only")
    current = provenance(**code_provenance(str(repo)))

    assert current["git_commit"] != saved["git_commit"]
    state = load_or_resume(
        tiny_trainer(tmp_path, current), str(tmp_path),
        "qair_ARC-Easy_s42", epochs=8, provenance=current,
    )
    assert state["start_epoch"] == 2


@pytest.mark.parametrize("commit", [False, True], ids=["uncommitted", "committed"])
@pytest.mark.parametrize("filename", ["model.py", "requirements.txt"])
def test_code_change_refuses_resume(tmp_path, repo, filename, commit):
    saved = provenance(**code_provenance(str(repo)))
    save(tiny_trainer(tmp_path, saved), 1, 0.3)

    (repo / filename).write_text("changed\n")
    if commit:
        git(repo, "commit", "-q", "-am", "code change")
    current = provenance(**code_provenance(str(repo)))

    with pytest.raises(CheckpointMismatchError, match="code_sha256"):
        load_or_resume(
            tiny_trainer(tmp_path, current), str(tmp_path),
            "qair_ARC-Easy_s42", epochs=8, provenance=current,
        )


def test_dirty_diff_is_recorded(repo):
    assert code_provenance(str(repo))["code_diff_sha256"] is None
    (repo / "model.py").write_text("x = 2\n")
    assert code_provenance(str(repo))["code_diff_sha256"] is not None


def test_fingerprint_is_none_outside_git(tmp_path):
    assert code_provenance(str(tmp_path)) == {
        "git_commit": None, "code_sha256": None, "code_diff_sha256": None,
    }


# ---------------------------------------------------------------------------
# Real model, CPU, fake 29-question cache (needs pennylane)
# ---------------------------------------------------------------------------

@pytest.fixture
def smoke_cache(tmp_path, monkeypatch):
    pytest.importorskip("pennylane")

    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    rows = fake_rows()
    monkeypatch.setattr(dataset_module, "load_arc",
                        lambda cfg: {"train": rows, "validation": rows})

    cache_dir = tmp_path / "cache"
    for split in ("train", "validation"):
        QAIRDataset(split=split, cache_dir=str(cache_dir), arc_config=ARC,
                    allow_build=True)

    import training.ablations as ablations_module
    import training.runner as runner_module
    monkeypatch.setattr(runner_module, "device", "cpu")
    monkeypatch.setattr(ablations_module, "device", "cpu")

    return cache_dir


def test_two_seeds_give_two_runs(tmp_path, smoke_cache):
    from training.runner import run_training

    ckpt_dir = tmp_path / "ckpt"
    losses = {}

    for seed in (42, 43):
        history = run_training(
            cache_dir=str(smoke_cache), ckpt_dir=str(ckpt_dir),
            train_samples=16, val_samples=8, epochs=1, seed=seed,
            with_test=False, arc_config=ARC,
        )
        losses[seed] = history["loss"][0]

        ckpt = load_checkpoint_file(latest(ckpt_dir, run_key("qair", ARC, seed)))
        assert ckpt["provenance"]["seed"] == seed
        assert ckpt["provenance"]["n_train"] == 16
        assert ckpt["provenance"]["code_sha256"] is not None

    assert losses[42] != losses[43]


def test_ablation_grid_refuses_checkpoint_from_other_code(tmp_path, smoke_cache, capsys):
    from training.ablations import run_ablation_suite

    ckpt_dir = tmp_path / "ckpt"

    def grid():
        return run_ablation_suite(
            cache_dir=str(smoke_cache), ckpt_dir=str(ckpt_dir), epochs=1,
            seeds=[42], with_test=False, arc_config=ARC,
            configs=["A1_baseline"], train_samples=16, val_samples=8,
        )

    grid()
    grid()
    assert "[SKIP] A1_baseline_ARC-Easy_s42 already finished (1 epochs)" in capsys.readouterr().out

    path = latest(ckpt_dir, run_key("A1_baseline", ARC, 42))
    ckpt = load_checkpoint_file(path)
    ckpt["provenance"]["code_sha256"] = "0" * 64
    torch.save(ckpt, path)
    before = sha256(path)

    with pytest.raises(CheckpointMismatchError, match="code_sha256"):
        grid()

    assert sha256(path) == before
