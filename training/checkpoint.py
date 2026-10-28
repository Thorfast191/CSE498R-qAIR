import hashlib
import json
import os
import subprocess

import torch

from config import (
    BASE_LR, BATCH_SIZE, CACHE_VERSION, DETERMINISTIC, LABEL_SMOOTHING, NEW_LR,
)
from training.environment import environment_record
from training.seed import restore_rng_state


# Format 1 (v46 and earlier): epoch, best_acc, model, optimizer, scheduler,
# scaler. Format 2 adds "format" and "provenance". Format 3 (A1/A2) removes
# the parameters that never received gradient. Format-1 and -2 files still
# load for evaluation (load_model_state drops exactly REMOVED_PARAMETER_KEYS);
# only format 3 is resumed.
CHECKPOINT_FORMAT = 3

# Removed in format 3, or built only by the arms that use them. Dropped from
# an older checkpoint only when the model being loaded does not have them.
REMOVED_PARAMETER_KEYS = (
    "interference.hypothesis_temp",
    "collapse.confidence.0.weight",
    "collapse.confidence.0.bias",
    "collapse.confidence.2.weight",
    "collapse.confidence.2.bias",
    "phase_head.weight",
    "phase_head.bias",
    "llm_prior_weight",
    "reasoner.validator_weight",
    "fusion.energy_alpha",
    "fusion.validator_alpha",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files whose contents define the code a checkpoint was trained with. Docs
# and notebooks are excluded, so a docs-only commit does not block a resume.
CODE_PATHSPECS = ("*.py", "requirements.txt")

# Recorded for the reader, never compared: the full commit changes with
# docs-only commits, the device does not define the run, and only the
# versions in "environment" (torch, CUDA, pennylane, lightning) decide the
# computation (E11).
INFORMATIONAL_KEYS = (
    "git_commit", "device", "cublas_workspace_config", "environment_info",
    "pin_check", "parameter_counts",
)


class CheckpointMismatchError(RuntimeError):
    """A checkpoint exists at the run's path but was not made by this run."""


def load_checkpoint_file(path, map_location="cpu"):
    """
    torch.load with weights_only=True: tensors and plain containers only, so
    loading a checkpoint, history or results file can never execute code
    stored in it. Every v46 file in ckpt/ loads this way without an
    allow-list.
    """

    return torch.load(path, map_location=map_location, weights_only=True)


def load_model_state(model, ckpt, source):
    """
    load_state_dict(strict=True) after dropping, from a format-1/2
    checkpoint only, the REMOVED_PARAMETER_KEYS this model does not have.
    Each dropped key is printed; any other difference still raises.
    """

    state = ckpt["model"]

    if ckpt.get("format", 1) < 3:

        expected = set(model.state_dict())
        dropped = [k for k in state if k in REMOVED_PARAMETER_KEYS and k not in expected]

        if dropped:
            print(f"[CHECKPOINT] {source} (format {ckpt.get('format', 1)}): "
                  f"dropped removed parameters {dropped}")
            state = {k: v for k, v in state.items() if k not in dropped}

    model.load_state_dict(state)


def run_key(arm, arc_config, seed):
    """Checkpoint name; distinct arms, datasets and seeds never share a file."""

    return f"{arm}_{arc_config}_s{seed}"


def _git(repo_root, *args):

    return subprocess.run(
        ["git", "-C", repo_root, *args],
        capture_output=True, check=True,
    ).stdout


def code_provenance(repo_root=None):
    """
    code_sha256: sha256 over the sorted paths and working-tree contents of
    every tracked file matching CODE_PATHSPECS. code_diff_sha256: sha256 of
    the uncommitted diff of those files (None when clean). All three values
    are None when git is unavailable, and a resume then refuses.
    """

    repo_root = repo_root or REPO_ROOT

    try:
        commit = _git(repo_root, "rev-parse", "HEAD").decode().strip()
        paths = sorted(
            _git(repo_root, "ls-files", "-z", "--", *CODE_PATHSPECS)
            .decode().split("\0")[:-1]
        )
        diff = _git(repo_root, "diff", "HEAD", "--", *CODE_PATHSPECS)
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "code_sha256": None, "code_diff_sha256": None}

    digest = hashlib.sha256()

    for path in paths:
        digest.update(path.encode("utf-8") + b"\0")
        full_path = os.path.join(repo_root, path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                digest.update(hashlib.sha256(f.read()).digest())
        else:
            digest.update(b"<deleted>")

    return {
        "git_commit": commit,
        "code_sha256": digest.hexdigest(),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def training_config(epochs, patience, weight_decay, train_samples, val_samples,
                    deterministic_warn_only):
    """The optimization settings that change the weights a run produces."""

    return {
        "deterministic": DETERMINISTIC,
        "deterministic_warn_only": deterministic_warn_only,
        "epochs": epochs,
        "patience": patience,
        "batch_size": BATCH_SIZE,
        "base_lr": BASE_LR,
        "new_lr": NEW_LR,
        "weight_decay": weight_decay,
        "label_smoothing": LABEL_SMOOTHING,
        "train_samples": train_samples,
        "val_samples": val_samples,
    }


def run_provenance(arm, arc_config, seed, model_config, train_config,
                   train_loader, val_loader, device, pin_check, model,
                   repo_root=None):
    """
    Everything that must match for a checkpoint to be resumed, plus a few
    informational fields. Normalized through JSON so it holds only plain
    types and reloads under weights_only=True.
    """

    train_ds, val_ds = train_loader.dataset, val_loader.dataset

    environment, environment_info = environment_record()

    provenance = {
        "arm": arm,
        "arc_config": arc_config,
        "seed": seed,
        "model_config": model_config,
        "train_config": train_config,
        "cache_version": CACHE_VERSION,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "train_hypotheses_sha256": train_ds.hypotheses_sha256(),
        "val_hypotheses_sha256": val_ds.hypotheses_sha256(),
        **code_provenance(repo_root),
        "environment": environment,
        "environment_info": environment_info,
        "pin_check": pin_check,
        "parameter_counts": model.parameter_counts(),
        "device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    return json.loads(json.dumps(provenance))


def check_provenance(path, ckpt, provenance):
    """Raises unless the checkpoint at path was made by the run `provenance` describes."""

    saved = ckpt.get("provenance")

    if saved is None:
        raise CheckpointMismatchError(
            f"{path} has no provenance (checkpoint format "
            f"{ckpt.get('format', 1)}, written before A3). It cannot be "
            f"resumed and will not be overwritten; move it out of the "
            f"checkpoint directory to start this run."
        )

    if provenance.get("code_sha256") is None:
        raise CheckpointMismatchError(
            f"Cannot resume {path}: git is unavailable, so the code "
            f"fingerprint of this run is unknown."
        )

    keys = sorted((set(saved) | set(provenance)) - set(INFORMATIONAL_KEYS))

    mismatches = [
        f"  {k}: checkpoint={saved.get(k)!r}  this run={provenance.get(k)!r}"
        for k in keys
        if saved.get(k) != provenance.get(k)
    ]

    if mismatches:
        raise CheckpointMismatchError(
            f"{path} was written by a different run (checkpoint commit "
            f"{saved.get('git_commit')}). Refusing to resume it.\n"
            + "\n".join(mismatches)
        )


RESUME_STATE_KEYS = ("history", "epochs_no_improve", "stopped_early", "rng")


def is_finished(state, epochs):
    """A run is finished when it reached `epochs` or stopped early (A10)."""

    return state["stopped_early"] or state["start_epoch"] >= epochs


def load_or_resume(trainer, ckpt_dir, name, epochs, provenance):
    """
    Loads f"{name}_latest.pt" if present and restores trainer state in
    place (model, optimizer, scheduler, scaler). Shared by runner.py and
    ablations.py so the two entry points can't drift on resume behavior.

    The checkpoint must carry the same provenance (config, seed, data,
    code fingerprint) as this run, or CheckpointMismatchError is raised and
    the file is left untouched.

    Returns the resume state: start_epoch, best_acc, history (None for a
    fresh run), epochs_no_improve, stopped_early and rng. History and the
    early-stopping state come from the checkpoint itself, saved together
    with the weights, never from the exported {name}_history.pt. The RNG
    streams (python, numpy, torch, CUDA, train loader) are restored, so the
    resumed epochs are the epochs an uninterrupted run would train (A11).
    """

    latest_ckpt = os.path.join(ckpt_dir, f"{name}_latest.pt")

    state = {
        "start_epoch": 0,
        "best_acc": 0.0,
        "history": None,
        "epochs_no_improve": 0,
        "stopped_early": False,
        "rng": None,
    }

    print("\nCheckpoint Search:")
    print(latest_ckpt)
    print("Exists:", os.path.exists(latest_ckpt))

    if not os.path.exists(latest_ckpt):
        # A best checkpoint without a latest one would be overwritten by
        # this run, or evaluated as this run if it never improves on it.
        best_ckpt = os.path.join(ckpt_dir, f"{name}_best.pt")
        if os.path.exists(best_ckpt):
            check_provenance(
                best_ckpt, load_checkpoint_file(best_ckpt), provenance
            )
        return state

    ckpt = load_checkpoint_file(latest_ckpt, map_location=trainer.device)

    check_provenance(latest_ckpt, ckpt, provenance)

    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointMismatchError(
            f"{latest_ckpt} is checkpoint format {ckpt.get('format')}; only "
            f"format {CHECKPOINT_FORMAT} can be resumed (the parameter set "
            f"changed in A1/A2)."
        )

    missing = [k for k in RESUME_STATE_KEYS if k not in ckpt]

    if missing:
        raise CheckpointMismatchError(
            f"{latest_ckpt} lacks the resume state {missing}; resuming it "
            f"would discard its history, early-stopping state or RNG state."
        )

    print(f"\n[RESUME {name}]")

    load_model_state(trainer.model, ckpt, latest_ckpt)
    trainer.optim.load_state_dict(ckpt["optimizer"])

    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optim, T_max=epochs, eta_min=1e-6,
    )

    if ckpt.get("scheduler") is not None:
        trainer.scheduler.load_state_dict(ckpt["scheduler"])

    if ckpt.get("scaler") is not None:
        trainer.scaler.load_state_dict(ckpt["scaler"])

    state.update(
        start_epoch=ckpt["epoch"] + 1,
        best_acc=ckpt.get("best_acc", 0.0),
        **{k: ckpt[k] for k in RESUME_STATE_KEYS},
    )

    restore_rng_state(
        ckpt["rng"], getattr(trainer.train_loader, "generator", None)
    )

    print(f"Resuming from epoch {state['start_epoch']}"
          f" (epochs without improvement {state['epochs_no_improve']},"
          f" stopped early {state['stopped_early']})")

    return state


class NoBestCheckpointError(RuntimeError):
    """Test would be evaluated on weights validation never selected (B3)."""


def select_for_test(model, ckpt_dir, name, provenance, device):
    """
    B3: loads the checkpoint validation selected, for the single test
    evaluation of a run, and returns where the test number comes from.
    Raises instead of evaluating whatever weights the model holds when no
    best checkpoint exists.
    """

    if not load_best(model, ckpt_dir, name, provenance, device):
        raise NoBestCheckpointError(
            f"no {name}_best.pt in {ckpt_dir}: validation never selected a "
            f"checkpoint, so there is nothing to evaluate on test.")

    best = load_checkpoint_file(os.path.join(ckpt_dir, f"{name}_best.pt"))
    return {"checkpoint": f"{name}_best.pt", "epoch": best["epoch"] + 1,
            "val_acc": best["best_acc"]}


def load_best(model, ckpt_dir, name, provenance, device):
    """
    Loads f"{name}_best.pt" into model after the same provenance check as a
    resume, so a best checkpoint left by another run is never evaluated as
    this one. Returns False when no best checkpoint exists.
    """

    best_path = os.path.join(ckpt_dir, f"{name}_best.pt")

    if not os.path.exists(best_path):
        return False

    ckpt = load_checkpoint_file(best_path, map_location=device)

    check_provenance(best_path, ckpt, provenance)

    load_model_state(model, ckpt, best_path)

    return True
