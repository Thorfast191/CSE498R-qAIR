import hashlib
import json
import os
import subprocess

import torch

from config import BASE_LR, BATCH_SIZE, CACHE_VERSION, LABEL_SMOOTHING, NEW_LR


# Format 1 (v46 and earlier): epoch, best_acc, model, optimizer, scheduler,
# scaler. Format 2 adds "format" and "provenance". Format-1 files still load
# for evaluation; they are never resumed.
CHECKPOINT_FORMAT = 2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files whose contents define the code a checkpoint was trained with. Docs
# and notebooks are excluded, so a docs-only commit does not block a resume.
CODE_PATHSPECS = ("*.py", "requirements.txt")

# Recorded for the reader, never compared: the full commit changes with
# docs-only commits, and the device does not define the run.
INFORMATIONAL_KEYS = ("git_commit", "torch_version", "device")


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


def training_config(epochs, patience, weight_decay, train_samples, val_samples):
    """The optimization settings that change the weights a run produces."""

    return {
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
                   train_loader, val_loader, device, repo_root=None):
    """
    Everything that must match for a checkpoint to be resumed, plus a few
    informational fields. Normalized through JSON so it holds only plain
    types and reloads under weights_only=True.
    """

    train_ds, val_ds = train_loader.dataset, val_loader.dataset

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
        "torch_version": str(torch.__version__),
        "device": str(device),
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


def load_or_resume(trainer, ckpt_dir, name, epochs, provenance):
    """
    Loads f"{name}_latest.pt" if present and restores trainer state in
    place (model, optimizer, scheduler, scaler). Shared by runner.py and
    ablations.py so the two entry points can't drift on resume behavior.

    The checkpoint must carry the same provenance (config, seed, data,
    code fingerprint) as this run, or CheckpointMismatchError is raised and
    the file is left untouched.

    Returns (start_epoch, best_acc, resumed_history). resumed_history is
    the contents of history.pt if present, else None.
    """

    latest_ckpt = os.path.join(ckpt_dir, f"{name}_latest.pt")

    start_epoch = 0
    best_acc = 0.0
    resumed_history = None

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
        return start_epoch, best_acc, resumed_history

    ckpt = load_checkpoint_file(latest_ckpt, map_location=trainer.device)

    check_provenance(latest_ckpt, ckpt, provenance)

    print(f"\n[RESUME {name}]")

    trainer.model.load_state_dict(ckpt["model"])
    trainer.optim.load_state_dict(ckpt["optimizer"])

    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optim, T_max=epochs, eta_min=1e-6,
    )

    if ckpt.get("scheduler") is not None:
        trainer.scheduler.load_state_dict(ckpt["scheduler"])

    if ckpt.get("scaler") is not None:
        trainer.scaler.load_state_dict(ckpt["scaler"])

    best_acc = ckpt.get("best_acc", 0.0)
    start_epoch = ckpt["epoch"] + 1

    print(f"Resuming from epoch {start_epoch}")

    # Per-run history. This used to read a single shared "history.pt",
    # which every ablation config overwrote in turn -- so a resumed run
    # could be handed a different config's curves.
    history_path = os.path.join(ckpt_dir, f"{name}_history.pt")
    if os.path.exists(history_path):
        try:
            resumed_history = load_checkpoint_file(history_path)
        except Exception as e:
            print(f"[WARN] Could not load history.pt: {e}")

    return start_epoch, best_acc, resumed_history


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

    model.load_state_dict(ckpt["model"])

    return True
