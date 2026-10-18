import os
import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=False, warn_only=False):
    """
    Seeds python, numpy and torch.

    This function existed for the project's whole history and was never
    called from any entry point, so no run was reproducible: collate_fn
    draws a fresh torch.randperm per batch to shuffle options, dropout is
    active throughout, and hypothesis generation samples. main.py,
    runner.py and ablations.py all call it now.

    deterministic=True forces deterministic cuDNN kernels and
    torch.use_deterministic_algorithms: an op with no deterministic
    implementation raises, or only warns with warn_only=True. Training
    entry points pass config.DETERMINISTIC (E10); evaluation scripts keep
    the default. The PennyLane lightning.gpu simulator is outside torch's
    control and may have its own nondeterminism.
    """

    # cuBLAS reads this when its first handle is created, so it must be set
    # before any CUDA matmul. A value set by the user is kept.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    print(f"[SEED] Set to {seed} (deterministic={deterministic}, "
          f"warn_only={warn_only})")


def seeded_generator(seed=42):
    """
    Generator for DataLoader shuffling, so batch ORDER is reproducible
    too and not just weight init.
    """

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def capture_rng_state(loader_generator=None):
    """
    Every random stream a training epoch draws from (A11): python, numpy,
    torch CPU and CUDA, and the train loader's shuffling generator. Only
    tensors, ints, floats, tuples and None, so the state reloads under
    torch.load(weights_only=True).
    """

    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()

    return {
        "python": random.getstate(),
        "numpy": {
            "name": np_name,
            "keys": torch.from_numpy(np_keys.astype(np.int64)),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader_generator.get_state() if loader_generator is not None else None,
    }


def restore_rng_state(state, loader_generator=None):
    """Inverse of capture_rng_state; resumes every stream where it stopped."""

    random.setstate(state["python"])

    np_state = state["numpy"]
    np.random.set_state((
        np_state["name"],
        np_state["keys"].numpy().astype(np.uint32),
        np_state["pos"],
        np_state["has_gauss"],
        np_state["cached_gaussian"],
    ))

    torch.set_rng_state(state["torch"])

    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

    if loader_generator is not None and state["loader"] is not None:
        loader_generator.set_state(state["loader"])
