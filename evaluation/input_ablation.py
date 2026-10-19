"""
Input-ablation regression test.

This is the single most important test in the project, and its absence
is why a total failure of the central mechanism went unnoticed while the
model reported a plausible-looking 33.7%.

qAIR claims to reason over MULTIPLE HYPOTHESES held in superposition.
That claim is falsifiable in one line: corrupt the hypotheses and see
whether the answer changes. On the pre-audit model it did not -- not
"changed a little", but bit-identically unchanged at 294/869 under every
corruption tried:

    intact H, O                        294/869 = 0.3383
    H := zeros                         294/869 = 0.3383
    H := gaussian noise                294/869 = 0.3383
    H rows shuffled within sample      294/869 = 0.3383
    H from a DIFFERENT question        294/869 = 0.3383
    H := O (pure option echo)          294/869 = 0.3383
    O := zeros                         223/869 = 0.2566   <- only O mattered

v45 makes this per-channel, because the model now has more inputs and
"the hypotheses matter" is no longer the only question worth asking. In
particular the cached LLM log-prior is a strong feature that can carry
the accuracy by itself -- a model that scores well with `H := zeros` and
collapses with `llm_logprob := zeros` is a wrapper around its own
generator, and reporting it as multi-hypothesis reasoning would be the
same category of error the v44 audit found, one level up.

    python -m evaluation.input_ablation --ckpt ckpt/qair_v45_best.pt
"""

import argparse
import os

import torch
from functools import partial
from torch.utils.data import DataLoader

from config import (
    CACHE_DIR,
    BATCH_SIZE,
    EMBEDDING_DIM,
    ARC_CONFIG,
    resolve_device,
)
from training.checkpoint import load_checkpoint_file, load_model_state
from training.dataset import QAIRDataset, collate_fn
from training.evaluate import batch_to_model
from training.seed import set_seed
from models.full_model import QAIRvNext


def _zero_where(inputs, keep):
    """Zero the hypothesis rows where `keep` is False."""

    H = inputs["H"].clone()

    H[~keep] = 0.0

    return {**inputs, "H": H}


CORRUPTIONS = {

    "intact": lambda x: x,

    # ---- the whole hypothesis channel ----

    "H := zeros": lambda x: {**x, "H": torch.zeros_like(x["H"])},

    "H := noise": lambda x: {**x, "H": torch.randn_like(x["H"]) * x["H"].std()},

    "H shuffled within sample": lambda x: {
        **x, "H": x["H"][:, torch.randperm(x["H"].shape[1], device=x["H"].device)]
    },

    "H rolled across batch": lambda x: {**x, "H": torch.roll(x["H"], 1, dims=0)},

    "H := O (option echo)": lambda x: {
        **x,
        "H": x["O"].repeat(1, (x["H"].shape[1] + x["O"].shape[1] - 1)
                           // x["O"].shape[1], 1)[:, : x["H"].shape[1]].clone(),
    },

    # ---- one evidence direction at a time ----

    "support := zeros": lambda x: _zero_where(x, x["polarity"] < 0),

    "attack := zeros": lambda x: _zero_where(x, x["polarity"] > 0),

    # ---- the structural tags ----

    "polarity := 0": lambda x: {**x, "polarity": torch.zeros_like(x["polarity"])},

    "align := 0": lambda x: {**x, "align": torch.zeros_like(x["align"])},

    # ---- the other inputs ----

    "llm_logprob := zeros": lambda x: {
        **x, "llm_logprob": torch.zeros_like(x["llm_logprob"])
    },

    "O := zeros": lambda x: {**x, "O": torch.zeros_like(x["O"])},
}

HYPOTHESIS_CHANNELS = (
    "H := zeros",
    "H := noise",
    "H shuffled within sample",
    "H rolled across batch",
    "H := O (option echo)",
    "support := zeros",
    "attack := zeros",
)


@torch.no_grad()
def input_ablation(model, loader, device):

    model.eval()

    results = {}

    for name, fn in CORRUPTIONS.items():

        correct = 0
        total = 0

        for batch in loader:

            inputs = batch_to_model(batch, device)

            y = batch["y"].to(device)

            out = model(**fn(inputs))

            correct += (out["scores"].argmax(dim=1) == y).sum().item()
            total += y.size(0)

        results[name] = correct / total

    return results


def report(results, threshold=0.005):

    base = results["intact"]

    print("\n" + "=" * 72)
    print("INPUT ABLATION -- which inputs is this model actually using?")
    print("=" * 72)

    for name, acc in results.items():

        delta = acc - base

        flag = ""

        if name in HYPOTHESIS_CHANNELS and abs(delta) < 1e-9:
            flag = "   <-- IDENTICAL: channel unused"

        print(f"  {name:<28s} {acc:.4f}   delta={delta:+.4f}{flag}")

    h_deltas = [results[k] - base for k in HYPOTHESIS_CHANNELS if k in results]

    worst = min(h_deltas) if h_deltas else 0.0

    llm_delta = results.get("llm_logprob := zeros", base) - base

    print("-" * 72)

    passed = worst <= -threshold

    if not passed:
        print(
            f"VERDICT: FAIL. Corrupting the hypotheses costs at most "
            f"{-worst:.4f} accuracy.\n"
            f"  The multi-hypothesis mechanism is inert; this model is an "
            f"option-prior\n  classifier. Do not report quantum, collapse, or "
            f"superposition results from\n  this checkpoint."
        )
    else:
        print(
            f"VERDICT: PASS. Destroying the hypotheses costs {-worst:.4f} "
            f"accuracy, so the\n  model is genuinely using them."
        )

    print()

    if llm_delta > -threshold:
        print(
            f"  LLM prior      : removing it costs {-llm_delta:.4f} -- the "
            f"cached generator\n                   likelihood is not carrying "
            f"this result."
        )
    elif -llm_delta > -worst:
        print(
            f"  LLM prior      : removing it costs {-llm_delta:.4f}, MORE than "
            f"destroying the\n                   hypotheses ({-worst:.4f}). "
            f"Most of this accuracy is the generator's,\n                   "
            f"not the architecture's. Report the use_llm_prior=False arm "
            f"as the\n                   reasoning result."
        )
    else:
        print(
            f"  LLM prior      : removing it costs {-llm_delta:.4f}; "
            f"hypotheses cost {-worst:.4f}.\n                   Both channels "
            f"contribute."
        )

    print("=" * 72)

    return passed


# Every key QAIRvNext needs to rebuild a trained arm. All arms share one
# parameter set, so load_state_dict cannot detect a wrong config (A5): a
# missing key is an error, never a default. "backend" is optional because
# only the classical-control arms set it.
MODEL_CONFIG_KEYS = (
    "model_dim",
    "use_quantum",
    "use_validator",
    "persistent_steps",
    "n_qubits",
    "use_question",
    "phase_mode",
    "phase_source",
    "mixing",
    "use_llm_prior",
    "use_attack",
)


class ModelConfigError(RuntimeError):
    """The configuration a checkpoint was trained with is missing or contradicted."""


def build_model_from_config(cfg, device):

    missing = [k for k in MODEL_CONFIG_KEYS if k not in cfg]

    if missing:
        raise ModelConfigError(f"model config lacks {missing}: {cfg}")

    return QAIRvNext(
        dim=EMBEDDING_DIM,
        backend=cfg.get("backend"),
        **{k: cfg[k] for k in MODEL_CONFIG_KEYS},
    ).to(device)


def _architecture(cfg):
    return {k: cfg.get(k) for k in MODEL_CONFIG_KEYS + ("backend",)}


def resolve_model_config(ckpt, ckpt_path, config_path=None):
    """
    Format-2 checkpoints carry their model config in their provenance; an
    explicit config_path must then agree with it. Format-1 (v46)
    checkpoints need config_path or the {run}_config.pt written next to
    {run}_best.pt (only qair_v45 / qair_v46 have one). Anything else raises.
    """

    explicit = load_checkpoint_file(config_path) if config_path else None

    provenance = ckpt.get("provenance")

    if provenance is not None:

        cfg = provenance["model_config"]

        if explicit is not None and _architecture(explicit) != _architecture(cfg):
            raise ModelConfigError(
                f"--config {config_path} contradicts the config stored in "
                f"{ckpt_path}:\n  --config   : {_architecture(explicit)}\n"
                f"  checkpoint: {_architecture(cfg)}"
            )

        return cfg

    if explicit is not None:
        return explicit

    sidecar = (
        ckpt_path[: -len("_best.pt")] + "_config.pt"
        if ckpt_path.endswith("_best.pt") else None
    )

    if sidecar is None or not os.path.exists(sidecar):
        raise ModelConfigError(
            f"{ckpt_path} is a format-1 checkpoint with no stored config and "
            f"no {sidecar or '<run>_config.pt'}. Pass --config with the "
            f"exact config it was trained with; there is no default."
        )

    return load_checkpoint_file(sidecar)


def load_for_ablation(ckpt_path, config_path, split, cache_dir, arc_config,
                      max_samples, device):
    """
    Rebuilds the checkpoint's model and the evaluation loader. For a
    format-2 checkpoint, arc_config and (on validation) max_samples default
    to the run's, and the validation data must be the data the run
    validated on: same size and same hypotheses hash.
    """

    ckpt = load_checkpoint_file(ckpt_path, map_location=device)

    model = build_model_from_config(
        resolve_model_config(ckpt, ckpt_path, config_path), device,
    )
    load_model_state(model, ckpt, ckpt_path)

    provenance = ckpt.get("provenance")

    if provenance is not None:

        if arc_config is not None and arc_config != provenance["arc_config"]:
            raise ModelConfigError(
                f"--arc-config {arc_config} but {ckpt_path} was trained on "
                f"{provenance['arc_config']}"
            )

        arc_config = provenance["arc_config"]

        if split == "validation" and max_samples is None:
            max_samples = provenance["n_val"]

    ds = QAIRDataset(
        split=split, max_samples=max_samples, cache_dir=cache_dir,
        arc_config=arc_config or ARC_CONFIG,
    )

    if provenance is not None and split == "validation":

        seen = (len(ds), ds.hypotheses_sha256())
        trained = (provenance["n_val"], provenance["val_hypotheses_sha256"])

        if seen != trained:
            raise ModelConfigError(
                f"validation data differs from the run's: this cache gives "
                f"(n, sha256) = {seen}, {ckpt_path} validated on {trained}"
            )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, shuffle_options=False),
    )

    return model, loader


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument(
        "--arc-config", type=str, default=None,
        help=f"Default: the checkpoint's dataset, else {ARC_CONFIG}.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(0)

    device = resolve_device()

    model, loader = load_for_ablation(
        args.ckpt, args.config, args.split, args.cache_dir,
        args.arc_config, args.max_samples, device,
    )

    passed = report(input_ablation(model, loader, device))

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
