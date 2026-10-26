"""
Per-epoch training diagnostics (DoD §2.2, §2.7; issues B8, C2).

  * GradNormTracker: per top-level module, the mean and max per-step L2
    gradient norm, after unscaling and before clipping; the total norm, the
    fraction of steps clipped and the number of steps AMP skipped.
  * parameter_drift: every 0-d parameter (phases, temperatures, gates) and
    its drift from the run's start; per top-level module the relative
    weight change ||W - W0|| / ||W0||.

W0 and the initial scalars are stored in the history (INITIAL_WEIGHTS,
INITIAL_SCALARS), so a resumed run measures from the run's true start.
Parameters attached to the model itself (polarity_phase, llm_prior_weight)
are grouped under ROOT.
"""

import torch

ROOT = "model"
TOTAL = "total"

INITIAL_WEIGHTS = "initial_weights"
INITIAL_SCALARS = "initial_scalars"

# Per-epoch history keys, one entry appended per epoch.
EPOCH_KEYS = ("grad_norm", "clipped_fraction", "skipped_steps",
              "scalars", "scalar_drift", "weight_change")


def module_of(name, children):
    head = name.split(".", 1)[0]
    return head if head in children else ROOT


def parameters_by_module(model):
    """{top-level module: [(name, parameter), ...]} in registration order."""

    children = {name for name, _ in model.named_children()}
    groups = {}

    for name, param in model.named_parameters():
        groups.setdefault(module_of(name, children), []).append((name, param))

    return groups


def scalar_values(model):
    return {name: param.item() for name, param in model.named_parameters()
            if param.ndim == 0}


def weight_snapshot(model):
    """W0: a CPU copy of every parameter, keyed by name."""

    return {name: param.detach().cpu().clone()
            for name, param in model.named_parameters()}


def start_of_run(history, model):
    """Records W0 and the initial scalars once, at the run's start."""

    if INITIAL_WEIGHTS not in history:
        history[INITIAL_WEIGHTS] = weight_snapshot(model)
        history[INITIAL_SCALARS] = scalar_values(model)

    for key in EPOCH_KEYS:
        history.setdefault(key, [])


def parameter_drift(model, history):
    """(scalars, scalar drift, relative weight change per module)."""

    scalars = scalar_values(model)
    drift = {name: value - history[INITIAL_SCALARS][name]
             for name, value in scalars.items()}

    initial = history[INITIAL_WEIGHTS]
    change = {}

    with torch.no_grad():
        for module, params in parameters_by_module(model).items():
            diff = sum(
                (param.detach().cpu() - initial[name]).pow(2).sum() for name, param in params
            )
            norm = sum(initial[name].pow(2).sum() for name, _ in params)
            change[module] = (diff.sqrt() / norm.sqrt().clamp(min=1e-12)).item()

    return scalars, drift, change


class GradNormTracker:
    """Accumulates per-step gradient norms over one epoch."""

    def __init__(self, model, max_norm):
        self.groups = {module: [p for _, p in params]
                       for module, params in parameters_by_module(model).items()}
        self.max_norm = max_norm
        self.reset()

    def reset(self):
        self.sums = {}
        self.maxes = {}
        self.steps = 0
        self.clipped = 0
        self.skipped = 0

    def measure(self):
        """
        Per-module norms of the current gradients. Call after unscaling and
        before clipping (clip_grad_norm_ rescales in place); the caller adds
        the total norm that clip_grad_norm_ returns. A missing gradient
        counts as 0.
        """

        norms = []

        for params in self.groups.values():
            grads = [p.grad.detach().norm() for p in params if p.grad is not None]
            norms.append(torch.stack(grads).pow(2).sum().sqrt().cpu() if grads
                         else torch.zeros(()))

        return dict(zip(self.groups, torch.stack(norms).tolist()))

    def add(self, norms, skipped):
        """skipped: AMP found inf/NaN gradients and skipped the step."""

        if skipped:
            self.skipped += 1
            return

        self.steps += 1
        self.clipped += int(norms[TOTAL] > self.max_norm)

        for key, value in norms.items():
            self.sums[key] = self.sums.get(key, 0.0) + value
            self.maxes[key] = max(self.maxes.get(key, 0.0), value)

    def summary(self):
        grad_norm = {key: {"mean": self.sums[key] / self.steps, "max": self.maxes[key]}
                     for key in self.sums}
        clipped = self.clipped / self.steps if self.steps else 0.0
        return grad_norm, clipped, self.skipped


def describe_grad_norm(grad_norm, clipped, skipped):
    total = grad_norm.get(TOTAL, {"mean": 0.0, "max": 0.0})
    modules = " ".join(f"{k} {v['mean']:.3g}/{v['max']:.3g}"
                       for k, v in grad_norm.items() if k != TOTAL)
    return (f"Grad norm     : total {total['mean']:.3g}/{total['max']:.3g} (mean/max) | "
            f"{modules} | clipped {clipped:.0%} of steps, AMP skipped {skipped}")


def describe_drift(drift, change):
    scalars = " ".join(f"{k} {v:+.4g}" for k, v in drift.items())
    modules = " ".join(f"{k} {v:.3g}" for k, v in change.items())
    return (f"Scalar drift  : {scalars}\n"
            f"Weight change : {modules} (||W - W0|| / ||W0||)")
