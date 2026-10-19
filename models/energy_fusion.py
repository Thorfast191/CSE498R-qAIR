import torch
import torch.nn as nn


class EnergyFusion(nn.Module):
    """
    Combines the answer selector's per-hypothesis energy with the optional
    quantum and validator energy signals into a single collapse energy,
    via learned sigmoid-gated blend weights.

    Fixed in the audit
    ------------------
    * Hypothesis energy was reduced over options with a hard
      `torch.min`, which routes gradient to the argmin option only. It is
      now a temperature-controlled soft-min (-tau * logsumexp(-E/tau)),
      which is smooth, gradient-complete, and recovers the hard min as
      tau -> 0.
    * The validator term was added unbounded (`+ w * validator_energy`)
      while the quantum term was a convex blend -- inconsistent
      semantics, and the validator could dominate the energy scale
      arbitrarily. Both are convex blends now.
    """

    def __init__(self, use_quantum_energy=True, use_validator_energy=True):
        """
        Each blend weight exists only when its energy can arrive (A2): a
        weight for an absent signal would never receive gradient.
        """

        super().__init__()

        self.use_quantum_energy = use_quantum_energy
        self.use_validator_energy = use_validator_energy

        if use_quantum_energy:
            self.energy_alpha = nn.Parameter(torch.tensor(0.0))

        # Learnable validator guidance weight. sigmoid(0.0) = 0.5 at init --
        # moderate influence, model can learn to trust it more or suppress
        # it based on whether the validator's signal actually helps.
        if use_validator_energy:
            self.validator_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        answer_energy,
        quantum_energy=None,
        validator_energy=None,
        tau=1.0,
        option_mask=None,
    ):
        """
        answer_energy : (B, K, N)
        option_mask   : (B, N) bool, True for real options -- padded
                        options must not win the soft-min.
        returns       : (B, K) collapse energy
        """

        e = answer_energy / tau

        if option_mask is not None:
            e = e.masked_fill(~option_mask.unsqueeze(1), float("inf"))

        # Soft-min over options: -tau * logsumexp(-e / tau).
        collapse_energy = -tau * torch.logsumexp(-e / tau, dim=-1)

        if (quantum_energy is not None) != self.use_quantum_energy:
            raise ValueError(
                f"EnergyFusion built with use_quantum_energy="
                f"{self.use_quantum_energy} but quantum_energy is "
                f"{'absent' if quantum_energy is None else 'present'}"
            )

        if (validator_energy is not None) != self.use_validator_energy:
            raise ValueError(
                f"EnergyFusion built with use_validator_energy="
                f"{self.use_validator_energy} but validator_energy is "
                f"{'absent' if validator_energy is None else 'present'}"
            )

        if quantum_energy is not None:

            alpha = torch.sigmoid(self.energy_alpha)

            collapse_energy = alpha * collapse_energy + (1.0 - alpha) * quantum_energy

        if validator_energy is not None:

            v_weight = torch.sigmoid(self.validator_alpha)

            collapse_energy = (
                (1.0 - v_weight) * collapse_energy + v_weight * validator_energy
            )

        return collapse_energy
