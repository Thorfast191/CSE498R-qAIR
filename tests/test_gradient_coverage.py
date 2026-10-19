"""
A1/A2 regression test -- every parameter of every arm receives a gradient,
except the documented no-gradient allowance
(QAIRvNext.no_gradient_parameters), and every allowed parameter really
gets none (docs/baseline/issues.md; DoD §2.2).

    python -m pytest tests/test_gradient_coverage.py

Real model on CPU, one batch of the fake cache; needs pennylane.
"""

from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)
from training.checkpoint import REMOVED_PARAMETER_KEYS

# A parameter is dead when its gradient is None or exactly zero. Small is
# not dead: validator.reliability.* measure ~1e-7 and are on the graph.
# An allowed parameter may also show float rounding noise (the energy_net
# output bias, exactly zero in exact arithmetic, measures ~1e-8 to ~5e-7),
# bounded relative to the model's largest gradient.
RELATIVE_NOISE = 1e-5

ARMS = [
    "A0_no_question", "A1_baseline", "A1b_quantum_only",
    "A1c_classical_control_only", "A2_validator", "A3_persistent",
    "A3c_classical_control_persistent", "A4_full_hybrid",
    "P0_classical_collapse", "P1_phase_zero", "PS_learned_phase_head",
    "M0_softmax_mixing", "E0_support_only", "F0_full_with_llm_prior",
]


@pytest.fixture
def batch(smoke_cache):
    from training.dataset import QAIRDataset, collate_fn

    ds = QAIRDataset(split="train", max_samples=16, cache_dir=str(smoke_cache),
                     arc_config=ARC)
    loader = DataLoader(ds, batch_size=16,
                        collate_fn=partial(collate_fn, shuffle_options=False))
    return next(iter(loader))


def backward(arm, batch):
    from training.ablations import ABLATIONS, build_model
    from training.evaluate import batch_to_model
    from training.losses import compute_loss

    torch.manual_seed(0)
    model = build_model(ABLATIONS[arm])
    model.train()
    out = model(**batch_to_model(batch, "cpu"), y=batch["y"])
    compute_loss(out, batch["y"]).backward()
    return model


def dead(param):
    return param.grad is None or not param.grad.any()


def largest_gradient(model):
    return max(p.grad.abs().max().item() for p in model.parameters()
               if p.grad is not None)


@pytest.mark.parametrize("arm", ARMS)
def test_every_parameter_outside_the_allowance_gets_a_gradient(arm, batch):
    model = backward(arm, batch)
    allowance = model.no_gradient_parameters()

    params = dict(model.named_parameters())
    idle = {n for n, p in params.items() if dead(p)}
    noise = RELATIVE_NOISE * largest_gradient(model)

    assert idle - set(allowance) == set(), f"{arm}: unexpected dead parameters"

    live_in_allowance = {
        n for n in allowance
        if not dead(params[n]) and params[n].grad.abs().max().item() > noise
    }
    assert live_in_allowance == set(), f"{arm}: allowance lists live parameters"


@pytest.mark.parametrize("arm", ARMS)
def test_removed_parameters_exist_only_where_used(arm, batch):
    model = backward(arm, batch)
    names = {n for n, _ in model.named_parameters()}

    assert not names & {"interference.hypothesis_temp",
                        "collapse.confidence.0.weight", "collapse.confidence.2.weight"}
    assert ("llm_prior_weight" in names) == model.use_llm_prior
    assert ("phase_head.weight" in names) == model.use_phase_head
    assert ("fusion.energy_alpha" in names) == model.use_quantum
    assert ("fusion.validator_alpha" in names) == model.use_validator
    assert ("reasoner.validator_weight" in names) == model.validator_feedback
    assert names.isdisjoint(set(REMOVED_PARAMETER_KEYS)
                            - {"llm_prior_weight", "phase_head.weight",
                               "phase_head.bias", "fusion.energy_alpha",
                               "fusion.validator_alpha",
                               "reasoner.validator_weight"})


@pytest.mark.parametrize("arm", ARMS)
def test_parameter_counts(arm, batch):
    model = backward(arm, batch)
    allowance = model.no_gradient_parameters()
    counts = model.parameter_counts()

    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["total"] - counts["trainable"] == sum(
        p.numel() for n, p in model.named_parameters() if n in allowance)


def test_energy_bias_does_not_change_the_output(batch):
    from training.evaluate import batch_to_model

    model = backward("A3_persistent", batch)
    model.eval()
    inputs = batch_to_model(batch, "cpu")

    with torch.no_grad():
        before = model(**inputs)["scores"].clone()
        model.selector.energy_net[6].bias += 5.0
        after = model(**inputs)["scores"]

    assert (after - before).abs().max().item() < 1e-4
