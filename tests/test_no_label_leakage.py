"""
DoD §2.5 -- no label leakage: inside the model, `y` reaches only the
validator's BCE target (models/validator.py:140-142). Every other output
is identical whether `y` is passed, omitted, or wrong.

    python -m pytest tests/test_no_label_leakage.py

Real model on CPU, one batch of the fake cache; needs pennylane.
"""

import pytest
import torch

from tests.test_gradient_coverage import ARMS, batch  # noqa: F401  (fixture)
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)

TARGET = "validator.relevance_target"


def flatten(out, prefix=""):
    """{dotted name: tensor} for every tensor in the model's output."""

    flat = {}

    for key, value in out.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if torch.is_tensor(item):
                    flat[f"{name}.{i}"] = item
        elif torch.is_tensor(value):
            flat[name] = value

    return flat


def wrong_labels(batch):
    """A different valid option for every question."""

    n_valid = batch["O_mask"].sum(dim=1)
    return (batch["y"] + 1) % n_valid


def forward(model, inputs, y, seed):
    torch.manual_seed(seed)   # same dropout masks in train mode
    with torch.no_grad():
        return flatten(model(**inputs, y=y))


def differing(a, b):
    assert set(a) - {TARGET} == set(b) - {TARGET}
    return sorted(k for k in set(a) & set(b) if not torch.equal(a[k], b[k]))


@pytest.fixture
def model_and_inputs(request, batch):
    from training.ablations import ABLATIONS, build_model
    from training.evaluate import batch_to_model

    torch.manual_seed(0)
    model = build_model(ABLATIONS[request.param])
    return model, batch_to_model(batch, "cpu"), batch


@pytest.mark.parametrize("model_and_inputs", ARMS, indirect=True)
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_outputs_do_not_depend_on_y(model_and_inputs, mode):
    model, inputs, batch = model_and_inputs
    model.train(mode == "train")

    without = forward(model, inputs, None, seed=1)
    right = forward(model, inputs, batch["y"], seed=1)
    wrong = forward(model, inputs, wrong_labels(batch), seed=1)

    assert differing(without, right) == []
    assert differing(right, wrong) == ([TARGET] if model.use_validator else [])
    assert TARGET not in without


@pytest.mark.parametrize("model_and_inputs", ["A3_persistent"], indirect=True)
def test_y_reaches_the_validator_target_only_as_one_hot(model_and_inputs):
    model, inputs, batch = model_and_inputs
    model.eval()

    target = forward(model, inputs, batch["y"], seed=1)[TARGET]
    B, K, N = target.shape
    expected = torch.nn.functional.one_hot(batch["y"], N).float()

    assert torch.equal(target, expected.view(B, 1, N).expand(B, K, N))
