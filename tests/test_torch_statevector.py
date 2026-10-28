"""
E12 -- the torch statevector computes the PennyLane circuit
(docs/baseline/issues.md E12).

    python -m pytest tests/test_torch_statevector.py

Against PennyLane lightning.qubit: outputs and gradients (inputs and
weights) to 1e-6 in float64 and 1e-5 through the float32 layers; same-seed
layers have identical weights under either simulator; the full forward of
every arm matches. The A0 input-dependence tests run under both simulators
(tests/test_circuit_input_dependence.py); same-seed determinism is covered
by tests/test_determinism_resume.py, which runs on the default (torch).
"""

import math

import pytest
import torch

qml = pytest.importorskip("pennylane")

import config  # noqa: E402
from config import EMBEDDING_DIM, MODEL_DIM, N_QUBITS  # noqa: E402
from models.quantum_layer import QuantumEvolutionLayer  # noqa: E402
from models.statevector import run_circuit  # noqa: E402
from tests.test_gradient_coverage import ARMS, batch  # noqa: E402,F401
from tests.test_checkpoint_provenance import smoke_cache  # noqa: E402,F401

N_LAYERS = 3
ROWS = 32
SEEDS = (0, 1, 2)


def layer(simulator, seed=0):
    torch.manual_seed(seed)
    return QuantumEvolutionLayer(MODEL_DIM, n_qubits=N_QUBITS, quantum_device="lightning.qubit",
                                 quantum_simulator=simulator)


def random_problem(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    inputs = (math.pi * torch.tanh(torch.randn(ROWS, N_QUBITS, generator=g))).to(dtype)
    weights = (2 * math.pi * torch.rand(N_LAYERS, N_QUBITS, 3, generator=g)).to(dtype)
    readout = torch.randn(ROWS, 3 * N_QUBITS, generator=g).to(dtype)
    return inputs, weights, readout


def outputs_and_grads(fn, inputs, weights, readout):
    inputs = inputs.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    out = fn(inputs, weights)
    (out * readout).sum().backward()
    return out.detach(), inputs.grad, weights.grad


def max_diffs(a, b):
    return [(x.double() - y.double()).abs().max().item() for x, y in zip(a, b)]


@pytest.mark.parametrize("seed", SEEDS)
def test_float64_matches_pennylane(seed):
    qnode = layer("pennylane").quantum.qnode

    def reference(inputs, weights):
        return torch.stack(list(qnode(inputs, weights)), dim=-1)

    problem = random_problem(seed, torch.float64)
    expected = outputs_and_grads(reference, *problem)
    got = outputs_and_grads(run_circuit, *problem)

    assert max(max_diffs(got, expected)) <= 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_float32_layers_match(seed):
    inputs, _, readout = random_problem(seed, torch.float32)
    results = []
    for simulator in ("pennylane", "torch"):
        circuit = layer(simulator, seed).quantum
        x = inputs.clone().requires_grad_(True)
        out = circuit(x)
        (out * readout).sum().backward()
        assert out.dtype == torch.float32
        results.append((out.detach(), x.grad, circuit.weights.grad))

    assert max(max_diffs(*results)) <= 1e-5


def test_same_seed_layers_are_interchangeable():
    torch_state = layer("torch").state_dict()
    pennylane_state = layer("pennylane").state_dict()

    assert torch_state.keys() == pennylane_state.keys()
    assert all(torch.equal(torch_state[k], pennylane_state[k]) for k in torch_state)


def test_device_name_and_no_pennylane_device():
    built = layer("torch")
    assert (built.device_name, built.device_fallback) == ("torch.statevector", None)


def test_unknown_simulator_raises():
    with pytest.raises(ValueError, match="quantum_simulator"):
        layer("qiskit")


def build_arm(arm, simulator, monkeypatch):
    from training.ablations import ABLATIONS, build_model

    monkeypatch.setattr(config, "QUANTUM_SIMULATOR", simulator)
    torch.manual_seed(0)
    return build_model(ABLATIONS[arm]).eval()


@pytest.mark.parametrize("arm", ARMS)
def test_full_forward_of_every_arm_matches(arm, batch, monkeypatch):
    from training.evaluate import batch_to_model

    outputs = []
    for simulator in ("pennylane", "torch"):
        model = build_arm(arm, simulator, monkeypatch)
        with torch.no_grad():
            outputs.append(model(**batch_to_model(batch, "cpu")))

    reference, got = outputs
    compared = 0
    for key, value in reference.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.allclose(got[key], value, atol=1e-5, rtol=1e-5), key
            compared += 1
    assert compared > 0
