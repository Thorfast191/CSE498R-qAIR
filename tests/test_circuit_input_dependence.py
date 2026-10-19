"""
A0 regression test -- the quantum circuit's output depends on its input
(docs/baseline/issues.md A0; DoD §2.1).

Hadamard followed by AngleEmbedding(rotation="X") makes every input a
global phase, so the circuit is a learned constant (the v40 bug, still on
local `main`). v46 uses rotation="Y" (models/quantum_layer.py). The
negative control puts the X rotation back into the real layer code and
checks that this test would have caught it.

    python -m pytest tests/test_circuit_input_dependence.py

CPU (lightning.qubit), N_QUBITS from config; needs pennylane.
"""

import math

import pytest
import torch

from config import EMBEDDING_DIM, N_QUBITS

qml = pytest.importorskip("pennylane")

N_INPUTS = 64
WEIGHT_SEEDS = (0, 1, 2)
MIN_STD = 1e-3        # DoD §2.1, per observable
CONSTANT_STD = 1e-12  # what the v40 circuit measures (float rounding only)


def build_layer(seed=0):
    from models.quantum_layer import QuantumEvolutionLayer

    torch.manual_seed(seed)
    layer = QuantumEvolutionLayer(EMBEDDING_DIM, n_qubits=N_QUBITS,
                                  quantum_device="lightning.qubit")
    return layer.eval()


def random_angles(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return (torch.rand(N_INPUTS, N_QUBITS, generator=g) * 2 - 1) * math.pi


def circuit_std(layer, seed):
    """Std over N_INPUTS distinct inputs, one value per observable."""

    with torch.no_grad():
        out = layer.quantum(random_angles(seed))

    assert out.shape == (N_INPUTS, 3 * N_QUBITS)
    return out.std(dim=0)


def phase_std(layer):
    """Std of the layer's circuit-driven phase over N_INPUTS hypotheses."""

    g = torch.Generator().manual_seed(7)
    H = torch.randn(1, N_INPUTS, EMBEDDING_DIM, generator=g)

    with torch.no_grad():
        _, _, phase = layer(H)

    return phase.std().item()


@pytest.fixture
def x_rotation(monkeypatch):
    """The v40 bug: AngleEmbedding with its default rotation="X"."""

    real = qml.AngleEmbedding

    def angle_embedding(features, wires, rotation="X", **kwargs):
        return real(features, wires, rotation="X", **kwargs)

    monkeypatch.setattr(qml, "AngleEmbedding", angle_embedding)


@pytest.mark.parametrize("seed", WEIGHT_SEEDS)
def test_every_observable_varies_with_the_input(seed):
    std = circuit_std(build_layer(seed), seed)

    assert (std > MIN_STD).all(), f"constant observables: {std.tolist()}"


def test_layer_phase_varies_with_the_hypothesis():
    assert phase_std(build_layer()) > MIN_STD


@pytest.mark.parametrize("seed", WEIGHT_SEEDS)
def test_x_rotation_bug_is_detected(seed, x_rotation):
    std = circuit_std(build_layer(seed), seed)

    assert (std < CONSTANT_STD).all()


def test_x_rotation_bug_makes_layer_phase_constant(x_rotation):
    assert phase_std(build_layer()) < CONSTANT_STD
