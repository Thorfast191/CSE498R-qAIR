"""
E12 -- the qAIR circuit as a batched torch statevector.

The same circuit as the PennyLane QNode in models/quantum_layer.py:

    Hadamard on every wire
    AngleEmbedding(inputs, rotation="Y")
    StronglyEntanglingLayers(weights)        # Rot(phi, theta, omega) per wire,
                                             # then CNOT(i, (i + r) % n), r = l % (n-1) + 1
    <X_i>, <Y_i>, <Z_i> for every wire

simulated exactly on a (rows, 2, ..., 2) complex128 state, all rows at
once, differentiated by torch autograd on whatever device the inputs are
on. PennyLane's TorchLayer runs one tape per row (512 per training step,
issues.md E12); this runs the whole batch as a few dozen tensor ops.

Wire 0 is the most significant qubit, as in PennyLane. The weights are a
float32 Parameter named "weights", initialised like TorchLayer's
(uniform on [0, 2*pi), drawn at the same point of model construction), so
checkpoints and same-seed models are interchangeable between the two
simulators. Outputs are returned as float32, as TorchLayer returns them.
"""

import math

import torch
import torch.nn as nn

# The embedding axis. "Y" is v46's; tests set "X" to reproduce the v40
# constant-circuit bug (tests/test_circuit_input_dependence.py).
ENCODING_ROTATION = "Y"

STATE_DTYPE = torch.complex128

_HADAMARD = torch.tensor([[1, 1], [1, -1]], dtype=STATE_DTYPE) / math.sqrt(2)


def rotation(axis, angle):
    """R_axis(angle) = exp(-i angle P / 2), shape (*angle.shape, 2, 2)."""

    c = torch.cos(angle / 2).to(STATE_DTYPE)
    s = torch.sin(angle / 2).to(STATE_DTYPE)
    zero = torch.zeros_like(c)

    if axis == "X":
        rows = [[c, -1j * s], [-1j * s, c]]
    elif axis == "Y":
        rows = [[c, -s], [s, c]]
    elif axis == "Z":
        rows = [[c - 1j * s, zero], [zero, c + 1j * s]]
    else:
        raise ValueError(f"rotation axis must be X, Y or Z, got {axis!r}")

    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def rot(phi, theta, omega):
    """PennyLane's Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)."""

    return rotation("Z", omega) @ rotation("Y", theta) @ rotation("Z", phi)


def apply_single(state, gate, wire):
    """Applies a 2x2 gate (shared, or one per row: (rows, 2, 2)) to `wire`."""

    axis = wire + 1
    moved = state.movedim(axis, -1)                    # (rows, ..., 2)
    if gate.dim() == 2:
        out = moved @ gate.transpose(-1, -2)
    else:
        shape = moved.shape
        flat = moved.reshape(shape[0], -1, 2)
        out = (flat @ gate.transpose(-1, -2)).reshape(shape)
    return out.movedim(-1, axis)


def apply_cnot(state, control, target):
    """CNOT: flips `target` on the control = 1 half of the state."""

    zero = state.select(control + 1, 0)
    one = state.select(control + 1, 1)
    target_axis = target + 1 if target < control else target
    return torch.stack([zero, one.flip(target_axis)], dim=control + 1)


def expectations(state, n_qubits):
    """[<X_0..X_{n-1}>, <Y_0..>, <Z_0..>], shape (rows, 3n), real."""

    rows = state.shape[0]
    xs, ys, zs = [], [], []

    for wire in range(n_qubits):
        a = state.select(wire + 1, 0).reshape(rows, -1)
        b = state.select(wire + 1, 1).reshape(rows, -1)
        overlap = (a.conj() * b).sum(-1)
        xs.append(2 * overlap.real)
        ys.append(2 * overlap.imag)
        zs.append((a.abs() ** 2).sum(-1) - (b.abs() ** 2).sum(-1))

    return torch.stack(xs + ys + zs, dim=-1)


def run_circuit(inputs, weights):
    """inputs (rows, n) angles, weights (layers, n, 3) -> (rows, 3n) float64."""

    rows, n_qubits = inputs.shape
    device = inputs.device

    # Product state after Hadamard + embedding: each wire is R(x) H |0>.
    ket0 = torch.tensor([1, 0], dtype=STATE_DTYPE, device=device)
    per_wire = rotation(ENCODING_ROTATION, inputs.double()) @ (_HADAMARD.to(device) @ ket0)

    state = per_wire[:, 0]
    for wire in range(1, n_qubits):
        state = (state.unsqueeze(-1) * per_wire[:, wire].unsqueeze(1)).reshape(rows, -1)
    state = state.reshape(rows, *([2] * n_qubits))

    weights = weights.double()
    for layer in range(weights.shape[0]):
        gates = rot(weights[layer, :, 0], weights[layer, :, 1], weights[layer, :, 2])
        for wire in range(n_qubits):
            state = apply_single(state, gates[wire], wire)
        if n_qubits > 1:
            r = layer % (n_qubits - 1) + 1
            for wire in range(n_qubits):
                state = apply_cnot(state, wire, (wire + r) % n_qubits)

    return expectations(state, n_qubits)


class TorchStatevectorCircuit(nn.Module):
    """Drop-in for the circuit's qml.qnn.TorchLayer: (rows, n) -> (rows, 3n)."""

    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        # TorchLayer: init(torch.Tensor(*shape)) with uniform_(b=2*pi).
        self.weights = nn.Parameter(
            nn.init.uniform_(torch.Tensor(n_layers, n_qubits, 3), b=2 * math.pi)
        )

    def forward(self, inputs):
        return run_circuit(inputs, self.weights).to(inputs.dtype)
