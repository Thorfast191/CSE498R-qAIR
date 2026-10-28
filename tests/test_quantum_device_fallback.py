"""
QUANTUM_DEVICE="auto" falls back to lightning.qubit when lightning.gpu
cannot be constructed OR cannot run a circuit (a CUDA 12 cuStateVec build
on Colab's CUDA 13 runtime, 2026-10-28), and the run manifest records the
device and the reason. An explicit "lightning.gpu" still raises.

    python -m pytest tests/test_quantum_device_fallback.py
"""

import pytest

pytest.importorskip("pennylane")

import models.quantum_layer as quantum_layer  # noqa: E402
from config import MODEL_DIM, N_QUBITS  # noqa: E402
from tests.test_cache_determinism import ARC  # noqa: E402
from tests.test_checkpoint_provenance import smoke_cache  # noqa: E402,F401

REAL_CONSTRUCT = quantum_layer._try_construct
CUDA_ERROR = "CUDA error: no kernel image is available for execution on the device"


@pytest.fixture
def gpu_constructs(monkeypatch):
    """A 'lightning.gpu' that constructs (backed by lightning.qubit)."""

    def construct(name, n_qubits):
        return REAL_CONSTRUCT("lightning.qubit", n_qubits)

    monkeypatch.setattr(quantum_layer, "_try_construct", construct)


@pytest.fixture
def gpu_fails_to_run(monkeypatch, gpu_constructs):
    real_probe = quantum_layer._probe
    probed = []

    def probe(dev):
        probed.append(dev)
        raise RuntimeError(CUDA_ERROR)

    monkeypatch.setattr(quantum_layer, "_probe", probe)
    yield probed
    monkeypatch.setattr(quantum_layer, "_probe", real_probe)


def layer(device="auto"):
    return quantum_layer.QuantumEvolutionLayer(MODEL_DIM, n_qubits=N_QUBITS,
                                               quantum_device=device)


def test_gpu_that_runs_is_used(gpu_constructs):
    built = layer()
    assert (built.device_name, built.device_fallback) == ("lightning.gpu", None)


def test_gpu_that_cannot_run_falls_back(gpu_fails_to_run):
    built = layer()

    assert len(gpu_fails_to_run) == 1                      # probed once, on the GPU device
    assert built.device_name == "lightning.qubit"
    assert CUDA_ERROR in built.device_fallback and "lightning.gpu" in built.device_fallback


def test_explicit_gpu_that_cannot_run_raises(gpu_fails_to_run):
    with pytest.raises(RuntimeError, match="requested explicitly"):
        layer("lightning.gpu")


def test_lightning_qubit_is_not_probed(monkeypatch):
    def probe(dev):
        raise AssertionError("lightning.qubit must not be probed")

    monkeypatch.setattr(quantum_layer, "_probe", probe)
    assert layer("lightning.qubit").device_fallback is None


def test_manifest_records_device_and_reason(tmp_path, smoke_cache, gpu_fails_to_run):
    from training.runner import run_training

    history = run_training(
        cache_dir=str(smoke_cache), ckpt_dir=str(tmp_path / "ckpt"), train_samples=16,
        val_samples=8, epochs=1, patience=None, with_test=False, arc_config=ARC,
    )

    manifest = history["manifest"]
    assert manifest["quantum_device"] == "lightning.qubit"
    assert CUDA_ERROR in manifest["quantum_device_fallback"]
