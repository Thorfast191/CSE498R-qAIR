"""
E10 -- scripts/t4_checks.py, the determinism pair and the quantum-device
check the Step 6 notebook runs on a T4 (docs/baseline/issues.md E10).

    python -m pytest tests/test_t4_checks.py

Real model and circuit on CPU with the fake cache from
test_checkpoint_provenance; needs pennylane.
"""

import json
import os
import sys

import pytest
import torch

import training.runner as runner_module
from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import t4_checks  # noqa: E402

N_TRAIN, N_VAL = 16, 8


def determinism(tmp_path, cache_dir):
    return t4_checks.main([
        "determinism", "--cache-dir", str(cache_dir), "--out-dir", str(tmp_path / "pair"),
        "--train", str(N_TRAIN), "--val", str(N_VAL), "--arc-config", ARC,
    ])


def read_report(tmp_path):
    with open(tmp_path / "pair" / "determinism.json") as f:
        return json.load(f)


def test_same_seed_pair_is_identical(tmp_path, smoke_cache, capsys):
    assert determinism(tmp_path, smoke_cache) == 0

    report = read_report(tmp_path)
    assert report["identical"] is True
    assert report["n_validation"] == N_VAL
    assert report["history_differences"] == report["parameter_differences"] == []
    assert report["differing_predictions"] == [] and report["max_score_abs_diff"] == 0
    assert report["manifest"]["seed"] == 42
    for run in t4_checks.RUNS:
        assert os.path.isdir(tmp_path / "pair" / run)
    assert "IDENTICAL: True" in capsys.readouterr().out


def test_a_different_run_is_reported(tmp_path, smoke_cache, monkeypatch):
    real = runner_module.run_training
    calls = []

    def second_run_uses_another_seed(**kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 2:
            kwargs["seed"] = 43
        return real(**kwargs)

    monkeypatch.setattr(runner_module, "run_training", second_run_uses_another_seed)
    determinism(tmp_path, smoke_cache)

    report = read_report(tmp_path)
    assert report["identical"] is False
    assert report["parameter_differences"] and report["history_differences"]
    assert report["max_score_abs_diff"] > 0


def flip_third(predictions, scores):
    return predictions[:3] + [predictions[3] + 1] + predictions[4:], scores


def nudge_scores(predictions, scores):
    return predictions, scores + 1e-6


@pytest.mark.parametrize("change, differing", [(flip_third, [3]), (nudge_scores, [])])
def test_prediction_and_score_differences_count(tmp_path, smoke_cache, monkeypatch,
                                                 change, differing):
    real = t4_checks.validation_predictions
    calls = []

    def second_run_changed(*args):
        calls.append(1)
        result = real(*args)
        return change(*result) if len(calls) == 2 else result

    monkeypatch.setattr(t4_checks, "validation_predictions", second_run_changed)
    determinism(tmp_path, smoke_cache)

    report = read_report(tmp_path)
    assert report["parameter_differences"] == report["history_differences"] == []
    assert report["differing_predictions"] == differing
    assert report["identical"] is False


def test_strict_mode_error_stops_without_retry(tmp_path, smoke_cache, monkeypatch, capsys):
    calls = []

    def raises(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("index_add_cuda_ does not have a deterministic implementation, "
                           "but you set 'torch.use_deterministic_algorithms(True)'")

    monkeypatch.setattr(runner_module, "run_training", raises)

    assert determinism(tmp_path, smoke_cache) == t4_checks.STRICT_RAISED
    assert len(calls) == 1
    assert "deterministic_warn_only" not in calls[0]
    assert read_report(tmp_path)["strict_mode_error"]["run"] == "run_a"
    assert "[STRICT DETERMINISM RAISED]" in capsys.readouterr().out


def test_other_runtime_errors_are_not_swallowed(tmp_path, smoke_cache, monkeypatch):
    def raises(**kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runner_module, "run_training", raises)

    with pytest.raises(RuntimeError, match="out of memory"):
        determinism(tmp_path, smoke_cache)


def test_differences_finds_every_kind():
    a = {"loss": [1.0, 0.5], "w": torch.zeros(2), "m": {"x": 1}, "nan": float("nan")}
    b = {"loss": [1.0, 0.25], "w": torch.tensor([0.0, 1e-9]), "m": {"y": 1},
         "nan": float("nan")}

    assert t4_checks.differences(a, a) == []
    assert t4_checks.differences(a, b) == ["/loss[1]", "/m/x", "/m/y", "/w"]
    assert t4_checks.differences(torch.zeros(2), torch.zeros(2, dtype=torch.float64)) == [""]


def quantum(tmp_path, device):
    return t4_checks.main(["quantum", "--device", device, "--n-inputs", "8",
                           "--out", str(tmp_path / "quantum.json")])


@pytest.fixture
def cpu(monkeypatch):
    pytest.importorskip("pennylane")
    monkeypatch.setattr(runner_module, "device", "cpu")


def test_quantum_check_on_the_reference_device(tmp_path, cpu, capsys):
    assert quantum(tmp_path, "lightning.qubit") == 0

    with open(tmp_path / "quantum.json") as f:
        report = json.load(f)
    keys = {"outputs", "input_grad", "weight_grad"}
    assert report["constructed_device"] == "lightning.qubit"
    assert report["reference_constructed_device"] == "lightning.qubit"
    assert report["qnode_devices"] == {"tested": "lightning.qubit", "reference": "lightning.qubit"}
    assert all(s > 0 for s in report["seconds_per_pass"].values())
    assert report["repeat_bit_identical"] == dict.fromkeys(keys, True)
    assert report["vs_reference_max_abs_diff"] == dict.fromkeys(keys, 0.0)
    assert "QUANTUM DEVICE CHECK" in capsys.readouterr().out


def test_quantum_check_sees_different_weights(tmp_path, cpu, monkeypatch):
    real = t4_checks.build_layer
    monkeypatch.setattr(t4_checks, "build_layer", lambda name, state=None: real(name))

    quantum(tmp_path, "lightning.qubit")

    with open(tmp_path / "quantum.json") as f:
        report = json.load(f)
    assert all(v > 0 for v in report["vs_reference_max_abs_diff"].values())


def test_unavailable_device_is_reported(tmp_path, cpu):
    try:
        t4_checks.build_layer("lightning.gpu")
        pytest.skip("lightning.gpu is installed here")
    except Exception:  # noqa: BLE001
        pass

    assert quantum(tmp_path, "lightning.gpu") == t4_checks.DEVICE_UNAVAILABLE
    with open(tmp_path / "quantum.json") as f:
        assert "unavailable" in json.load(f)
