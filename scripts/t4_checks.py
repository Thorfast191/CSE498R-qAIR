"""
E10 -- the two determinism checks the Step 6 notebook runs on a T4
(docs/baseline/issues.md E10).

    python scripts/t4_checks.py determinism --cache-dir <smoke cache> \\
        --out-dir <new folder>
    python scripts/t4_checks.py quantum --out <report.json>

determinism  two same-seed 1-epoch runs on a smoke cache, each in its own
             checkpoint folder; compares their histories, their best
             checkpoints parameter by parameter and their per-example
             validation predictions. Strict deterministic mode
             (config.DETERMINISTIC) stays on: if an operation has no
             deterministic implementation, the error is printed and
             recorded and the script exits 2 -- it never retries with
             --deterministic-warn-only.
quantum      the circuit of QuantumEvolutionLayer with identical weights
             and inputs: twice on the PennyLane device under test (outputs
             and gradients compared bit for bit), against lightning.qubit
             (largest absolute differences), and the torch statevector
             (E12) against lightning.qubit, with seconds per pass for each.
             Exits 3 when the device under test cannot be constructed.

Both write a JSON report and only read the cache; neither decides what to
do about a difference.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from config import EMBEDDING_DIM, MODEL_DIM, N_QUBITS, SEED  # noqa: E402

RUNS = ("run_a", "run_b")
REFERENCE_DEVICE = "lightning.qubit"
STRICT_RAISED = 2
DEVICE_UNAVAILABLE = 3


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def differences(a, b, path=""):
    """Paths at which two nested history values differ (tensors bit for bit)."""

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        same = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
        return [] if same else [path]

    if isinstance(a, dict) and isinstance(b, dict):
        found = [f"{path}/{k}" for k in sorted(set(a) ^ set(b), key=str)]
        for k in sorted(set(a) & set(b), key=str):
            found += differences(a[k], b[k], f"{path}/{k}")
        return found

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [path]
        found = []
        for i, (x, y) in enumerate(zip(a, b)):
            found += differences(x, y, f"{path}[{i}]")
        return found

    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []

    return [] if a == b else [path]


def validation_predictions(history, ckpt_dir, cache_dir, n_train, n_val, arc_config):
    """Per-example predicted option and scores of the run's best checkpoint."""

    from models.full_model import QAIRvNext
    from training.checkpoint import load_best
    from training.evaluate import batch_to_model
    from training.runner import build_loaders, device

    provenance = history["provenance"]
    run_name = history["manifest"]["run"]

    model = QAIRvNext(dim=EMBEDDING_DIM, **provenance["model_config"]).to(device)
    if not load_best(model, ckpt_dir, run_name, provenance, device):
        raise FileNotFoundError(f"no best checkpoint for {run_name} in {ckpt_dir}")

    _, val_loader, _ = build_loaders(
        cache_dir, n_train, n_val, seed=provenance["seed"], with_test=False,
        arc_config=arc_config,
    )

    model.eval()
    predictions, scores = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(**batch_to_model(batch, device))
            predictions += out["scores"].argmax(dim=1).cpu().tolist()
            scores.append(out["scores"].cpu())

    return predictions, torch.cat(scores)


def best_state(ckpt_dir, run_name):
    from training.checkpoint import load_checkpoint_file

    return load_checkpoint_file(os.path.join(ckpt_dir, f"{run_name}_best.pt"))["model"]


def compare_runs(runs, ckpt_dirs, cache_dir, n_train, n_val, arc_config):
    (hist_a, hist_b), (dir_a, dir_b) = runs, ckpt_dirs

    pred_a, scores_a = validation_predictions(hist_a, dir_a, cache_dir, n_train, n_val, arc_config)
    pred_b, scores_b = validation_predictions(hist_b, dir_b, cache_dir, n_train, n_val, arc_config)

    state_a = best_state(dir_a, hist_a["manifest"]["run"])
    state_b = best_state(dir_b, hist_b["manifest"]["run"])

    return {
        "history_differences": differences(hist_a, hist_b),
        "parameter_differences": differences(state_a, state_b),
        "n_validation": len(pred_a),
        "differing_predictions": [i for i, (x, y) in enumerate(zip(pred_a, pred_b)) if x != y],
        "max_score_abs_diff": (scores_a - scores_b).abs().max().item(),
    }


def run_determinism(cache_dir, out_dir, n_train=20, n_val=10, arc_config="ARC-Easy",
                    seed=SEED):
    from training.runner import run_training

    os.makedirs(out_dir, exist_ok=False)
    report = {"seed": seed, "arc_config": arc_config, "cache_dir": cache_dir,
              "questions": {"train": n_train, "validation": n_val}}

    histories, ckpt_dirs = [], []
    for run in RUNS:
        ckpt_dir = os.path.join(out_dir, run)
        os.makedirs(ckpt_dir)
        try:
            histories.append(run_training(
                cache_dir=cache_dir, ckpt_dir=ckpt_dir, train_samples=n_train,
                val_samples=n_val, epochs=1, patience=None, seed=seed,
                with_test=False, arc_config=arc_config,
            ))
        except RuntimeError as error:
            if "deterministic" not in str(error):
                raise
            report["strict_mode_error"] = {"run": run, "message": str(error)}
            return report
        ckpt_dirs.append(ckpt_dir)

    report.update(compare_runs(histories, ckpt_dirs, cache_dir, n_train, n_val, arc_config))
    report["manifest"] = histories[0]["manifest"]
    report["identical"] = not (report["history_differences"]
                               or report["parameter_differences"]
                               or report["differing_predictions"]
                               or report["max_score_abs_diff"])
    return report


def print_determinism(report):
    print("\n" + "=" * 66)
    print("E10 DETERMINISM PAIR (two same-seed 1-epoch runs)")
    print("=" * 66)

    if "strict_mode_error" in report:
        error = report["strict_mode_error"]
        print(f"[STRICT DETERMINISM RAISED] in {error['run']}:\n{error['message']}")
        print("Stop and report; no automatic retry with --deterministic-warn-only.")
        return

    print(f"device {report['manifest']['device_name']}, "
          f"quantum {report['manifest']['quantum_device']}")
    print(f"history differences   : {len(report['history_differences'])} "
          f"{report['history_differences'][:10]}")
    print(f"parameter differences : {len(report['parameter_differences'])} "
          f"{report['parameter_differences'][:10]}")
    print(f"differing predictions : {len(report['differing_predictions'])} of "
          f"{report['n_validation']} {report['differing_predictions'][:20]}")
    print(f"max |score_a - score_b|: {report['max_score_abs_diff']:.3e}")
    print(f"IDENTICAL: {report['identical']}")


# ---------------------------------------------------------------------------
# quantum
# ---------------------------------------------------------------------------

def circuit_pass(layer, z, readout, seconds=None):
    """
    Outputs, input gradient and weight gradient of one forward/backward;
    appends its wall-clock seconds to `seconds` when given.
    """

    import time

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    layer.zero_grad()
    z = z.detach().clone().requires_grad_(True)
    out = layer.quantum(z)
    (out * readout).sum().backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if seconds is not None:
        seconds.append(time.perf_counter() - start)

    return {
        "outputs": out.detach().cpu(),
        "input_grad": z.grad.detach().cpu(),
        "weight_grad": layer.quantum.weights.grad.detach().cpu(),
    }


def build_layer(name, state=None, simulator="pennylane"):
    from models.quantum_layer import QuantumEvolutionLayer

    layer = QuantumEvolutionLayer(MODEL_DIM, n_qubits=N_QUBITS, quantum_device=name,
                                  quantum_simulator=simulator)
    if state is not None:
        layer.load_state_dict(state)
    return layer


def max_abs(a, b):
    return {k: (a[k] - b[k]).abs().max().item() for k in a}


def run_quantum(device_name="lightning.gpu", n_inputs=128, seed=0):
    from training.runner import device

    torch.manual_seed(seed)
    reference = build_layer(REFERENCE_DEVICE).to(device)
    state = reference.state_dict()

    z = (math.pi * torch.tanh(torch.randn(n_inputs, N_QUBITS))).to(device)
    readout = torch.randn(n_inputs, 3 * N_QUBITS).to(device)

    report = {"device": device_name, "reference": REFERENCE_DEVICE,
              "torch_device": str(device), "n_inputs": n_inputs, "seed": seed}

    try:
        layer = build_layer(device_name, state).to(device)
    except Exception as error:   # noqa: BLE001 -- recorded, the caller decides
        report["unavailable"] = f"{type(error).__name__}: {error}"
        return report
    report["constructed_device"] = layer.device_name
    report["reference_constructed_device"] = reference.device_name
    # The device each circuit actually executes on, read from its QNode.
    report["qnode_devices"] = {"tested": layer.quantum.qnode.device.name,
                               "reference": reference.quantum.qnode.device.name}

    statevector = build_layer(REFERENCE_DEVICE, state, simulator="torch").to(device)
    statevector_seconds = []
    circuit_pass(statevector, z, readout, statevector_seconds)
    from_statevector = circuit_pass(statevector, z, readout, statevector_seconds)

    tested_seconds, reference_seconds = [], []
    first = circuit_pass(layer, z, readout, tested_seconds)
    second = circuit_pass(layer, z, readout, tested_seconds)
    expected = circuit_pass(reference, z, readout, reference_seconds)
    circuit_pass(reference, z, readout, reference_seconds)
    # The second pass of each, so one-time initialisation is not counted.
    report["seconds_per_pass"] = {"tested": tested_seconds[1],
                                  "reference": reference_seconds[1],
                                  "torch_statevector": statevector_seconds[1]}

    report["repeat_bit_identical"] = {k: torch.equal(first[k], second[k]) for k in first}
    report["repeat_max_abs_diff"] = max_abs(first, second)
    report["vs_reference_max_abs_diff"] = max_abs(first, expected)
    report["torch_statevector_vs_reference_max_abs_diff"] = max_abs(from_statevector, expected)
    report["dtype"] = str(first["outputs"].dtype)
    return report


def print_quantum(report):
    print("\n" + "=" * 66)
    print(f"E10 QUANTUM DEVICE CHECK ({report['device']} vs {report['reference']})")
    print("=" * 66)

    if "unavailable" in report:
        print(f"[UNAVAILABLE] {report['unavailable']}")
        return

    print(f"constructed device : {report['constructed_device']}  "
          f"(torch device {report['torch_device']}, {report['dtype']})")
    print(f"QNode devices      : tested {report['qnode_devices']['tested']}, "
          f"reference {report['qnode_devices']['reference']}")
    print(f"seconds per pass   : tested {report['seconds_per_pass']['tested']:.3f}, "
          f"reference {report['seconds_per_pass']['reference']:.3f}, "
          f"torch statevector {report['seconds_per_pass']['torch_statevector']:.4f} "
          f"({report['n_inputs']} rows, forward + backward)")
    print("torch statevector vs " + report["reference"] + " max|d| " + ", ".join(
        f"{k} {v:.3e}" for k, v in report["torch_statevector_vs_reference_max_abs_diff"].items()))
    for key in ("outputs", "input_grad", "weight_grad"):
        print(f"{key:<12s} repeat bit-identical {str(report['repeat_bit_identical'][key]):<5s} "
              f"repeat max|d| {report['repeat_max_abs_diff'][key]:.3e}  "
              f"vs {report['reference']} max|d| {report['vs_reference_max_abs_diff'][key]:.3e}")
    print("Reading aid: float32 machine epsilon is 1.19e-07, float64 2.22e-16. TorchLayer "
          "returns float32, so float64-level differences between the devices round away.")


# ---------------------------------------------------------------------------

def write_report(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Report: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="check", required=True)

    det = sub.add_parser("determinism")
    det.add_argument("--cache-dir", required=True, help="An existing smoke cache (read only).")
    det.add_argument("--out-dir", required=True, help="New folder for both runs and the report.")
    det.add_argument("--train", type=int, default=20)
    det.add_argument("--val", type=int, default=10)
    det.add_argument("--arc-config", default="ARC-Easy",
                     choices=["ARC-Challenge", "ARC-Easy", "merged"])
    det.add_argument("--seed", type=int, default=SEED)

    qua = sub.add_parser("quantum")
    qua.add_argument("--device", default="lightning.gpu")
    qua.add_argument("--n-inputs", type=int, default=128,
                     help="Rows per pass (one training step: batch 16 x 8 hypotheses).")
    qua.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.check == "determinism":
        report = run_determinism(args.cache_dir, args.out_dir, args.train, args.val,
                                 args.arc_config, args.seed)
        print_determinism(report)
        write_report(report, os.path.join(args.out_dir, "determinism.json"))
        return STRICT_RAISED if "strict_mode_error" in report else 0

    report = run_quantum(args.device, args.n_inputs)
    print_quantum(report)
    write_report(report, args.out)
    return DEVICE_UNAVAILABLE if "unavailable" in report else 0


if __name__ == "__main__":
    sys.exit(main())
