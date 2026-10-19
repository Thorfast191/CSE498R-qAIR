"""
A1/A2 -- removing the parameters that never received gradient does not
change the forward pass (docs/baseline/issues.md).

The code before the change (REFERENCE_COMMIT) is extracted with
`git archive` into a temporary directory and run in a subprocess. For
every arm it builds the model with a fixed seed, saves its weights and its
scores on a fixed synthetic batch; for ckpt/qair_v46_best.pt it saves the
scores of the v46 weights. This code then loads the same weights (format-1
semantics: the removed keys are dropped) and must give bit-identical
scores. The test also measures which initial weights a fixed seed changes.

    python -m pytest tests/test_forward_unchanged.py -s
"""

import json
import os
import subprocess
import sys
import tarfile
import io

import pytest
import torch

from tests.test_checkpoint_provenance import REPO_ROOT
from tests.test_gradient_coverage import ARMS
from training.checkpoint import load_checkpoint_file, load_model_state

REFERENCE_COMMIT = "35243a1"   # last commit before A1/A2

V46_CHECKPOINT = os.path.join(REPO_ROOT, "ckpt", "qair_v46_best.pt")
V46_CONFIG = os.path.join(REPO_ROOT, "ckpt", "qair_v46_config.pt")

# Runs inside the extracted old tree, so it may only use APIs that exist
# both before and after the change.
REFERENCE_SCRIPT = r'''
import json, os, sys
import torch
import training.ablations as ablations
from config import EMBEDDING_DIM
from models.full_model import QAIRvNext

ablations.device = "cpu"
out_dir, v46_ckpt, v46_config = sys.argv[1:4]

g = torch.Generator().manual_seed(1234)
B, N, D = 4, 4, EMBEDDING_DIM
K = 2 * N
O_mask = torch.ones(B, N, dtype=torch.bool)
O_mask[1, 3] = False                                   # one 3-option question
hyp_option = torch.arange(K) % N
H_mask = O_mask[:, hyp_option]
batch = {
    "H": torch.randn(B, K, D, generator=g),
    "O": torch.randn(B, N, D, generator=g),
    "Q": torch.randn(B, D, generator=g),
    "H_mask": H_mask,
    "O_mask": O_mask,
    "polarity": torch.where(torch.arange(K) < N, 1.0, -1.0).expand(B, K).clone(),
    "align": torch.nn.functional.one_hot(hyp_option, N).float().expand(B, K, N).clone(),
    "llm_logprob": torch.randn(B, N, generator=g),
}
torch.save(batch, os.path.join(out_dir, "batch.pt"))

def scores(model):
    model.eval()
    with torch.no_grad():
        return model(**batch)["scores"]

for arm in json.loads(os.environ["ARMS"]):
    torch.manual_seed(0)
    model = ablations.build_model(ablations.ABLATIONS[arm])
    torch.save({"model": model.state_dict(), "scores": scores(model)},
               os.path.join(out_dir, f"{arm}.pt"))

if os.path.exists(v46_ckpt):
    cfg = torch.load(v46_config, map_location="cpu", weights_only=True)
    keys = ("model_dim", "use_quantum", "use_validator", "persistent_steps",
            "n_qubits", "use_question", "phase_mode", "phase_source",
            "mixing", "use_llm_prior", "use_attack")
    model = QAIRvNext(dim=EMBEDDING_DIM, **{k: cfg[k] for k in keys})
    model.load_state_dict(torch.load(v46_ckpt, map_location="cpu",
                                     weights_only=True)["model"])
    torch.save({"scores": scores(model)}, os.path.join(out_dir, "v46.pt"))
'''


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    pytest.importorskip("pennylane")

    root = tmp_path_factory.mktemp("reference")
    tree, out = root / "tree", root / "out"
    tree.mkdir()
    out.mkdir()

    try:
        archive = subprocess.run(
            ["git", "-C", REPO_ROOT, "archive", REFERENCE_COMMIT],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git archive {REFERENCE_COMMIT} unavailable")

    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree, filter="data")

    (tree / "reference_script.py").write_text(REFERENCE_SCRIPT)

    subprocess.run(
        [sys.executable, "reference_script.py", str(out), V46_CHECKPOINT, V46_CONFIG],
        cwd=tree, check=True, capture_output=True,
        env={**os.environ, "ARMS": json.dumps(ARMS)},
    )

    return out


@pytest.fixture(scope="module")
def cpu_build():
    import training.ablations as ablations

    device, ablations.device = ablations.device, "cpu"
    yield ablations
    ablations.device = device


def scores(model, batch):
    model.eval()
    with torch.no_grad():
        return model(**batch)["scores"]


@pytest.mark.parametrize("arm", ARMS)
def test_same_weights_give_bit_identical_scores(arm, reference, cpu_build):
    batch = torch.load(reference / "batch.pt", weights_only=True)
    old = torch.load(reference / f"{arm}.pt", weights_only=True)

    torch.manual_seed(0)
    model = cpu_build.build_model(cpu_build.ABLATIONS[arm])
    load_model_state(model, {"model": old["model"]}, f"{REFERENCE_COMMIT}:{arm}")

    assert torch.equal(scores(model, batch), old["scores"])


def test_v46_checkpoint_gives_bit_identical_scores(reference):
    from evaluation.input_ablation import build_model_from_config, resolve_model_config

    if not os.path.exists(reference / "v46.pt"):
        pytest.skip(f"{V46_CHECKPOINT} not on this machine")

    batch = torch.load(reference / "batch.pt", weights_only=True)
    ckpt = load_checkpoint_file(V46_CHECKPOINT)
    model = build_model_from_config(resolve_model_config(ckpt, V46_CHECKPOINT), "cpu")
    load_model_state(model, ckpt, V46_CHECKPOINT)

    old = torch.load(reference / "v46.pt", weights_only=True)
    assert torch.equal(scores(model, batch), old["scores"])


@pytest.mark.parametrize("arm", ARMS)
def test_initial_weights_for_a_fixed_seed(arm, reference, cpu_build):
    """
    Measurement, recorded in issues.md A1/A2: the removed modules consumed
    random numbers during construction, so only modules built after them
    can change. The only such module is phase_head.
    """

    old = torch.load(reference / f"{arm}.pt", weights_only=True)["model"]

    torch.manual_seed(0)
    new = cpu_build.build_model(cpu_build.ABLATIONS[arm]).state_dict()

    changed = sorted(k for k in new if not torch.equal(new[k], old[k]))
    print(f"\n[INIT] {arm}: {len(new)} tensors, changed {changed}")

    assert set(changed) <= {"phase_head.weight", "phase_head.bias"}


# ---------------------------------------------------------------------------
# Checkpoint format 3 compatibility
# ---------------------------------------------------------------------------

def test_key_drop_loader_drops_only_known_keys_from_old_formats():
    from training.checkpoint import load_model_state

    model = torch.nn.Linear(2, 1)
    state = dict(model.state_dict())

    load_model_state(model, {"model": {**state, "llm_prior_weight": torch.tensor(-1.0)}},
                     "format-1 file")

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_model_state(model, {"model": {**state, "unknown.weight": torch.zeros(1)}},
                         "format-1 file")

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_model_state(model, {"format": 3, "model": {
            **state, "llm_prior_weight": torch.tensor(-1.0)}}, "format-3 file")


def test_format2_checkpoint_is_not_resumed(tmp_path):
    from tests.test_checkpoint_provenance import latest, provenance, save, tiny_trainer
    from training.checkpoint import CheckpointMismatchError, load_or_resume

    prov = provenance()
    save(tiny_trainer(tmp_path, prov), 0, 0.5)
    ckpt = load_checkpoint_file(latest(tmp_path))
    ckpt["format"] = 2
    torch.save(ckpt, latest(tmp_path))

    with pytest.raises(CheckpointMismatchError, match="format 2"):
        load_or_resume(tiny_trainer(tmp_path, prov), str(tmp_path),
                       "qair_ARC-Easy_s42", epochs=8, provenance=prov)
