"""
B3 -- validation selects the checkpoint; test is evaluated exactly once
per run, on that checkpoint, and the result says which epoch it came from
(docs/baseline/issues.md B3).

    python -m pytest tests/test_test_selection.py

Real model on CPU with a fake cache that also has a test split.
"""

import pytest

import training.dataset as dataset_module
import training.runner as runner_module
from config import cache_filename
from tests.test_cache_determinism import ARC, FakeEncoder, FakeGenerator, fake_rows
from training.checkpoint import NoBestCheckpointError, run_key
from training.dataset import QAIRDataset

pytest.importorskip("pennylane")

RUN = run_key("qair", ARC, 42)


@pytest.fixture
def cache_with_test(tmp_path, monkeypatch):
    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    rows = fake_rows()
    monkeypatch.setattr(dataset_module, "load_arc",
                        lambda cfg: {s: rows for s in ("train", "validation", "test")})

    cache_dir = tmp_path / "cache"
    for split in ("train", "validation", "test"):
        QAIRDataset(split=split, cache_dir=str(cache_dir), arc_config=ARC, allow_build=True)

    import training.ablations as ablations_module
    monkeypatch.setattr(runner_module, "device", "cpu")
    monkeypatch.setattr(ablations_module, "device", "cpu")
    return cache_dir


@pytest.fixture
def test_evaluations(monkeypatch):
    """Records every evaluate() call made on a test loader."""

    calls = []
    for module_name in ("training.runner", "training.ablations"):
        module = __import__(module_name, fromlist=["evaluate"])
        real = module.evaluate

        def counting(model, loader, device, real=real, **kwargs):
            if loader.dataset.cache_path.endswith(cache_filename("test", ARC)):
                calls.append(sum(p.detach().sum().item() for p in model.parameters()))
            return real(model, loader, device, **kwargs)

        monkeypatch.setattr(module, "evaluate", counting)
    return calls


def train(tmp_path, cache_dir, epochs=3):
    return runner_module.run_training(
        cache_dir=str(cache_dir), ckpt_dir=str(tmp_path / "ckpt"), train_samples=16,
        val_samples=8, epochs=epochs, patience=None, seed=42, with_test=True, arc_config=ARC,
    )


def test_test_comes_once_from_the_selected_epoch(tmp_path, cache_with_test, test_evaluations):
    history = train(tmp_path, cache_with_test)

    val = history["acc"]
    selected = val.index(max(val)) + 1                  # first best, as the trainer keeps it
    assert history["test_from"] == {"checkpoint": f"{RUN}_best.pt", "epoch": selected,
                                    "val_acc": max(val)}
    assert len(test_evaluations) == 1
    assert "acc" in history["test"] and len(history["test_records"]) > 0


def test_test_number_is_the_best_checkpoints(tmp_path, cache_with_test):
    from models.full_model import QAIRvNext
    from training.checkpoint import load_best
    from training.evaluate import evaluate
    from config import EMBEDDING_DIM

    history = train(tmp_path, cache_with_test)

    model = QAIRvNext(dim=EMBEDDING_DIM, **history["provenance"]["model_config"])
    assert load_best(model, str(tmp_path / "ckpt"), RUN, history["provenance"], "cpu")
    _, _, test_loader = runner_module.build_loaders(str(cache_with_test), 16, 8, seed=42,
                                                    with_test=True, arc_config=ARC)
    metrics, records = evaluate(model, test_loader, "cpu", return_records=True)

    assert metrics["acc"] == history["test"]["acc"] and records == history["test_records"]


def test_no_best_checkpoint_raises_before_test(tmp_path, cache_with_test, test_evaluations,
                                               monkeypatch):
    import training.train as train_module

    real = train_module.Trainer.save_checkpoint

    def never_best(self, *args, best=False, **kwargs):
        if not best:
            return real(self, *args, **kwargs)

    monkeypatch.setattr(train_module.Trainer, "save_checkpoint", never_best)

    with pytest.raises(NoBestCheckpointError, match="validation never selected"):
        train(tmp_path, cache_with_test, epochs=1)
    assert test_evaluations == []


def test_grid_entry_records_where_test_came_from(tmp_path, cache_with_test, test_evaluations):
    from training.ablations import run_ablation_suite

    results = run_ablation_suite(
        cache_dir=str(cache_with_test), ckpt_dir=str(tmp_path / "grid"), epochs=2, seeds=(42,),
        with_test=True, arc_config=ARC, configs=["A1_baseline"], train_samples=16,
        val_samples=8,
    )

    entry = results["A1_baseline"][0]
    val = entry["history"]["acc"]
    assert entry["test_from"]["epoch"] == val.index(max(val)) + 1
    assert "test_acc" in entry and len(test_evaluations) == 1
