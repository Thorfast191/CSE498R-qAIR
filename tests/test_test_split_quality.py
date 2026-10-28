"""
The [CACHE QUALITY] block is printed for train and validation and never
for the test split, so no gate or decision can see test numbers
(CLAUDE.md rule 5; Step 6 decision, docs/HANDOFF.md).

    python -m pytest tests/test_test_split_quality.py
"""

import training.dataset as dataset_module
from tests.test_cache_determinism import ARC, FakeEncoder, FakeGenerator, fake_rows
from training.dataset import QAIRDataset


def test_block_is_printed_for_train_and_validation_only(tmp_path, monkeypatch, capsys):
    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    rows = fake_rows()
    monkeypatch.setattr(dataset_module, "load_arc",
                        lambda cfg: {s: rows for s in ("train", "validation", "test")})

    for split in ("train", "validation", "test"):
        QAIRDataset(split=split, cache_dir=str(tmp_path), arc_config=ARC, allow_build=True)
        built = capsys.readouterr().out
        QAIRDataset(split=split, cache_dir=str(tmp_path), arc_config=ARC)
        loaded = capsys.readouterr().out

        for out in (built, loaded):
            if split == "test":
                assert "not printed: the test split is never gated" in out
                assert "support agreement" not in out and "fallback hypotheses" not in out
            else:
                assert f"[CACHE QUALITY {split} / {ARC}]" in out
                assert "support agreement" in out
