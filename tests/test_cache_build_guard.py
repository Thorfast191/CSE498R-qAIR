"""
E7 -- reading a cache never builds one (docs/baseline/issues.md).

Caches are built with the fakes from test_cache_determinism.py and
allow_build=True; they are then read the way every consumer reads them
(allow_build left at its default). During the reads the generator and
encoder constructors are replaced by ones that fail, so any attempt to
start generation is a test failure, not hours of LLM time.

    python -m pytest tests/test_cache_build_guard.py
"""

import os

import pytest

import training.dataset as dataset_module
from config import cache_filename
from tests.test_cache_determinism import ARC, FakeEncoder, FakeGenerator, fake_rows
from training.dataset import QAIRDataset


def _must_not_be_called(*args, **kwargs):
    raise AssertionError("a read-only load tried to construct the LLM or encoder")


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """Build a capped (8 of 29) and a complete (29) cache, then forbid models."""

    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    monkeypatch.setattr(dataset_module, "load_arc", lambda cfg: {"train": fake_rows()})

    QAIRDataset(split="train", cache_dir=str(tmp_path / "capped"), arc_config=ARC,
                max_samples=8, allow_build=True)
    QAIRDataset(split="train", cache_dir=str(tmp_path / "complete"), arc_config=ARC,
                allow_build=True)

    monkeypatch.setattr(dataset_module, "HypothesisGenerator", _must_not_be_called)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", _must_not_be_called)
    monkeypatch.setattr(dataset_module, "load_arc", _must_not_be_called)
    return tmp_path


def read(cache_dir, **kwargs):
    return QAIRDataset(split="train", cache_dir=str(cache_dir), arc_config=ARC, **kwargs)


def test_capped_cache_read_uncapped_raises(stores):
    with pytest.raises(RuntimeError) as err:
        read(stores / "capped", max_samples=None)
    msg = str(err.value)
    assert "8 samples stored, complete=False" in msg
    assert "gen_seed=0" in msg
    assert "python scripts/build_cache.py --splits train" in msg


def test_cap_larger_than_stored_raises(stores):
    with pytest.raises(RuntimeError, match="requested   : max_samples=20"):
        read(stores / "capped", max_samples=20)


def test_missing_cache_raises_with_build_command(stores):
    with pytest.raises(RuntimeError) as err:
        read(stores / "nothing_here", max_samples=5)
    assert "state       : missing" in str(err.value)
    assert "--limit 5" in str(err.value)
    # The refused read must not leave a lock or a cache file behind.
    assert not os.path.exists(os.path.join(stores / "nothing_here",
                                           cache_filename("train", ARC)))
    assert not os.path.exists(os.path.join(stores / "nothing_here",
                                           cache_filename("train", ARC) + ".lock"))


def test_reading_within_the_cap_works(stores):
    assert len(read(stores / "capped", max_samples=8).samples) == 8
    assert len(read(stores / "capped", max_samples=5).samples) == 5


def test_complete_cache_reads_uncapped(stores):
    assert len(read(stores / "complete", max_samples=None).samples) == 29
