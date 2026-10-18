"""
A9 -- a cache that fails to load is never renamed, deleted or rebuilt
(docs/baseline/issues.md).

Every failure case checks that the file on disk is byte-identical after
the failed load and that no ".corrupt" copy appears.

    python -m pytest tests/test_cache_load_errors.py
"""

import hashlib
import os

import pytest
import torch
from torch.torch_version import TorchVersion

import training.dataset as dataset_module
from config import cache_filename
from tests.test_cache_determinism import ARC, FakeEncoder, FakeGenerator, fake_rows
from training.dataset import CacheLoadError, QAIRDataset, header_path


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cache_file(cache_dir):
    return os.path.join(cache_dir, cache_filename("train", ARC))


def read(cache_dir, **kwargs):
    return QAIRDataset(split="train", cache_dir=str(cache_dir), arc_config=ARC, **kwargs)


def assert_untouched(path, digest):
    assert sha256(path) == digest
    assert not os.path.exists(path + ".corrupt")


@pytest.fixture
def built(monkeypatch, tmp_path):
    """A real (fake-generator) 8-sample cache with its JSON sidecar."""

    FakeGenerator.calls = 0
    FakeGenerator.raise_on_call = None
    FakeGenerator.prompt_tag = "v1"
    monkeypatch.setattr(dataset_module, "HypothesisGenerator", FakeGenerator)
    monkeypatch.setattr(dataset_module, "HypothesisEncoder", FakeEncoder)
    monkeypatch.setattr(dataset_module, "load_arc", lambda cfg: {"train": fake_rows()})
    read(tmp_path, max_samples=8, allow_build=True)
    return tmp_path


def test_truncated_cache_raises_with_sidecar_and_stays_identical(built):
    path = cache_file(built)
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data[: len(data) // 2])
    digest = sha256(path)

    with pytest.raises(CacheLoadError) as err:
        read(built, max_samples=8)

    msg = str(err.value)
    assert path in msg
    assert "sidecar header" in msg
    assert '"version": 46' in msg and '"gen_seed": 0' in msg and '"device": "cpu"' in msg
    assert "NOT been moved" in msg
    assert_untouched(path, digest)


def test_garbage_file_without_sidecar_raises_and_stays_identical(tmp_path):
    path = cache_file(tmp_path)
    with open(path, "wb") as f:
        f.write(b"this is not a torch archive")
    digest = sha256(path)

    with pytest.raises(CacheLoadError, match="no header sidecar"):
        read(tmp_path, max_samples=8)

    assert_untouched(path, digest)


def test_header_with_torchversion_loads(built):
    path = cache_file(built)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["provenance"]["segments"][0]["torch_version"] = torch.__version__
    assert isinstance(payload["provenance"]["segments"][0]["torch_version"], TorchVersion)
    torch.save(payload, path)

    ds = read(built, max_samples=8)

    assert len(ds.samples) == 8
    assert ds.provenance["segments"][0]["torch_version"] == str(torch.__version__)


class _WritesMarker:
    """Unpickling this would create a file: proof that code ran."""

    marker = None

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_unsafe_pickle_is_refused_without_executing(built):
    path = cache_file(built)
    marker = os.path.join(built, "EXECUTED")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    evil = _WritesMarker()
    evil.marker = marker
    payload["provenance"]["evil"] = evil
    torch.save(payload, path)
    assert not os.path.exists(marker)
    digest = sha256(path)

    with pytest.raises(CacheLoadError) as err:
        read(built, max_samples=8)

    assert "weights_only" in str(err.value) or "Unsupported global" in str(err.value)
    assert not os.path.exists(marker), "loading the cache executed pickled code"
    assert_untouched(path, digest)


def test_sidecar_written_next_to_cache(built):
    side = header_path(cache_file(built))
    assert os.path.exists(side)
    import json
    with open(side) as f:
        header = json.load(f)
    assert header["n_samples"] == 8 and header["complete"] is False
    assert header["hypotheses_sha256"] == read(built, max_samples=8).hypotheses_sha256()
