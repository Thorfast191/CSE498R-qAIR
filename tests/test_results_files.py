"""
E5 -- one result file per (arm, seed); ablation_results.pt is rebuilt from
all of them and refused when they come from different code or caches,
unless --allow-mixed (recorded and printed). docs/baseline/issues.md E5.

    python -m pytest tests/test_results_files.py

Real model on CPU with the fake cache from test_checkpoint_provenance;
needs pennylane.
"""

import os
import sys

import pytest
import torch

from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)
from training.checkpoint import load_checkpoint_file, run_key
from training.results import (
    COMBINED_NAME,
    MixedResultsError,
    combine_results,
    load_combined,
    result_path,
)


def grid(ckpt_dir, cache_dir, configs, seeds, **kwargs):
    from training.ablations import run_ablation_suite

    return run_ablation_suite(
        cache_dir=str(cache_dir), ckpt_dir=str(ckpt_dir), epochs=1, seeds=seeds,
        with_test=False, arc_config=ARC, configs=configs, train_samples=16,
        val_samples=8, **kwargs,
    )


def combined(ckpt_dir):
    return load_checkpoint_file(os.path.join(str(ckpt_dir), COMBINED_NAME))


def tamper(ckpt_dir, arm, seed, change):
    """Make one result file look as if it came from other code or caches."""

    path = result_path(str(ckpt_dir), run_key(arm, ARC, seed))
    saved = load_checkpoint_file(path)
    change(saved["entry"]["history"]["manifest"])
    torch.save(saved, path)


def other_code(manifest):
    manifest["code_sha256"] = "0" * 64


def other_cache(manifest):
    manifest["caches"]["train"]["file_sha256"] = "f" * 64


def test_two_invocations_both_appear(tmp_path, smoke_cache):
    grid(tmp_path, smoke_cache, ["A1_baseline"], [42])
    grid(tmp_path, smoke_cache, ["A2_validator"], [43])

    results = combined(tmp_path)

    assert results["format"] == 2 and results["allow_mixed"] is False
    assert len(results["sources"]) == 1
    assert {arm: [e["seed"] for e in runs] for arm, runs in results["results"].items()} \
        == {"A1_baseline": [42], "A2_validator": [43]}
    for arm, seed in (("A1_baseline", 42), ("A2_validator", 43)):
        assert os.path.exists(result_path(str(tmp_path), run_key(arm, ARC, seed)))


def test_rerun_replaces_only_its_own_entry(tmp_path, smoke_cache):
    grid(tmp_path, smoke_cache, ["A1_baseline"], [42, 43])
    grid(tmp_path, smoke_cache, ["A1_baseline"], [42])

    assert [e["seed"] for e in combined(tmp_path)["results"]["A1_baseline"]] == [42, 43]


@pytest.mark.parametrize("change", [other_code, other_cache])
def test_mixed_sources_are_refused(tmp_path, smoke_cache, change):
    grid(tmp_path, smoke_cache, ["A1_baseline"], [42])
    tamper(tmp_path, "A1_baseline", 42, change)
    before = combined(tmp_path)

    with pytest.raises(MixedResultsError, match="2 different"):
        grid(tmp_path, smoke_cache, ["A1_baseline"], [43])

    # The new run's own result file is kept; the combined file is not rewritten.
    assert os.path.exists(result_path(str(tmp_path), run_key("A1_baseline", ARC, 43)))
    assert [e["seed"] for e in combined(tmp_path)["results"]["A1_baseline"]] \
        == [e["seed"] for e in before["results"]["A1_baseline"]]


def test_allow_mixed_is_recorded_and_printed(tmp_path, smoke_cache, capsys):
    grid(tmp_path, smoke_cache, ["A1_baseline"], [42])
    tamper(tmp_path, "A1_baseline", 42, other_code)

    grid(tmp_path, smoke_cache, ["A1_baseline"], [43], allow_mixed=True)

    results = combined(tmp_path)
    assert results["allow_mixed"] is True and len(results["sources"]) == 2
    assert "[RESULTS MIXED -- --allow-mixed]" in capsys.readouterr().out


def test_stats_reads_both_formats(tmp_path, smoke_cache, monkeypatch, capsys):
    from evaluation import stats

    grid(tmp_path, smoke_cache, ["A1_baseline"], [42])
    v46 = tmp_path / "v46_results.pt"
    torch.save({"A1_baseline": [{"seed": 42, "best_val_acc": 0.5}]}, v46)

    assert load_combined(str(v46))["format"] == 1

    for path, note in ((tmp_path / COMBINED_NAME, "one code version"),
                       (v46, "v46-format file")):
        monkeypatch.setattr(sys, "argv", ["stats", "--results", str(path)])
        stats.main()
        assert note in capsys.readouterr().out


def test_cli_passes_allow_mixed(monkeypatch):
    import main as cli

    seen = {}
    monkeypatch.setattr(cli, "run_ablation_suite", lambda **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "ablation", "--allow-mixed"])
    cli.main()

    assert seen["allow_mixed"] is True


def test_combine_refuses_without_writing(tmp_path, smoke_cache):
    grid(tmp_path, smoke_cache, ["A1_baseline"], [42, 43])
    tamper(tmp_path, "A1_baseline", 43, other_cache)

    with pytest.raises(MixedResultsError):
        combine_results(str(tmp_path))
    assert combine_results(str(tmp_path), allow_mixed=True)["allow_mixed"] is True
