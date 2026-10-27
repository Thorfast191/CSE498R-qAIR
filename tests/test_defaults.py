"""
E9 -- defaults match the CSE498R baseline protocol, the CLI can reproduce
the capstone configuration, and the stale names stay gone
(docs/baseline/issues.md E9; PD-7).

    python -m pytest tests/test_defaults.py
"""

import inspect
import os
import subprocess
import sys

import pytest

import config
from tests.test_checkpoint_provenance import REPO_ROOT

# Fixed in E9; the v45 history comments and historical measurements stay.
STALE = ("qAIR-v45.1", "qair_v45_best", "869 examples", "299-example")


def test_config_defaults_are_the_baseline_protocol():
    assert config.ARC_CONFIG == "ARC-Easy"
    assert (config.EPOCHS, config.PATIENCE) == (20, 5)


def test_entry_points_default_to_the_protocol():
    from training.ablations import run_ablation_suite
    from training.runner import run_training

    for fn in (run_training, run_ablation_suite):
        params = inspect.signature(fn).parameters
        assert params["arc_config"].default == "ARC-Easy", fn.__name__
        assert params["epochs"].default == 20, fn.__name__
        assert params["patience"].default == 5, fn.__name__


@pytest.fixture
def cli(monkeypatch):
    import main as cli_module

    seen = {}
    monkeypatch.setattr(cli_module, "run_training", lambda **kw: seen.update(kw))
    monkeypatch.setattr(cli_module, "run_ablation_suite", lambda **kw: seen.update(kw))

    def run(*argv):
        seen.clear()
        monkeypatch.setattr(sys, "argv", ["main.py", *argv])
        cli_module.main()
        return dict(seen)

    return run


@pytest.mark.parametrize("mode", ["train", "ablation"])
def test_cli_defaults_and_capstone_flags(cli, mode):
    assert {k: cli("--mode", mode)[k] for k in ("arc_config", "epochs")} == \
        {"arc_config": "ARC-Easy", "epochs": 20}

    capstone = cli("--mode", mode, "--arc-config", "ARC-Challenge", "--epochs", "8")
    assert (capstone["arc_config"], capstone["epochs"]) == ("ARC-Challenge", 8)


def test_stale_names_are_gone_from_tracked_code():
    files = subprocess.run(
        ["git", "-C", REPO_ROOT, "ls-files", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout.split()

    found = []
    for path in files:
        if path.startswith("tests/"):
            continue
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
            text = f.read()
        found += [(path, s) for s in STALE if s in text]

    assert found == []
