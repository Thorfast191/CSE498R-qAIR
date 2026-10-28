"""
E11 -- the environment is checked at startup and recorded in every
checkpoint (docs/baseline/issues.md).

    python -m pytest tests/test_environment.py
"""

import os

import pytest

import training.environment as environment_module
from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)
from training.checkpoint import load_checkpoint_file, run_key
from training.environment import (
    EnvironmentMismatchError,
    check_pins,
    describe,
    environment_record,
    installed_version,
    parse_pins,
)

RUN = run_key("qair", ARC, 42)


def requirements(tmp_path, text):
    path = tmp_path / "requirements.txt"
    path.write_text(text)
    return str(path)


def test_parse_pins_reads_only_exact_pins(tmp_path):
    path = requirements(tmp_path, (
        "torch\n"
        "\n"
        "transformers==4.52.4\n"
        "huggingface_hub==0.33.4  # comment\n"
        "#     pip install pennylane-lightning[gpu]==0.42.0\n"
        "pennylane-lightning[gpu]==0.42.0\n"
        "umap-learn==0.5.9.post2\n"
    ))

    assert parse_pins(path) == {
        "transformers": "4.52.4",
        "huggingface-hub": "0.33.4",
        "pennylane-lightning": "0.42.0",
        "umap-learn": "0.5.9.post2",
    }


def test_repository_pins_torch_to_the_colab_build():
    assert parse_pins()["torch"] == "2.11.0+cu130"


@pytest.mark.parametrize("cuda, installed, ok", [
    (None, "2.11.0+cu130", True),     # exact
    (None, "2.11.0", True),           # macOS CPU wheel
    (None, "2.11.0+cpu", True),       # Linux CPU wheel
    (None, "2.12.0", False),          # CPU build, other version
    ("13.0", "2.11.0+cu130", True),
    ("12.6", "2.11.0+cu126", False),  # CUDA build, other CUDA label
    ("13.0", "2.11.0", False),
    (None, None, False),              # not installed
])
def test_torch_pin_cpu_build_rule(monkeypatch, cuda, installed, ok):
    from training import environment

    monkeypatch.setattr(environment.torch.version, "cuda", cuda)
    assert environment.matches_pin("torch", "2.11.0+cu130", installed) is ok


def test_cpu_rule_is_torch_only():
    from training.environment import matches_pin

    assert not matches_pin("transformers", "4.52.4+x", "4.52.4")


def test_matching_pins_pass(tmp_path):
    version = installed_version("pennylane")
    if version is None:
        pytest.skip("pennylane not installed")

    result = check_pins(path=requirements(tmp_path, f"pennylane=={version}\n"))

    assert result == {"mismatches": [], "overridden": False}


def test_mismatch_raises_and_override_is_recorded(tmp_path):
    path = requirements(tmp_path, "pennylane==0.0.1\nplotly==0.0.1\n")

    with pytest.raises(EnvironmentMismatchError, match="pennylane: pinned 0.0.1"):
        check_pins(path=path)

    result = check_pins(override=True, path=path)

    # plotly is pinned but not a runtime package: recorded, never checked.
    assert [m["package"] for m in result["mismatches"]] == ["pennylane"]
    assert result["overridden"] is True


def test_record_splits_compared_and_recorded_versions():
    compared, recorded = environment_record()

    assert set(compared) == {"torch", "cuda", "pennylane", "pennylane-lightning"}
    assert {"python", "transformers", "numpy", "packages_sha256"} <= set(recorded)
    assert len(recorded["packages_sha256"]) == 64


def test_training_refuses_mismatch_before_writing_anything(tmp_path, smoke_cache,
                                                           monkeypatch):
    from training.runner import run_training

    monkeypatch.setattr(environment_module, "REQUIREMENTS_PATH",
                        requirements(tmp_path, "transformers==0.0.1\n"))
    ckpt_dir = tmp_path / "ckpt"

    with pytest.raises(EnvironmentMismatchError, match="transformers"):
        run_training(cache_dir=str(smoke_cache), ckpt_dir=str(ckpt_dir),
                     train_samples=16, val_samples=8, epochs=1, seed=42,
                     with_test=False, arc_config=ARC)

    assert not ckpt_dir.exists()


def test_override_is_in_provenance_and_every_result(tmp_path, smoke_cache,
                                                    monkeypatch, capsys):
    from training.ablations import run_ablation_suite
    from training.runner import run_training

    monkeypatch.setattr(environment_module, "REQUIREMENTS_PATH",
                        requirements(tmp_path, "transformers==0.0.1\n"))
    ckpt_dir = tmp_path / "ckpt"

    history = run_training(
        cache_dir=str(smoke_cache), ckpt_dir=str(ckpt_dir), train_samples=16,
        val_samples=8, epochs=1, seed=42, with_test=False, arc_config=ARC,
        ignore_pin_mismatch=True,
    )

    ckpt = load_checkpoint_file(os.path.join(ckpt_dir, f"{RUN}_latest.pt"))
    assert ckpt["provenance"]["pin_check"]["overridden"] is True
    assert history["provenance"]["pin_check"] == ckpt["provenance"]["pin_check"]
    assert describe(ckpt["provenance"]) in capsys.readouterr().out
    assert "pins OVERRIDDEN" in describe(ckpt["provenance"])

    with open(os.path.join(ckpt_dir, f"{RUN}_env.txt")) as f:
        assert any(line.startswith("torch==") for line in f)

    results = run_ablation_suite(
        cache_dir=str(smoke_cache), ckpt_dir=str(ckpt_dir), epochs=1,
        seeds=[42], with_test=False, arc_config=ARC, configs=["A1_baseline"],
        train_samples=16, val_samples=8, ignore_pin_mismatch=True,
    )
    entry = results["A1_baseline"][0]
    assert "pins OVERRIDDEN" in entry["environment"]
    summary = capsys.readouterr().out.split("Ablation Suite Complete")[1]
    assert entry["environment"] in summary
