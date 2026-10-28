"""
scripts/step7_runs.py -- the Step 7 session plan: cache hashes checked
against the canonical manifest before anything trains, the PD-7 / PD-10
protocol, finished runs never started again (test evaluated once), and
the status table that says where to continue.

    python -m pytest tests/test_step7_runs.py
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import step7_runs  # noqa: E402
from config import cache_filename  # noqa: E402
from tests.test_checkpoint_provenance import smoke_cache  # noqa: E402,F401  (fixture)
from training.checkpoint import run_key  # noqa: E402
from training.manifest import file_sha256  # noqa: E402
from training.results import result_path, save_result  # noqa: E402

F0 = "F0_full_with_llm_prior"


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    """A cache directory and the manifest recording its hashes, made canonical."""

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    caches = {}
    for split in step7_runs.CANONICAL_SHA256_PREFIX:
        path = cache_dir / cache_filename(split, step7_runs.ARC_CONFIG)
        path.write_bytes(f"{split} cache".encode())
        caches[split] = {"file_sha256": file_sha256(path)}

    manifest = tmp_path / "cache_manifest.json"
    manifest.write_text(json.dumps({"caches": caches}))
    monkeypatch.setattr(step7_runs, "CANONICAL_SHA256_PREFIX",
                        {s: c["file_sha256"][:16] for s, c in caches.items()})
    return str(cache_dir), str(manifest)


def test_canonical_cache_passes(canonical):
    hashes = step7_runs.verify_cache(*canonical)
    assert set(hashes) == {"train", "validation", "test"}


def test_changed_missing_or_non_canonical_caches_stop(canonical, monkeypatch):
    cache_dir, manifest = canonical
    test_file = os.path.join(cache_dir, cache_filename("test", step7_runs.ARC_CONFIG))

    with open(test_file, "ab") as f:
        f.write(b"x")
    with pytest.raises(step7_runs.Step7Error, match="test: .* hashes to"):
        step7_runs.verify_cache(cache_dir, manifest)

    os.remove(test_file)
    with pytest.raises(step7_runs.Step7Error, match="test: .* does not exist"):
        step7_runs.verify_cache(cache_dir, manifest)

    monkeypatch.setattr(step7_runs, "CANONICAL_SHA256_PREFIX",
                        dict(step7_runs.CANONICAL_SHA256_PREFIX, train="0" * 16))
    with pytest.raises(step7_runs.Step7Error, match="train: the manifest records"):
        step7_runs.verify_cache(cache_dir, manifest)


def test_the_real_canonical_prefixes():
    assert step7_runs.CANONICAL_SHA256_PREFIX == {
        "train": "4bc0031289a03e4f", "validation": "0760535c445ad6dc",
        "test": "5faa6d62c82612d6"}


def test_protocol_is_the_pre_registered_one():
    step7_runs.check_protocol()

    class Capstone:
        EPOCHS, PATIENCE, DETERMINISTIC = 8, 5, True
        DETERMINISTIC_WARN_ONLY, QUANTUM_SIMULATOR = False, "torch"

    with pytest.raises(step7_runs.Step7Error, match="EPOCHS"):
        step7_runs.check_protocol(Capstone)


def test_plan_runs_f0_first_and_covers_the_pd14_arms():
    from evaluation import stats

    assert step7_runs.SESSIONS[1] == stats.PRIMARY_ARM
    assert set(step7_runs.SESSIONS.values()) == \
        {stats.PRIMARY_ARM} | {arm for arm, _ in stats.SECONDARY_FAMILY}
    assert set(stats.LLM_BASELINES) <= set(step7_runs.BASELINE_MODES)
    assert step7_runs.SEEDS == (0, 1, 2)


def latest(ckpt_dir, arm, seed, epoch, stopped_early=False):
    path = os.path.join(ckpt_dir, f"{run_key(arm, step7_runs.ARC_CONFIG, seed)}_latest.pt")
    torch.save({"epoch": epoch, "best_acc": 0.5, "stopped_early": stopped_early}, path)


def finished(ckpt_dir, arm, seed):
    run = run_key(arm, step7_runs.ARC_CONFIG, seed)
    save_result(ckpt_dir, run, arm, {
        "seed": seed, "best_val_acc": 0.6, "test_acc": 0.55,
        "history": {"loss": [1.0] * 9}, "test_from": {"epoch": 3, "val_acc": 0.6}})


def baseline_file(out_dir, mode, acc=0.5):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{mode}_test.json"), "w") as f:
        json.dump({"mode": mode, "split": "test", "accuracy": acc, "records": []}, f)


def test_status_states_and_where_to_continue(tmp_path):
    ckpt, base = str(tmp_path / "ckpt"), str(tmp_path / "baselines")
    os.makedirs(ckpt)

    rows, baselines, nxt = step7_runs.status(ckpt, base)
    assert {r["state"] for r in rows} == {"not started"} and nxt == 1
    assert len(rows) == 12 and baselines == {m: None for m in step7_runs.BASELINE_MODES}

    finished(ckpt, F0, 0)
    latest(ckpt, F0, 1, epoch=3)
    latest(ckpt, F0, 2, epoch=8, stopped_early=True)
    rows, _, nxt = step7_runs.status(ckpt, base)
    by_seed = {r["seed"]: r for r in rows if r["arm"] == F0}

    assert by_seed[0]["state"] == "finished" and by_seed[0]["test_acc"] == 0.55
    assert by_seed[0]["epochs_done"] == 9 and by_seed[0]["test_epoch"] == 3
    assert (by_seed[1]["state"], by_seed[1]["epochs_done"]) == ("in progress", 4)
    assert by_seed[2]["state"] == "trained, test pending"
    assert nxt == 1
    table = step7_runs.format_status(rows, *step7_runs.status(ckpt, base)[1:])
    assert "NEXT: session 1." in table and "9 (b3)" in table


def test_baselines_come_after_all_runs_and_then_all_done(tmp_path):
    ckpt, base = str(tmp_path / "ckpt"), str(tmp_path / "baselines")
    os.makedirs(ckpt)
    for arm in step7_runs.SESSIONS.values():
        for seed in step7_runs.SEEDS:
            finished(ckpt, arm, seed)

    assert step7_runs.status(ckpt, base)[2] == step7_runs.BASELINE_SESSION
    for mode in step7_runs.BASELINE_MODES:
        baseline_file(base, mode)
    rows, baselines, nxt = step7_runs.status(ckpt, base)
    assert nxt is None and "ALL DONE" in step7_runs.format_status(rows, baselines, nxt)


def test_train_session_verifies_first_and_skips_finished_runs(tmp_path, canonical):
    cache_dir, manifest = canonical
    ckpt = str(tmp_path / "ckpt")
    os.makedirs(ckpt)
    finished(ckpt, F0, 1)

    calls = []
    step7_runs.train_session(1, cache_dir, ckpt, manifest, runner=calls.append)
    assert [c[c.index("--seeds") + 1] for c in calls] == ["0", "2"]
    assert all(c[c.index("--configs") + 1] == F0 for c in calls)

    os.remove(os.path.join(cache_dir, cache_filename("train", step7_runs.ARC_CONFIG)))
    calls.clear()
    with pytest.raises(step7_runs.Step7Error):
        step7_runs.train_session(1, cache_dir, ckpt, manifest, runner=calls.append)
    assert calls == []

    with pytest.raises(step7_runs.Step7Error, match="baselines"):
        step7_runs.train_session(5, cache_dir, ckpt, manifest, runner=calls.append)


def test_train_command_is_the_protocol_run(monkeypatch):
    import main as cli

    seen = {}
    monkeypatch.setattr(cli, "run_ablation_suite", lambda **kw: seen.update(kw))
    cmd = step7_runs.train_command(F0, 2, "/drive/cache", "/drive/ckpt")
    monkeypatch.setattr(sys, "argv", cmd[1:])
    cli.main()

    assert seen["configs"] == [F0] and seen["seeds"] == (2,)
    assert (seen["cache_dir"], seen["ckpt_dir"]) == ("/drive/cache", "/drive/ckpt")
    assert (seen["arc_config"], seen["epochs"], seen["with_test"]) == ("ARC-Easy", 20, True)
    assert seen["deterministic_warn_only"] is False and seen["ignore_pin_mismatch"] is False
    assert seen["train_samples"] is None and seen["val_samples"] is None


def test_baseline_commands_score_the_test_split(tmp_path, monkeypatch, canonical):
    from evaluation import baselines

    cache_dir, manifest = canonical
    out = str(tmp_path / "baselines")
    seen = []

    def fake(name):
        def run(cache, split, max_samples, arc_config, limit=None):
            seen.append((name, cache, split, max_samples, arc_config))
            return 0.5, [1, 0]
        return run

    monkeypatch.setattr(baselines, "run_cached_llm_baseline", fake("llm"))
    monkeypatch.setattr(baselines, "run_vote", fake("vote"))
    monkeypatch.setattr(baselines, "run_letter_pick", fake("letter"))

    def runner(cmd):
        monkeypatch.setattr(sys, "argv", ["baselines", *cmd[3:]])
        baselines.main()

    baseline_file(out, "llm")
    step7_runs.run_baselines(cache_dir, out, manifest, runner=runner)

    assert [s[0] for s in seen] == ["letter", "vote"]
    assert all(s[1:] == (cache_dir, "test", None, "ARC-Easy") for s in seen)
    assert set(step7_runs.baseline_status(out).values()) == {0.5}


def test_status_reads_a_real_result_file(tmp_path, smoke_cache):
    from training.ablations import run_ablation_suite

    ckpt = tmp_path / "ckpt"
    run_ablation_suite(cache_dir=str(smoke_cache), ckpt_dir=str(ckpt), epochs=1, seeds=(0,),
                       with_test=False, arc_config=step7_runs.ARC_CONFIG, configs=[F0],
                       train_samples=16, val_samples=8)

    assert os.path.exists(result_path(str(ckpt), run_key(F0, step7_runs.ARC_CONFIG, 0)))
    row = step7_runs.run_status(str(ckpt), F0, 0)
    assert (row["state"], row["epochs_done"]) == ("finished", 1)
    assert 0.0 <= row["best_val_acc"] <= 1.0 and row["test_acc"] is None
