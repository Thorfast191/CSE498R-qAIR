# qAIR CSE498R — research log

One entry per fix or run: date, item, commit, environment, what was run and
what came out. Commit hashes are filled in by the follow-up entry when the
fix commit cannot name itself.

---

## 2026-10-18 — E3: seeded, reproducible hypothesis generation

- **Commit:** `b5c0ae8` (branch `baseline`, on top of `d0a074a`).
- **Change:** per-batch seeded sampling (`batch_seed(gen_seed, raw_idx)`,
  sha256), autosave only at batch boundaries, `CACHE_VERSION` 46 with a
  provenance header, resume guard, `config.GEN_SEED = 0`,
  `scripts/build_cache.py --gen-seed`. Files: `models/generator.py`,
  `training/dataset.py`, `config.py`, `scripts/build_cache.py`,
  `tests/test_cache_determinism.py` (new).
- **Environment:** scratch venv outside the repo; Python 3.12.12, torch
  2.14.1 (CPU), transformers 4.52.4, datasets 3.6.0, huggingface_hub 0.33.4,
  pytest 9.1.1; Apple M4. The local HF token had expired; the real-model test
  ran with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (public model, cached weights).
- **Commands and results:**
  - `python -m pytest tests/test_cache_determinism.py -q` → 8 passed,
    1 skipped (slow) in 2.75 s.
  - Mutation check on scratch copies: old mid-batch autosave → 1 failed
    (resume test); unseeded batches → 2 failed (same-seed, resume).
  - `QAIR_SLOW=1 python -m pytest tests/test_cache_determinism.py::test_real_qwen_same_seed_same_hash -s`
    → 1 passed in 70 s; both 5-question ARC-Easy CPU builds:
    `hypotheses_sha256 = 5b4f840469306962c4f5d6379befd5827b7c91f70fb35dbe988f9d41cd23f51d`;
    Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`;
    `prompt_sha256 = 787016449994f6391a2356b0e13e490851ddd1f90bf6f186b59c5506b661191f`.
- **Not done:** no ARC-Easy cache built (smoke caches lived in pytest temp
  dirs); no training.
- **New issue logged:** A9 (any cache-load error moves the cache aside and
  rebuilds it), found when a `TorchVersion` header value failed
  `torch.load(weights_only=True)` in the first test run.

---

## 2026-10-18 — E7: reading a cache never builds one

- **Commit:** `6aa20a5`.
- **Change:** `QAIRDataset(..., allow_build=False)`; a read that the stored
  cache cannot satisfy raises with path, state, request and the build command,
  before any lock or model is touched. `scripts/build_cache.py` passes
  `allow_build=True`. Files: `training/dataset.py`, `scripts/build_cache.py`,
  `tests/test_cache_determinism.py` (helper builds with `allow_build=True`),
  `tests/test_cache_build_guard.py` (new).
- **Environment:** same scratch venv as E3.
- **Results:** `python -m pytest tests -q` → 13 passed, 1 skipped in 3.83 s.
  Mutation (`allow_build` default True) → 3 failed (the three refusal tests),
  2 passed.
- **Not done:** no ARC-Easy cache, no training.

---

## 2026-10-18 — A9: a cache that fails to load is never touched

- **Commit:** `0313a15`.
- **Change:** `safe_load_cache` (`weights_only=True`, `TorchVersion`
  allow-listed via `torch.serialization.safe_globals`); load failures raise
  `CacheLoadError` with path, exception and the JSON sidecar header; `_save`
  writes `<cache>.header.json`. Files: `training/dataset.py`,
  `tests/test_cache_load_errors.py` (new).
- **Results:** `python -m pytest tests -q` → 18 passed, 1 skipped in 3.2 s.
  Mutations: rename handler restored → 3 failed; `weights_only=False` → the
  unsafe-pickle test failed (its marker file was created).
- **Scope note:** the checkpoint `torch.load` calls (checkpoint.py, runner.py,
  ablations.py, input_ablation.py, sample_inference.py) were added to A3's
  scope in issues.md.
- **Not done:** no ARC-Easy cache, no training.

---

## 2026-10-18 — A6: cache edge cases

- **Commit:** `0a4f464`.
- **Change:** complete caches are final; a missing split raises; batch
  scoring failures retried per question, with per-sample `score_failed`
  reported by `report_quality`; cache version kept at 46 with a loader
  default. Files: `training/dataset.py`, `tests/test_cache_edge_cases.py` (new).
- **Results:** `python -m pytest tests -q` → 23 passed, 1 skipped in 4.2 s.
  Mutations: old early return → 2 failed; split substitution → 1 failed;
  batch zeros → 1 failed.
- **Not done:** no ARC-Easy cache, no training.


---

## 2026-10-19 — A3 + A8: provenance-checked checkpoints

- **Commit:** `f605c55`.
- **Change:** run key `f"{arm}_{arc_config}_s{seed}"`; checkpoint format 2
  with `provenance` (config, seed, data hashes, code fingerprint; commit,
  torch version, device informational); resume and best-checkpoint loads
  raise `CheckpointMismatchError` on any mismatch, on a format-1 file and on
  a stale best; every checkpoint/history/results load uses
  `load_checkpoint_file` (`weights_only=True`). Files: `training/checkpoint.py`,
  `training/train.py`, `training/runner.py`, `training/ablations.py`,
  `evaluation/input_ablation.py`, `evaluation/sample_inference.py`,
  `evaluation/stats.py`, `visualization/generate_report.py`,
  `tests/test_checkpoint_provenance.py` (new). `notebooks/qAIR_v46.ipynb`
  untouched; its Trainer cells no longer run (approved).
- **Environment:** `qenv` scratch venv outside the repo; Python 3.12.12,
  torch 2.14.1 (CPU), pennylane 0.42.3 and pennylane-lightning 0.42.0
  installed for this item (requirements.txt pins), pytest 9.1.1; Apple M4.
- **Results:** `python -m pytest tests/test_checkpoint_provenance.py -q` →
  28 passed in 6.99 s; `python -m pytest tests -q` → 51 passed, 1 skipped in
  7.71 s. Smoke runs (fake 29-question cache, 16 train / 8 val, 1 epoch, CPU,
  `deterministic=False`): seed 42 epoch-1 train loss 1.8829, seed 43 2.6331,
  two checkpoint files. Mutations (seed-free key → 3 failed; no-op check → 14;
  `weights_only=False` → 1; commit compared → 2; no stale-best check → 1;
  `requirements.txt` not fingerprinted → 2).
- **Not done:** no ARC-Easy cache, no real training.

---

## 2026-10-19 — A4 + A10: resume keeps history and early-stopping state

- **Commit:** `4083ecf`.
- **Change:** checkpoints store `history`, `epochs_no_improve`,
  `stopped_early`; `load_or_resume` returns the full state from the
  checkpoint (no `_history.pt` read, no catch-all warning); `Trainer.train`
  continues history and count; `Trainer.train_from` skips finished runs
  (all epochs, or stopped early) in `runner.py` and the grid;
  `run_training(patience=…)`. Files: `training/train.py`,
  `training/checkpoint.py`, `training/runner.py`, `training/ablations.py`,
  `tests/test_resume_state.py` (new), `tests/test_checkpoint_provenance.py`
  (new signatures), CLAUDE.md (commit-before-training note).
- **Environment:** `qenv` scratch venv as for A3.
- **Results:** `python -m pytest tests/test_resume_state.py -q` → 5 passed
  in 18.24 s; `python -m pytest tests -q` → 56 passed, 1 skipped in 23.74 s.
  Mutations: history dropped → 2 failed; count reset → 1; `stopped_early`
  ignored → 2; never saved → 3; missing state accepted → 1.
- **New issue logged:** A11. Probe (scratch test outside the repo; fake cache,
  16/8, seed 42, CPU, `deterministic=False`, scripted validation): 4-epoch
  losses uninterrupted `[1.8829, 1.3686, 1.0157, 0.8861]`, interrupted after
  epoch 2 and resumed `[1.8829, 1.3686, 0.9016, 0.8123]`.
- **Note:** **Always commit before starting a training run; the code fingerprint ignores untracked .py files.**
- **Not done:** no ARC-Easy cache, no real training.

---

## 2026-10-19 — Untracked three files committed by mistake

- **What happened:** `git add docs` in the A4 docs commit `2dedcd8` (pushed)
  also added three untracked local files:
  `docs/baseline/qAIR_baseline_issues.docx` (28,458 bytes),
  `docs/prompts/PROMPTS_steps_3_to_11.md` (12,199 bytes),
  `docs/prompts/PROMPT_issue_dossier.md` (10,991 bytes).
- **Check:** pattern scan for tokens, API keys, passwords and private keys
  (`.md` files and the docx's XML): no secrets; the only hits were "tokens"
  in the NLP sense. `notebooks/qAIR_v46.ipynb` was not among them.
- **Action (approved option a, no force-push):** `7abc115` untracks them
  (still on disk) and adds them to `.gitignore`. They remain in the history
  of `2dedcd8`. Docs commits now stage files by name only.

---

## 2026-10-19 — A5: input_ablation rebuilds the trained architecture or refuses

- **Commit:** `b00595a` (docs first went in as `9c5f550` naming `7abc115` by mistake; corrected in the next commit).
- **Change:** config from checkpoint provenance (explicit `--config` must
  agree); format-1 checkpoints need `--config` or `_config.pt`; no default
  config and no per-key defaults; validation data must match the run's size
  and hash; grid provenance gains `model_dim`. Files:
  `evaluation/input_ablation.py`, `training/ablations.py`,
  `tests/test_input_ablation_config.py` (new).
- **Environment:** `qenv` scratch venv as for A3.
- **Results:** `python -m pytest tests/test_input_ablation_config.py -q` →
  10 passed in 11.12 s; `python -m pytest tests -q` → 66 passed, 1 skipped in
  31.64 s. Intact vs logged validation accuracy (fake cache, 8 val, 2 epochs,
  seed 42, CPU): qair 0.875/0.875, A1_baseline 0.75/0.75,
  A3c_classical_control_persistent 0.625/0.625. Mutations: default fallback
  → 1 failed; no contradiction check → 1; no data check → 1; altered stored
  config → 3.
- **Not done:** no ARC-Easy cache, no real training.
