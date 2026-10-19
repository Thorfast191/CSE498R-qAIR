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

---

## 2026-10-19 — E10 + A11: deterministic training; resume restores every RNG stream

- **Commit:** `bbea237`.
- **Change:** `config.DETERMINISTIC = True` / `DETERMINISTIC_WARN_ONLY =
  False`; `set_seed(…, deterministic, warn_only)` sets
  `CUBLAS_WORKSPACE_CONFIG` if unset and the torch determinism flags both
  ways; `--deterministic-warn-only` in `main.py`; both flags in the
  provenance training config. Checkpoints store `rng` (python, numpy, torch,
  CUDA, train-loader generator), restored on resume and required. The grid
  re-seeds the shared loader generator per arm. Files: `config.py`,
  `main.py`, `training/seed.py`, `training/checkpoint.py`,
  `training/train.py`, `training/runner.py`, `training/ablations.py`,
  `tests/test_determinism_resume.py` (new).
- **Environment:** `qenv` scratch venv as for A3 (CPU only).
- **Results:** `python -m pytest tests/test_determinism_resume.py -q` →
  8 passed in 21.04 s; `python -m pytest tests -q` → 74 passed, 1 skipped in
  49.63 s. The A11 probe (4 epochs, interrupted at the third validation)
  now matches the uninterrupted run in every history value and in the
  final weights. Seven mutations each fail ≥ 1 test.
- **Pending (Step 6 notebook, T4):** two same-seed 1-epoch runs + per-example
  prediction diff; print any op that raises under strict mode; check
  `lightning.gpu` for its own nondeterminism.
- **Not done:** no ARC-Easy cache, no real training.

---

## 2026-10-19 — E11: environment checked at startup and recorded

- **Commit:** `4c4a766`.
- **Change:** `training/environment.py` (pin check for runtime packages,
  environment record, package list, `describe`); provenance `environment`
  (compared) / `environment_info` and `pin_check` (recorded);
  `--ignore-pin-mismatch`; `{run}_env.txt`; environment line printed and
  stored with every result; `requirements.txt` comment (torch pin deferred
  to Step 6). Files: `training/environment.py` (new),
  `training/checkpoint.py`, `training/runner.py`, `training/ablations.py`,
  `main.py`, `requirements.txt`, `tests/test_environment.py` (new),
  `tests/test_checkpoint_provenance.py`.
- **Environment:** `qenv` scratch venv (torch 2.14.1 CPU, cuda None,
  pennylane 0.42.3, pennylane-lightning 0.42.0; transformers 4.52.4,
  accelerate 1.7.0, datasets 3.6.0, huggingface_hub 0.33.4 match their
  pins).
- **Results:** `python -m pytest tests/test_environment.py -q` → 7 passed;
  `python -m pytest tests -q` → 85 passed, 1 skipped in 52.03 s. Five
  mutations each fail ≥ 1 test (the grid-summary one only after the test
  was tightened to read the summary section).
- **Pending (Step 6):** T4 torch/CUDA/pip freeze, `pip check`, uninstall
  diffusers/gradio/jax, then pin torch.
- **Not done:** no ARC-Easy cache, no real training.

---

## 2026-10-19 — A1 + A2: parameters that never receive gradient

- **Commit:** `4309bde`.
- **Diagnosis first (scratch probes outside the repo):** dead parameters per
  arm (one forward/backward on a fake batch); `selector.energy_net.6.bias`
  gradient 7e-9..9e-8 vs weights 0.31..1.44, +5 shift changes scores
  ≤ 4.3e-6: cancelled by `_masked_standardize` in every arm.
- **Change:** removed `hypothesis_temp` and the `confidence` head; built
  `phase_head`, `llm_prior_weight`, validator-only weights and
  `fusion.energy_alpha` only when used; `no_gradient_parameters()` allowance;
  `parameter_counts()` in provenance and printed results; checkpoint format 3
  with `load_model_state`. Files: `models/collapse.py`, `models/interference.py`,
  `models/energy_fusion.py`, `models/persistent_reasoner.py`,
  `models/full_model.py`, `training/checkpoint.py`, `training/train.py`,
  `training/runner.py`, `training/ablations.py`, `training/environment.py`,
  `evaluation/input_ablation.py`, `evaluation/sample_inference.py`,
  `tests/test_gradient_coverage.py` (new), `tests/test_forward_unchanged.py`
  (new), `tests/test_checkpoint_provenance.py`, `tests/test_input_ablation_config.py`.
- **Environment:** `qenv` scratch venv as for A3.
- **Results:** `python -m pytest tests/test_gradient_coverage.py
  tests/test_forward_unchanged.py -q` → 74 passed; `python -m pytest tests -q`
  → 159 passed, 1 skipped in 82.63 s. Bit-identical scores vs `35243a1` for
  14 arms and `qair_v46_best.pt`. Init change for seed 0 only in `phase_head`
  of A0, A1, A2, PS. Six mutations each fail ≥ 1 test.
- **Not done:** no ARC-Easy cache, no real training.

## 2026-10-19 — A0: circuit input-dependence regression test

- **Commit:** `8bfc7bb`. Test only, no code change.
- **Test:** `tests/test_circuit_input_dependence.py` (new). Real
  `QuantumEvolutionLayer`, `lightning.qubit`, `N_QUBITS = 6`, 64 random inputs
  in (−π, π), weight seeds 0/1/2; threshold std > 1e-3 per observable (DoD §2.1).
- **Environment:** `qenv` scratch venv as for A3.
- **Results:** minimum per-observable std 5.701e-02 / 5.374e-02 / 3.923e-02;
  layer phase std over 64 hypotheses 1.347e-02. With `rotation="X"` forced
  (v40 bug) both are < 1e-12. Source mutations (`rotation="X"`, `inputs * 0`,
  `self.quantum(z * 0)`) fail 4 / 4 / 1 tests. `python -m pytest tests -q` →
  167 passed, 1 skipped in 85.96 s.
- **Not done:** `lightning.gpu` is checked in the Step 6 T4 notebook.

## 2026-10-19 — DoD §2.5, §2.6: label-leakage and option-shuffle tests

- **Commit:** `0997a99`. Tests only, no code change.
- **Tests:** `tests/test_no_label_leakage.py`, `tests/test_option_shuffle_alignment.py`
  (new). Details and mutations in issues.md §10.
- **Environment:** `qenv` scratch venv as for A3.
- **Results:** 32 passed for the two files; `python -m pytest tests -q` →
  199 passed, 1 skipped in 122.09 s. 7 of 8 mutations fail ≥ 1 test; the
  surviving one (`polarity` not permuted) is equivalent.
- **Finding:** A12, latent misalignment in `collate_fn` when K is not a
  multiple of N; unreachable from caches built by this code. Proposed, not
  fixed.

## 2026-10-19 — Phase 1 close-out

- **State:** Phase 1 closed on CPU. Fixed: E3, E7, A9, A6, A3 + A8, A4 + A10,
  A5, E10 + A11, E11 (except the torch pin), A1 + A2. Regression tests: A0,
  gradient coverage, forward unchanged, DoD §2.5, §2.6. Table with commits in
  issues.md §1 "Phase 1 close-out".
- **Suite:** 199 passed, 1 skipped in 122.09 s (`qenv`, torch 2.14.1 CPU).
- **Not done:** no T4 run, no ARC-Easy cache, no real training. A7 moves to
  Tier 2 under provisional decision PD-2. A12 open (proposed).
- **Next:** proposal for DoD §2.3, §2.7 and §2.2 logging (before any baseline
  run), then the Step 6 T4 notebook and Phase 2.

## 2026-10-19 — DoD §2.3: collapse correctness test

- **Commit:** `f355965`. Test only, no code change.
- **Test:** `tests/test_collapse_correctness.py` (new); hand-computed tiny
  example, details in issues.md §10.
- **Environment:** `qenv` scratch venv as for A3.
- **Results:** 22 passed; `python -m pytest tests -q` → 221 passed, 1 skipped
  in 122.99 s. 7 mutations fail ≥ 1 test; 3 single-mask removals are
  equivalent (redundant masking).
