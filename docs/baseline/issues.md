# qAIR baseline — issue inventory (baseline = v46)

Audit date: 2026-10-17. Re-verified against **v46** on branch `baseline`.
**Nothing in this file has been fixed**, except the repository steps you
approved explicitly (marked *resolved by approved git step*).

## 0. Scope and conventions

- **Baseline code = v46**, commit `9552498`, tag `capstone-final`, branch
  `baseline` in `qAIR CSE498R/qAIR/qAIR`. The commits on top of it are
  housekeeping only: untracked checkpoints, `.gitignore`, CLAUDE.md, and docs.
  The Python source is byte-identical to `9552498`.
- Every `file:line` refers to the `baseline` working tree unless prefixed:
  - `[v40]` = local `main` @ `ac614e6` (untouched; not the baseline).
  - `[v47 patch]` = `~/qair_backup/v47_uncommitted.patch` (sha256
    `59a0d595…6010437`; touches `CLAUDE.md`, `models/validator.py`,
    `training/evaluate.py`, `training/train.py`). **Not applied, not approved.**
- v46 and the tree audited earlier differ only in those 4 files [VERIFIED:
  `git diff --stat 9552498` in the CSE499B clone]. Evidence in all other files
  is unchanged; line numbers in `train.py` and `evaluate.py` were re-taken from v46.
- Run evidence comes from the v46 checkpoints (`ckpt/*.pt`, now untracked but on
  disk), read with a torch-free zip/pickle reader, and from the executed
  notebook `~/Work/Research/qAIR/notebooks/qAIR_v46 (1).ipynb` (0-based cells).
  Both come **from v46 runs**, so they are direct evidence about the baseline.
- `docs/future_scope/01_current_state.md` does not exist anywhere on this
  machine [VERIFIED: `mdfind`, `find`]. The starting list was CLAUDE.md's
  "Known issues" (mapping in §8).
- Severity: **Critical** invalidates results · **High** biases or blocks a
  correct measurement · **Medium** wrong output in a reachable case · **Low**
  hygiene.

---

## 1. Fix order

C items are never in the fix order: they stay **needs supervisor
discussion**. "Blocked" means the item cannot start until someone outside
this repo decides.

**Phase 1 — correctness and reproducibility (before any ARC-Easy cache build)**
1. **E3** Unseeded hypothesis generation → caches not reproducible. *Blocks the cache build.*
2. **E7** A capped cache silently regenerates (hours of LLM time). *Blocks the cache build.*
3. **A6** Cache edge cases (complete flag rewritten, silent split substitution, silent zero scores). Same file as E3/E7 (`training/dataset.py`).
   **A9** (found during E3): any load error moves the cache aside and rebuilds it. Proposed right after E7, same file.
4. **A3** Run name lacks the seed / resume ignores config → a second seed resumes the first.
5. **A4** Resumed history discarded.
   **A10** (found during A4): an early-stopped run trains again when re-run. Fixed with A4 (same saved state).
   **A11** (found during A4): a resumed run is not RNG-continuous with an uninterrupted one. Proposed with E10.
6. **A5** `input_ablation` rebuilds the wrong architecture without a saved config.
7. **A1, A2** Parameters that never receive gradient. The fix approach (remove vs exclude from the optimizer) needs confirmation because of checkpoint compatibility (rule 9).
8. **A7** Validator relevance target has no hypothesis axis. **Blocked: supervisor approval.** A candidate fix exists in the v47 patch. *Provisional decision PD-2 (2026-10-19): an extra flagged arm in Tier 2, v46 target the default; the v47 patch is reference only.*
9. Regression tests from E8: circuit input-dependence (A0, `8bfc7bb`) and gradient coverage (A1/A2, `4309bde`) (DoD §2.1–2.2). *Done.*
   Added: no label leakage (DoD §2.5) and option-shuffle alignment (DoD §2.6), `0997a99`. **A12** (found writing §2.6): latent shuffle misalignment when K is not a multiple of N; fixed `59f4974`.

**Phase 2 — remaining engineering**
10. **E6** Run provenance manifest (commit, cache hash, versions, device, seed). *Fixed `eb855a2`.*
11. **E5** `ablation_results.pt` overwritten per invocation. *Fixed `fa24deb`.*
12. **E8** (rest) Timed 20-question smoke run on T4 to replace the cost assumptions; `tests/`. *Script `6f99a49`; T4 measurement in the Step 6 notebook.*
13. **E9** Stale docs and names. *Fixed `6802e75`.*
→ Then **build the ARC-Easy cache**. This is a >1 h T4 run, so it needs your go-ahead (CLAUDE.md).

**Phase 2 close-out (2026-10-27).** E6, E5, E8 (script) and E9 are fixed on CPU;
full suite 256 passed, 1 skipped in 144.45 s. Not yet done on a T4: the
smoke timing measurement (E8), the same-seed determinism runs and the
`lightning.gpu` check (E10), the torch pin (E11) and the ARC-Easy cache build;
all are in the Step 6 notebook. Defaults are now `ARC_CONFIG = "ARC-Easy"`,
`EPOCHS = 20`; the capstone configuration needs
`python main.py --arc-config ARC-Challenge --epochs 8 --train-samples 80 --val-samples 20`
(`--train-samples`/`--val-samples` added 2026-10-28).

**Phase 3 — evaluation (once the ARC-Easy cache exists)**
14. **B4** LLM letter-pick baseline · **B5** majority-vote baseline.
15. **B6** Paired item-level statistics; per-seed McNemar; Holm.
16. **B3** Select on validation, report test once · **B7** probe ceiling selection.
17. **B1, B2, B8, B10** Satisfied by full-split, 3-seed runs with drift/grad-norm logging and a full-split cache gate.

**Phase 4 — data**
18. **D1, D2** Re-measure the hypothesis signal and diversity on the full ARC-Easy cache.
19. **D3, D4** Prompt sees one option; encoder truncation / single sample.

**Needs supervisor discussion (no change without approval):** C1, C2, C3, C4, C5.

**Resolved by approved git step:** E1, E2 (for the baseline), E4 (partly; see entry).
**Withdrawn:** B9.
**Verified not present in v46:** A0.

### Phase 1 close-out (2026-10-19)

Phase 1 is closed except A7, which never was a Phase 1 fix: under
provisional decision PD-2 (`supervisor_decisions.md`) it becomes an extra
flagged arm in Tier 2 (Step 9), with the v46 target as the default. All
work below is tested on CPU only (`qenv` venv: torch 2.14.1 CPU, PennyLane
0.42.3, lightning 0.42.0); nothing has run on a T4 and no ARC-Easy cache
exists yet.

| Item | What it guarantees now | Fix / test commit |
|---|---|---|
| E3 | Seeded, per-batch-reseeded generation; seed and decoding config in the cache header | `b5c0ae8` |
| E7 | Reading a cache never builds one (`allow_build=False`) | `6aa20a5` |
| A9 | A failed cache load leaves the file untouched | `0313a15` |
| A6 | Complete flag final; missing split raises; `score_failed` flag | `0a4f464` |
| A3 + A8 | Run key `{arm}_{dataset}_s{seed}`; provenance-checked resume and best-checkpoint load; safe `torch.load` | `f605c55` |
| A4 + A10 | History and early-stopping state survive a resume; a finished run is skipped | `4083ecf` |
| A5 | `input_ablation` rebuilds the trained architecture or refuses | `b00595a` |
| E10 + A11 | Deterministic mode on by default; resume restores every RNG stream | `bbea237` (CPU) |
| E11 | Pins checked at startup, environment recorded and compared on resume | `4c4a766` (torch pin pending) |
| A1 + A2 | No parameter without gradient outside a documented allowance; counts in every result; checkpoint format 3 | `4309bde` |
| Regression: gradient coverage, forward unchanged (DoD §2.2 test part) | | `4309bde` |
| Regression: A0 circuit input-dependence (DoD §2.1) | | `8bfc7bb` |
| Regression: no label leakage (DoD §2.5), option-shuffle alignment (DoD §2.6) | | `0997a99` |

Full suite at close: **199 passed, 1 skipped** (the real-Qwen test) in
122.09 s.

**Still open after Phase 1:**
- **Step 6 T4 notebook** (not run): two same-seed 1-epoch runs with a
  per-example prediction diff, printing any op that raises in strict
  deterministic mode; `lightning.gpu` nondeterminism check; torch/CUDA,
  `pip freeze`, `pip check`, uninstall diffusers/gradio/jax, then pin torch
  (E10, E11).
- **A12** (found writing the §2.6 test): latent shuffle misalignment when K
  is not a multiple of N; fixed 2026-10-27 in `59f4974`.
- **Before any baseline run** (decision 2026-10-19, approved): DoD §2.3
  collapse correctness (done, `f355965`), §2.7 per-epoch parameter drift and
  §2.2 per-epoch per-module gradient norms (done, `402808e`).
- **Phase 2:** E6 manifest, E5, E8, E9 — closed 2026-10-27 (see Phase 2
  close-out above); then the ARC-Easy cache build (Step 6).

**Process incidents (both fixed forward with new commits, no force-push):**
`7abc115` untracked three local files that `git add docs` had committed in
`2dedcd8`; `38cdc0e` corrected docs commit `9c5f550`, which named the wrong
A5 fix hash after a zsh word-splitting failure skipped the code commit.

---

## A. Bugs

### A0 — Input-ignoring circuit (Hadamard + AngleEmbedding default `rotation="X"`)
- **Category:** A · **Status:** **verified NOT present in v46**; present in `[v40]`.
- **v46 lines:** `models/quantum_layer.py:178-183`:
  ```python
  for i in range(n_qubits):
      qml.Hadamard(wires=i)
  qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
  qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
  ```
  `[v40] models/quantum_layer.py:30-35` has the same circuit without `rotation=`,
  so it uses the default X rotation.
- **Numerical test** (scratch script; runs the literal `circuit` source from each
  commit, with only `interface="torch"`→`"numpy"` substituted;
  PennyLane 0.42.3; 6 qubits, 3 layers, 64 random inputs in (−π, π), 3 weight draws):

  | commit | device | per-observable std over inputs (min – max) | max \|f(x0)−f(x1)\| |
  |---|---|---|---|
  | v46 `9552498` | lightning.qubit | 3.6e-02 – 2.2e-01 / 7.8e-02 – 1.7e-01 / 4.3e-02 – 3.0e-01 | 7.4e-01 / 3.6e-01 / 3.4e-01 |
  | v40 `ac614e6` | default.qubit | 6.2e-17 – 2.3e-16 / 6.8e-17 – 1.9e-16 / 7.5e-17 – 2.6e-16 | 2.2e-16 / 2.8e-16 / 3.9e-16 |

  v46's output depends on its input. v40's is constant up to floating-point
  rounding (about 1e-16), so the test detects the bug when it is present.
- **Regression test (2026-10-19, `8bfc7bb`):** `tests/test_circuit_input_dependence.py`
  runs the real `QuantumEvolutionLayer` circuit on `lightning.qubit`, 6 qubits,
  64 random inputs in (−π, π), 3 weight draws. Every one of the 18 observables
  must have std > 1e-3 (DoD §2.1); measured minimum 5.7e-02 / 5.4e-02 / 3.9e-02.
  The layer's phase must vary over 64 hypotheses (measured std 1.3e-02).
  Negative control: forcing `AngleEmbedding`'s default `rotation="X"` into the
  real layer code gives std < 1e-12 for both. Mutations of the source
  (`rotation="X"`; `inputs * 0`; zeroed circuit input in `forward`) each fail
  ≥ 1 test. Do not train from local `main`.

### A1 — `CoherentCollapse.hypothesis_temp` is a dead parameter
- **Category:** A · **Severity:** Medium · **Status:** **fixed** with A2 in `4309bde`
- **Evidence:** `models/full_model.py:405` always passes
  `hyp_temperature=collapse_out["temperature"]`, so `models/interference.py:141`
  (the only read of `hypothesis_temp`, defined at `:104`) never runs. Drift after
  training is exactly `-0.000000` in all five v46-run checkpoints
  (`qair_v46_best/latest`, `A3_persistent_s42_latest`,
  `P0_classical_collapse_s42_latest`, `F0_full_with_llm_prior_s42_latest`).
- **Proposed fix:** Use a single temperature source. Removing the parameter
  changes the state_dict, so keep a loader for old checkpoints (rule 9) or exclude
  it from the optimizer instead. **Approach needs confirmation.**
- **Verify:** The gradient-coverage test (DoD §2.2) reports no
  `grad is None`/zero parameter.

### A2 — Other parameters that never receive gradient
- **Category:** A · **Severity:** Low–Medium · **Status:** **fixed** with A1 in `4309bde`; fix and evidence below
- **Evidence:** `models/collapse.py:55-60,85`: the `confidence` head is only
  returned (`models/full_model.py:470`) and never used in `training/losses.py`.
  `models/full_model.py:203,384-387`: `phase_head` is unused when
  `phase_source="circuit"` and a bottleneck exists (the default).
  `llm_prior_weight` is unused when `use_llm_prior=False` (expected in the
  ablation arms).
- **Proposed fix / verify:** As for A1. Record trainable-parameter counts per arm.
- **Measured before the fix (2026-10-19, scratch probe, one forward/backward
  per arm on a fake batch):** besides the dossier's 7 tensors (675 params in
  the main config), arm-specific dead parameters: `llm_prior_weight` (prior
  off), `reasoner.validator_weight` and `fusion.validator_alpha` (validator
  off), `fusion.energy_alpha` (no bottleneck: A0, A1), and in P0/P1 the phase
  path (`polarity_phase`, `quantum.phase_gain`, `selector.phase_net.*`); in
  PS `quantum.phase_gain`; in E0 `polarity_phase` gets an exactly zero
  gradient.
- **Diagnosis of `selector.energy_net.6.bias` (zero gradient in A1b):** the
  output bias of the learned energy is a constant added to every
  (hypothesis, option) pair; `_masked_standardize`
  (`models/answer_selector.py:16-32`) subtracts the per-sample mean, so it
  cancels exactly in **every** arm. Measured: gradient 7e-9 to 9e-8 against
  max |weight gradient| 0.31 to 1.44; shifting the bias by +5 changes the
  scores by at most 4.3e-6 (float precision). The exact 0.0 in A1b was a
  rounding coincidence. AdamW normalizes these noise gradients to full-size
  steps, so the bias random-walks during training without changing the
  output.
- **Fix (2026-10-19, decisions approved 2026-10-19), A1 and A2 together:**
  1. removed `interference.hypothesis_temp` and its unreachable branch
     (`hyp_temperature` is now a required keyword argument), and the
     `collapse.confidence` head with its `collapse_confidence` output (no
     consumer);
  2. built only when used: `phase_head` (`phase_source="learned"` or no
     bottleneck; `use_phase_head`), `llm_prior_weight` (`use_llm_prior`),
     `reasoner.validator_weight` (validator feedback), `fusion.validator_alpha`
     (validator), `fusion.energy_alpha` (bottleneck). `EnergyFusion` and
     `PersistentReasoner` raise if given a signal they were built without;
     the verbose prints in `training/train.py` skip absent scalars;
  3. **documented no-gradient allowance** (`QAIRvNext.no_gradient_parameters`,
     name → reason): `selector.energy_net.6.bias` in every arm (diagnosis
     above); P0/P1 phase path, kept for parameter matching (decision
     2026-10-19); `quantum.phase_gain` in PS; `polarity_phase` in E0. The
     last two were not in the approved removal list, so they are in the
     allowance, not removed;
  4. `parameter_counts()` (total; trainable = total − allowance) in the
     provenance (informational) and in the environment line printed with
     every result, plus a column in the grid summary;
  5. **checkpoint format 3**; `load_model_state` drops, from format-1/2
     checkpoints only, exactly the keys in `REMOVED_PARAMETER_KEYS` that the
     model does not have, prints them, and loads strictly; only format 3 is
     resumed. Used by `load_or_resume`, `load_best`, `input_ablation` and
     `sample_inference`.
  - No tensor shapes change; the forward pass is unchanged (below).
- **Per-arm parameter counts after the fix** (`QAIRvNext.parameter_counts`):

  | arm | tensors | total | trainable | no-gradient allowance |
  |---|---|---|---|---|
  | A0_no_question | 54 | 481,808 | 481,807 | energy_net.6.bias |
  | A1_baseline | 54 | 629,648 | 629,647 | energy_net.6.bias |
  | A1b_quantum_only | 81 | 757,326 | 757,325 | energy_net.6.bias |
  | A1c_classical_control_only | 86 | 758,082 | 758,081 | energy_net.6.bias |
  | A2_validator | 83 | 928,028 | 928,027 | energy_net.6.bias |
  | A3_persistent, A4, M0 | 110 | 1,055,706 | 1,055,705 | energy_net.6.bias |
  | A3c_classical_control_persistent | 115 | 1,056,462 | 1,056,461 | energy_net.6.bias |
  | P0_classical_collapse, P1_phase_zero | 110 | 1,055,706 | 1,054,803 | energy_net.6.bias, polarity_phase, quantum.phase_gain, selector.phase_net.{weight,bias} |
  | PS_learned_phase_head | 112 | 1,055,835 | 1,055,833 | energy_net.6.bias, quantum.phase_gain |
  | E0_support_only | 110 | 1,055,706 | 1,055,704 | energy_net.6.bias, polarity_phase |
  | F0_full_with_llm_prior (= `--mode train` config) | 111 | 1,055,707 | 1,055,706 | energy_net.6.bias |

  (`energy_net.6.bias` = `selector.energy_net.6.bias`; the main config was
  1,056,382 parameters in v46, 675 more, as the dossier predicted.)
- **Initialization order (measured, `tests/test_forward_unchanged.py`):** the
  removed `confidence` head drew random numbers during construction, so a
  module built after it initializes differently for the same seed. Only
  `phase_head` is built after it: its initial weights change in the 4 arms
  that still have it (A0, A1, A2, PS); every other initial weight of every
  arm is bit-identical to the pre-change code (`35243a1`) for seed 0.
  Consequence: same-seed runs of the new code do not reproduce old v46 runs
  in those arms; v46 runs were not reproducible anyway
  (`deterministic=False`, E10).
- **Test evidence:**
  - `tests/test_gradient_coverage.py` (14 arms × 3 tests + 1): every parameter
    outside the allowance has a non-None, non-zero gradient; every allowed
    parameter is None/zero or below 1e-5 × the largest gradient; removed
    parameters exist only where used; counts add up; shifting the energy bias
    leaves scores unchanged (< 1e-4).
  - `tests/test_forward_unchanged.py`: the pre-change tree (`git archive
    35243a1`) and this code, given the same weights, give **bit-identical**
    scores for all 14 arms on a synthetic batch (one padded option), and for
    `qair_v46_best.pt` (7 keys dropped, logged); the key-drop loader drops only
    known keys and only from old formats; a format-2 checkpoint is not resumed.
  - Full suite: 159 passed, 1 skipped. Mutations (`phase_head` always built →
    19 failed; `llm_prior_weight` always built → 26; allowance too wide → 14;
    loader drops any key → 1; forward changed in the `phase_head` path → 4;
    format check removed → 1).

### A3 — Single-run name lacks the seed; resume ignores the config
- **Category:** A · **Severity:** High · **Status:** **fixed** in `f605c55`
- **Evidence:** `training/runner.py:125` sets `name="qair_v45.1"`.
  `training/checkpoint.py:15,25-46` loads `{name}_latest.pt` with no seed or
  config check. `--mode train --seed 43` after a seed-42 run therefore resumes
  seed 42. `training/ablations.py:227` does include the seed.
- **Proposed fix:** Put seed, arm and dataset in the run name; store the config in
  the checkpoint; refuse to resume on a mismatch.
- **Verify:** Two 1-epoch smoke runs (≤20 examples, separate cache dir) with
  different seeds → two checkpoint files, different initial loss.
- **Added to A3's scope (2026-10-18, from A9):** every checkpoint/history/config
  `torch.load` call omits `weights_only`, so under torch ≥ 2.6 (default
  `weights_only=True`) any non-allow-listed object in a checkpoint makes it
  fail to load: `training/checkpoint.py:30` (latest checkpoint), `:56`
  (resumed history), `training/runner.py:219` (best checkpoint),
  `training/ablations.py:300` (best checkpoint), `evaluation/input_ablation.py:255`
  (config), `:262` (checkpoint), `evaluation/sample_inference.py:136`
  (checkpoint). A3 rewrites the checkpoint format, so it must load these
  explicitly with `weights_only=True` (allow-listing only plain, safe types),
  never fall back to `weights_only=False`, and test that a saved checkpoint
  reloads under that setting.
- **Fix (2026-10-19, decisions approved 2026-10-19):**
  1. run key `run_key(arm, arc_config, seed)` = `f"{arm}_{arc_config}_s{seed}"`
     for `--mode train` (arm `"qair"`, was `"qair_v45.1"`) and for every
     grid arm (was `f"{name}_s{seed}"`);
  2. checkpoint format 2 adds `format` and `provenance`: arm, dataset, seed,
     resolved model config (grid: plus `n_qubits`), training config (epochs,
     patience, batch size, both learning rates, weight decay, label
     smoothing, requested train/val sizes), cache version, actual train/val
     sizes, train and validation `hypotheses_sha256`, a **code fingerprint**
     and, for information only, the full git commit, torch version and device;
  3. code fingerprint (replaces a commit check, so a docs-only commit does
     not block a resume): `code_sha256` = sha256 over sorted paths and
     working-tree contents of tracked `*.py` and `requirements.txt`, plus
     `code_diff_sha256` = sha256 of `git diff HEAD` over the same files
     (None when clean). Both are compared. Untracked `.py` files are not
     covered. Without git both are None and a resume refuses;
  4. `load_or_resume` and `load_best` raise `CheckpointMismatchError` on any
     compared difference (each key with checkpoint and current value), on a
     format-1 file at the run's path ("never resumed, never overwritten"),
     and on a `_best.pt` without a `_latest.pt` whose provenance differs
     (it would otherwise be overwritten or evaluated as this run). The file
     is left untouched. The grid's `[SKIP]` now requires matching provenance;
  5. one loader, `training.checkpoint.load_checkpoint_file`
     (`weights_only=True`, no allow-list, never `False`), at all 7 sites
     above plus `evaluation/stats.py` (results) and
     `visualization/generate_report.py` (history). All 21 v46 files in
     `ckpt/` load this way without an allow-list.
  - No tensor shape changes. Format-1 checkpoints still load for
    evaluation (rule 9); they are only refused for resume.
  - `Trainer` and `load_or_resume` now require `provenance`, so the
    `Trainer` / `load_or_resume` cells of `notebooks/qAIR_v46.ipynb` no
    longer run. The notebook is left untouched as historical evidence
    (decision 2026-10-19); a new notebook comes in step 6.
  - **Always commit before starting a training run; the code fingerprint ignores untracked .py files.**
  - Not done here: the resumed history is still discarded (A4);
    `input_ablation` still falls back to a default config (A5).
- **Test evidence:** `tests/test_checkpoint_provenance.py` 28 passed (CPU,
  torch 2.14.1, pennylane 0.42.3, pennylane-lightning 0.42.0): round trip
  under `weights_only=True`; matching provenance resumes; a change in seed,
  model config, training config, train hash, `code_sha256` or
  `code_diff_sha256` each raises and leaves the file byte-identical; commit,
  torch version and device alone do not block; format-1 file, unknown code
  fingerprint and stale best each raise; a pickle whose `__reduce__` creates
  a file is refused and the file is not created; 5 v46 files load. In a
  temporary git repo, a docs-only commit still resumes; a change to a `.py`
  file or `requirements.txt`, committed or not, refuses. Real model on a fake
  29-question cache, 16 train / 8 val, 1 epoch, CPU: seeds 42 and 43 give
  `qair_ARC-Easy_s42` / `_s43` with epoch-1 train loss 1.8829 / 2.6331; a
  grid arm (A1_baseline) re-run with matching provenance prints `[SKIP]`, and
  with a tampered `code_sha256` raises with the file unchanged. Full suite:
  51 passed, 1 skipped. Mutations (seed-free key; no-op check;
  `weights_only=False`; commit compared; no stale-best check at start;
  `requirements.txt` not fingerprinted) each fail ≥ 1 test.

### A4 — Resumed training history discarded and overwritten
- **Category:** A · **Severity:** Medium · **Status:** **fixed** in `4083ecf`
- **Evidence:** `training/runner.py:175` discards the resumed history
  (`start_epoch, best_acc, _ = load_or_resume(…)`). `training/train.py:132-145`
  starts a fresh `history`, and `:203` overwrites the file every epoch.
  `training/ablations.py:265-278` uses `resumed_history` only for finished runs.
  `epochs_no_improve` resets on resume (`training/train.py:147`). Observed:
  `A3_persistent_s42_latest.pt` is at epoch 7, but its history file holds
  **1** epoch; `qair_v45_history.pt` holds 2.
- **Proposed fix / verify:** Merge histories and persist `epochs_no_improve`.
  An interrupted-then-resumed smoke run should have one history entry per epoch.
- **Fix (2026-10-19, approved 2026-10-19):**
  1. `Trainer.save_checkpoint` stores `history`, `epochs_no_improve` and
     `stopped_early` in every checkpoint, so they are saved atomically with
     the weights and covered by the A3 provenance check;
     `{run}_history.pt` remains the exported copy (written once per epoch,
     after `best_acc` is updated);
  2. `load_or_resume` returns the full resume state (start epoch, best acc,
     history, `epochs_no_improve`, `stopped_early`) read from the checkpoint
     only; the `_history.pt` read and its catch-all `except` that printed a
     warning and continued with no history are gone. A format-2 checkpoint
     without these keys raises (none exists outside tests; format stays 2);
  3. `Trainer.train(…, history, epochs_no_improve)` appends to the restored
     history and continues the early-stopping count; `Trainer.train_from`
     (shared by `runner.py` and `ablations.py`) trains or skips;
     `run_training` gains a `patience` argument (default `PATIENCE`).
  - No tensor shape changes.
- **Test evidence:** `tests/test_resume_state.py` (real model on CPU, fake
  cache, scripted validation accuracy): a 4-epoch run interrupted after
  epoch 2 and resumed has `acc == [0.5, 0.625, 0.375, 0.75]`, 4 losses and
  the same exported history as the uninterrupted run; with patience 2 and
  accs `[0.5, 0.375, 0.375, 0.375]`, interrupted after epoch 2, it stops
  after 3 epochs like the uninterrupted run (checkpoint: epoch 2,
  `epochs_no_improve` 2, `stopped_early` True); a checkpoint missing
  `history` is refused. Mutations: history dropped on resume → 2 failed;
  count reset → 1; `stopped_early` ignored → 2; never saved → 3; missing
  state accepted → 1.
- **Limitation found (A11):** the resumed *training* is not the same
  computation: losses for that 4-epoch run, uninterrupted
  `[1.8829, 1.3686, 1.0157, 0.8861]`, resumed `[1.8829, 1.3686, 0.9016, 0.8123]`.

### A5 — `input_ablation` can rebuild the wrong architecture
- **Category:** A · **Severity:** High · **Status:** **fixed** in `b00595a`
- **Evidence:** `evaluation/input_ablation.py:252-258` falls back to a default
  config, and `:232` defaults `use_llm_prior=True`. Ablation runs never write
  `_config.pt` (it is written only at `training/runner.py:188-206`); `ckpt/` has
  config files only for `qair_v45` and `qair_v46`. All arms share one parameter
  set, so `load_state_dict` succeeds silently.
  **Correction (2026-10-19, measured during A1/A2):** arms do *not* all share one parameter set. In v46, A0/A1 had 63 tensors, A2 90, A1b 91, A1c 96, the quantum full-stack arms 118 and A3c 123. The failure mode holds for any two arms with the same set (e.g. A3 vs F0, which differ only in `use_llm_prior`; or A3 vs P0/P1/M0/E0/A4). Since A1/A2 the sets differ more (e.g. `llm_prior_weight` exists only with the prior on), which also lets strict loading catch more mismatches.
- **Proposed fix / verify:** Store the config in every checkpoint and refuse to
  run without it. The "intact" accuracy should equal the logged validation
  accuracy.
- **Fix (2026-10-19, approved 2026-10-19):** (the config has been in every
  checkpoint's provenance since A3)
  1. `resolve_model_config`: a format-2 checkpoint is rebuilt from
     `provenance["model_config"]`; an explicit `--config` must agree on every
     architecture key, else `ModelConfigError`. A format-1 (v46) checkpoint
     needs `--config` or its `{run}_config.pt`; otherwise it raises. The
     silent full-hybrid default is gone;
  2. `build_model_from_config` requires every key in `MODEL_CONFIG_KEYS`
     (model_dim, use_quantum, use_validator, persistent_steps, n_qubits,
     use_question, phase_mode, phase_source, mixing, use_llm_prior,
     use_attack); only `backend` is optional (classical-control arms);
  3. `load_for_ablation`: for a format-2 checkpoint, `--arc-config` defaults
     to the run's and must not contradict it; on validation, `--max-samples`
     defaults to the run's `n_val`, and the validation set must match the
     run's `(n_val, val_hypotheses_sha256)`;
  4. grid provenance now records `model_dim` (approved); grid checkpoints
     written before this change (tests only) no longer resume.
  - No tensor shape changes.
- **Test evidence:** `tests/test_input_ablation_config.py` 10 passed (real
  model, CPU, fake 29-question cache, 16 train / 8 val, 2 epochs, seed 42):
  "intact" equals the checkpoint's logged `best_acc` exactly for `qair`
  (0.875 = 0.875), `A1_baseline` (0.75 = 0.75) and
  `A3c_classical_control_persistent` (0.625 = 0.625); these are accuracies on
  fake data and mean nothing beyond the equality. The old default dict, a
  format-1 file without config, a contradicting `--config`, 7 instead of 8
  validation samples, and `--arc-config ARC-Challenge` each raise.
  `qair_v46_best.pt` builds from `qair_v46_config.pt`, and
  `A3_persistent_s42_best.pt` from an explicit config. Full suite: 66 passed,
  1 skipped. Mutations: default fallback restored → 1 failed; contradiction
  check removed → 1; data check removed → 1; stored config altered
  (`use_llm_prior` flipped) → 3 (the three equality tests).

### A6 — Cache edge cases silently change state or data
- **Category:** A · **Severity:** Medium · **Status:** **fixed** in `0a4f464`
- **Evidence:** (1) `training/dataset.py:154-172` + `:437-439`: a complete cache
  smaller than `max_samples` is re-saved as `complete=False`.
  (2) `:203-209`: a missing split is silently replaced by `validation`.
  (3) `:256-260`: a scoring failure stores zeros for `llm_logprob`;
  `:235-254`: generation failure → template fallbacks, with no per-sample flag
  for the scoring failure.
- **Proposed fix / verify:** Treat `complete` as final, raise on a missing split,
  add a per-sample `score_failed` flag. Unit tests with a fake generator.
- **State after E7:** case (1) could still happen through
  `scripts/build_cache.py` (`allow_build=True`), e.g. `--limit 3000` on a
  complete 2,251-item train cache; a plain read raised instead of returning
  the cache.
- **Fix (2026-10-18):**
  1. a complete cache is final: it returns all samples (or the first
     `max_samples`, printing `[CACHE COMPLETE] … returning all N` when fewer
     exist) and never enters the build path, even with `allow_build=True`;
  2. a missing split raises `ValueError` with the available splits, checked
     before any lock or model load (`load_arc` moved ahead of the generator);
  3. a batch scoring failure is retried per question; only questions that
     still fail get zero `llm_logprob` and `score_failed=True` (others
     `False`); `report_quality` prints `score_failed n/N` and warns if > 0.
  - **Cache version kept at 46** (decided 2026-10-18): 46 was introduced by
    E3 the same day and no real v46 cache existed yet. The loader sets
    `score_failed = False` when the key is absent (v45 caches, which never
    recorded it).
- **Test evidence:** `tests/test_cache_edge_cases.py` 5 passed: a complete
  29-sample cache read with `max_samples=40` returns 29 with and without
  `allow_build`, stays byte-identical and `complete=True`, and never constructs
  the generator; `split="test"` on a train/validation-only dataset raises and
  leaves no cache or lock; with a batch scorer that fails and one question
  that fails alone, exactly that sample is flagged with zeros and the
  report prints `score_failed 1/8`; a sample without the key reads as
  `False`. Full suite 23 passed, 1 skipped. Mutations (old early return; split
  substitution; batch zeros) each fail their test.

### A7 — v46 validator relevance target has no hypothesis axis *(new)*
- **Category:** A · **Severity:** High · **Status:** open · **fix needs supervisor approval** · provisional decision PD-2 (2026-10-19, `supervisor_decisions.md`): implement only as an extra arm behind a flag in Tier 2; the v46 target stays the default and the main baseline
- **Evidence:** `models/validator.py:140-142`:
  ```python
  if y is not None:
      target = torch.zeros_like(relevance)
      target[torch.arange(B, device=H.device), :, y] = 1.0
  ```
  The `:` is the hypothesis axis, so every hypothesis (support and attack) gets
  the same target. `relevance` reaches the collapse energy only through its mean
  over options (`:159-170`, `mean(relevance)` at `:167`). Under this target that
  mean is 1/N for every k [VERIFIED: algebra], so the term cannot separate
  hypotheses even at zero loss. Consistent with this (but not proof of cause),
  collapse entropy in the v46 run is 2.0763 vs log 8 ≈ 2.0794 (near-uniform;
  `qair_v46_history.pt`).
- **Candidate fix:** `[v47 patch]` `models/validator.py::_relevance_target` gates
  the target by hypothesis soundness. Its docstring reports synthetic-task
  measurements that were **not reproduced in this session** [HYPOTHESIS until
  re-run]. The patch also adds the `amp_entropy` metric to `training/evaluate.py`
  and `training/train.py`. It changes the training objective, so it is **not
  applied**; it needs supervisor approval.
- **Verify (if approved):** On a synthetic batch where soundness is recoverable,
  the target's option-mean varies across k, and collapse weights on sound
  hypotheses exceed those on unsound ones. Then compare full runs with and without
  the change, one variable only (rule 4).

## B. Evaluation problems

### B1 — Capstone evaluation used n=20 validation / 80 train
- **Severity:** Critical · **Status:** open
- **Evidence:** `ckpt/qair_v46_config.pt` has `train_samples 80`,
  `val_samples 20`, seed 42. Notebook cell 12 sets `TRAIN_SAMPLES = 80`,
  `VAL_SAMPLES = 20`, `EPOCHS = 8`. Cell 41: `acc=0.5000 95% CI [0.3000, 0.7000]`,
  `McNemar vs the direct LLM: b=2 c=4 p=0.6875`. This is the **v46 baseline's own
  result**.

### B2 — One seed
- **Severity:** High · **Status:** open
- **Evidence:** Every saved run is seed 42 (`qair_v46_config.pt`, `*_s42_*`);
  `config.SEEDS` (`config.py:286`) was not used.

### B3 — Validation selects the checkpoint and supplies the reported number; test never evaluated
- **Severity:** High · **Status:** **fixed 2026-10-28** (CPU); closed by the full-split runs
- **Evidence:** `training/train.py:325-329` saves `best` on the max validation
  accuracy. The reported 0.50 is the max of
  `[0.35, 0.35, 0.40, 0.45, 0.40, 0.40, 0.45, 0.50]` (`qair_v46_history.pt`).
  `training/runner.py:87-95` skips test when its cache is missing; the v46
  history has no `test` key.
- **Already in place before 2026-10-28:** selection on validation only; test
  evaluated after training from the best checkpoint in `run_training` and in
  the grid.
- **Fix (2026-10-28):** `select_for_test` (`training/checkpoint.py`) loads the
  best checkpoint for the single test evaluation and raises
  `NoBestCheckpointError` when none exists; before, `load_best` returned False
  and test was evaluated on whatever weights the model held (the last epoch).
  The history (`test_from`) and each grid entry record the checkpoint, its
  1-based epoch and its validation accuracy; the printed test line names them.
- **Test evidence:** `tests/test_test_selection.py` (4 tests, fake cache with
  a test split): test is evaluated once per run, from the first-best
  validation epoch; the reported test accuracy and records equal a fresh
  evaluation of the best checkpoint; with no best checkpoint the run raises
  before any test evaluation; grid entries carry `test_from`. 4 mutations each
  fail ≥ 1 test.

### B4 — No "LLM letter pick" baseline (mandatory)
- **Severity:** High · **Status:** **fixed 2026-10-28** (CPU; not yet run on the ARC-Easy cache)
- **Evidence:** `models/generator.py:357-451` is cloze-style option likelihood
  only; `evaluation/baselines.py` has no lettered-options MCQ prompt.
- **Fix:** `python -m evaluation.baselines --mode letter`. The generator's own
  model (`HypothesisGenerator`: same name; the run refuses when the cache
  header records another generator or revision, `GeneratorMismatchError`).
  Prompt, fixed before any result (`LETTER_PROMPT`): chat template, one user
  turn "Answer the following multiple-choice question. Reply with the letter
  of the correct option only." + question + `A. …` lines + `Answer:`. Score:
  log-softmax of the next-token logits at the answer position (left padding),
  read at each option's letter token; argmax, ties to the lowest index. Every
  letter must be a single token, else `ValueError`.
- **Test evidence:** `tests/test_baselines_vote_letter.py`: a character-level
  fake model that puts the answer only at the last real token shows the
  answer position is read (right padding or the first position fail); the
  prompt letters every option; a multi-token letter is refused; another
  generator name or revision is refused; records and predictions are
  written. 6 mutations each fail ≥ 1 test. Slow test (`QAIR_SLOW=1`, real
  Qwen2.5-0.5B-Instruct, CPU, 478 s): A–H are 8 distinct single tokens; two
  runs on 20 ARC-Easy validation questions give bit-identical scores. No
  accuracy was computed.

### B5 — No majority-vote-over-hypotheses baseline (mandatory)
- **Severity:** High · **Status:** **fixed 2026-10-28** (CPU; not yet run on the ARC-Easy cache)
- **Evidence:** `evaluation/baselines.py` modes: `llm`, `asymmetry`, `probe`,
  `direct`.
- **Fix:** `--mode vote` (also in `--mode all`). The weighting is the one
  pre-registered in the dossier (B5, 2026-10-17), committed before any
  ARC-Easy result exists: `v_n = Σ_k [hyp_option_k = n] · polarity_k ·
  cos(H_k, O_n)`, all hypotheses (fallbacks included), argmax with ties to the
  lowest index. Computed from each sample's `hyp_option` and `polarity`, so it
  does not depend on the hypothesis order; on the mode-major cache it equals
  the gate's "support − attack margin". The degenerate count vote is printed
  next to it for reference.
- **Test evidence:** `tests/test_baselines_vote_letter.py`: a hand-built
  sample with known cosines gives the expected votes; shuffled hypotheses give
  the same votes; ties go to option 0; on the fake cache the accuracy equals
  the gate's margin line. Mutations (no polarity, misassigned option) fail.
- **Pre-registration kept (user, 2026-10-28):** the weighting above is the
  B5 baseline as pre-registered on 2026-10-17 and stays as is. Any
  support-only vote (motivated by the attack finding, D5) is exploratory /
  post hoc and must be labelled so.
- **`--out-dir`:** `llm`, `vote` and `letter` write `{mode}_{split}.json` with
  per-item records and predictions (for B6 and the Step 7 tables).

### B6 — Pooling records across seeds makes McNemar/bootstrap anti-conservative
- **Severity:** High · **Status:** **fixed 2026-10-28** (CPU; PD-14 families)
- **Evidence:** `evaluation/stats.py:95-100` pools `test_records` across seeds;
  `:127` runs McNemar on the pooled list; `:120` bootstraps (item, seed) cells.
  Holm is only printed (`:138-140`).
- **Fix (decision PD-14, 2026-10-28):** `evaluation/stats.py`. `paired_bootstrap`
  resamples items (10,000, seed 0), keeps every seed of a resampled item and
  averages them; p is the centred bootstrap's two-sided p. `compare` adds
  exact McNemar per seed and whether the arm is ahead in every seed. `holm`
  adjusts within a family. `success_report` runs the primary family (F0 vs
  {option likelihood, letter pick}, Holm over 2) and the secondary family
  ({A3, P0, P1} × the same, Holm over 6) and prints the PD-11 verdict;
  `python -m evaluation.stats --results … --baselines-dir …` reads the
  `evaluation/baselines.py --out-dir` test files (refusing other splits).
  The exploratory arm-vs-arm table no longer pools seeds either.
- **Test evidence:** `tests/test_stats_b6.py` (11 tests): Holm step-down;
  identical arms give p = 1, a clear win p < 0.01; items, not cells, are
  resampled; deterministic baselines repeat across seeds and mismatched
  lengths raise; per-seed McNemar and direction; every PD-11 condition is
  required (including Holm, not raw p); the families are PD-14's. Simulation
  under H0 (1000 runs, n = 200, 3 seeds, item-driven seed correlation
  κ = 0.8): pooled McNemar rejects > 12 % (18.3 % in a 300-run check), the
  item bootstrap 2–8 % (5.0 %). 8 mutations each fail ≥ 1 test.

### B7 — Probe "data ceiling" selects its best epoch on validation
- **Severity:** Medium · **Status:** **fixed 2026-10-28** (CPU) · **Evidence:** `evaluation/baselines.py:333-348`.
- **Fix:** `fit_probe`: a fixed 20 % holdout of the TRAIN split (seed 0)
  chooses the epoch count (first best, up to 60); the probe is retrained from
  the same seed on all of train for that many epochs; validation is
  evaluated exactly once. Printed with the chosen epochs and holdout accuracy.
- **Test evidence:** `tests/test_probe_selection.py` (6 tests): the holdout is
  fixed and disjoint; permuting the validation labels does not change the
  chosen epochs; validation is evaluated once, last; the final probe trains
  on all 40 train rows for exactly the chosen epochs; the first best holdout
  epoch is chosen. 7 mutations each fail ≥ 1 test.

### B8 — Optimization budget too small for claims about learned scalars
- **Severity:** High · **Status:** open
- **Evidence:** 80 / `BATCH_SIZE` 16 (`config.py:268`) = 5 steps/epoch × 8 epochs
  = **~40 steps**. All learned scalars drifted < 0.01 (`polarity_phase`
  ≤ 0.0059 rad). [HYPOTHESIS] With AdamW at LR 5e-4 / 2e-4 (`config.py:272-273`),
  40 steps cannot move a scalar by more than ~0.02, so "phases not learning" may
  be a budget artifact.

### B9 — *Withdrawn*
- Previously: "the capstone number measures pre-v47 code". The baseline is v46,
  and the p=0.6875 result came from a v46 run, so the result and the baseline are
  the same code. Its weaknesses are covered by B1–B3.

### B10 — Cache-quality gate run on capped caches
- **Severity:** Medium · **Status:** open
- **Evidence:** Notebook cell 14: the gate ran on 80 train / 20 val samples.
  Validation `support agreement 0.4000` (n=20) vs train `0.1750` (n=80).

## E. Engineering

### E1 — Restore incomplete (working tree was v40) — *resolved by approved git step*
- Branch `baseline` was created at v46 `9552498` and tagged `capstone-final`.
  Local `main` (v40) is untouched. Commit `7530559` never existed locally or on
  the remote.

### E2 — Capstone history scattered; v47 edits uncommitted — *resolved for the baseline*
- v46 history was fetched from the CSE499B clone (remote `cse499b`). The v47
  diff was backed up to `~/qair_backup/v47_uncommitted.patch` and is **still
  uncommitted** in the CSE499B clone (left as found). Nothing was pushed.

### E3 — Hypothesis generation is unseeded → caches not reproducible
- **Severity:** High · **Status:** **fixed** in `b5c0ae8` · **Phase 1, item 1**
- **Evidence:** `models/generator.py:306-314` uses `do_sample=True`,
  `temperature=0.7`, `top_p=0.95`. `scripts/build_cache.py` and
  `training/dataset.py` never call `set_seed`/`manual_seed` [VERIFIED: grep].
  `set_seed(deterministic=True)` is never used (`training/seed.py:8`).
- **Proposed fix:** Seed the build; record the seed, generation config, model
  revision and prompt hash in the cache header.
- **Verify:** Two 20-question builds with the same seed give byte-identical
  hypotheses (separate cache dirs).
- **Second cause, found while fixing:** the autosave fired on `count % 50`
  inside the per-sample loop (`training/dataset.py:328-334` in v46), i.e. in
  the middle of a generation batch. A resumed build then regrouped the
  remaining questions into different batches, so even a seeded build could not
  reproduce an uninterrupted one.
- **Fix (2026-10-18):**
  - each generation batch runs under `seeded_rng(batch_seed(gen_seed, raw
    index of its first question))`; `batch_seed` uses sha256, not Python's
    salted `hash()`; `fork_rng` leaves the caller's RNG untouched;
  - autosave only at batch boundaries (`autosave_every`, default 50);
  - cache `CACHE_VERSION` 46 with a `provenance` header: `gen_seed`,
    generator and HF revision, decoding config, `prompt_sha256` (rendered
    chat prompts + parser source), and per-segment device, device name,
    torch and transformers versions; plus `hypotheses_sha256`;
  - resuming with a different seed, generator, revision, decoding config or
    prompt raises; a cache without provenance (v45) loads read-only and is
    never resumed; a device or library change between segments is recorded
    with a warning;
  - `config.GEN_SEED = 0`; `scripts/build_cache.py --gen-seed`.
  - **Reproducibility is per device**: CPU and GPU sample differently from the
    same seed.
- **Test evidence** (`tests/test_cache_determinism.py`, CPU, scratch venv
  torch 2.14.1 / transformers 4.52.4):
  - fast suite with a fake generator that samples from torch's RNG through
    the real `seeded_rng` helper: **8 passed** in 2.75 s — same seed gives an
    identical cache; a different seed differs; a build interrupted inside its
    third batch and resumed equals the uninterrupted build; resuming with a
    different seed or prompt raises (and releases the lock); a v45 cache is
    never resumed; `batch_seed` is identical in a fresh process with a
    different `PYTHONHASHSEED`; `seeded_rng` restores the caller's RNG;
  - mutation check: the old mid-batch autosave fails the resume test; an
    unseeded batch fails the same-seed and resume tests;
  - real model (`QAIR_SLOW=1`): two 5-question ARC-Easy builds with
    Qwen2.5-0.5B-Instruct (revision `7ae55760…`) on CPU give the same
    `hypotheses_sha256` = `5b4f8404…f51d` (**1 passed**, 70 s).

### E4 — Checkpoints committed to git — *partly resolved by approved git step*
- Commit `b3cefd5` on `baseline` untracks the 21 `ckpt/*.pt` files, removes
  `!ckpt/*.pt`, and adds `ckpt/` and `cache/` to `.gitignore`. The files remain on
  disk and **in history** (no rewrite, by decision). `[v40]` main still tracks
  0-byte `cache/arc_train.pt` and `cache/arc_validation.pt` (main untouched).

### E5 — `ablation_results.pt` overwritten per invocation
- **Severity:** Medium · **Status:** **fixed 2026-10-27** in `fa24deb` · **Evidence:** `training/ablations.py:212,318`.
- **Fix:** each (arm, seed) saves `{run}_result.pt` (`training/results.py`);
  `ablation_results.pt` is rebuilt from every result file in the directory,
  in format 2: `{"format": 2, "results": {arm: [entries by seed]},
  "allow_mixed", "sources"}`. Result files whose manifests differ in code
  fingerprint, uncommitted-diff hash or any cache file sha256 are refused
  (`MixedResultsError`, listing each source and its runs) unless
  `--allow-mixed` (decision 2026-10-27), which is recorded in the file and
  printed with the results by the grid summary and `evaluation/stats.py`.
  On refusal the new runs' result files are kept and the old combined file
  is left as it was. `stats.py` still reads the v46 bare-dict file (format 1,
  printed as "code version and caches unknown").
- **Test evidence:** `tests/test_results_files.py` (8 tests): two separate
  invocations both appear; re-running one (arm, seed) replaces only its
  entry; a result file from other code or another cache is refused, the new
  result file kept and the combined file unchanged; `allow_mixed` recorded
  and the mixed note printed; stats reads formats 1 and 2; `--allow-mixed`
  reaches the grid. Eight mutations each fail ≥ 1 test. Full suite: 247
  passed, 1 skipped in 142.45 s.

### E6 — Run provenance not recorded
- **Severity:** High · **Status:** **fixed 2026-10-27** in `eb855a2` (most fields already recorded by A3, E10, E11; the manifest completes and collects them)
- **Evidence:** `training/runner.py:189-205` saves no commit, cache hash, dataset
  revision or package versions; ablation runs save no config (A5).
  `requirements.txt:1` leaves `torch` unpinned; the circuit device is `"auto"`
  (`config.py:141`).
- **Fix:** `training/manifest.py` writes `{run}_manifest.json` (plain JSON)
  when a run starts, in `--mode train` and in the grid, and the returned
  history carries it as `history["manifest"]`. Fields: git commit, `dirty`
  flag, code fingerprint and diff hash; for every cache file used, its path,
  file sha256, the sha256 of the samples used, the header file and the
  generation provenance (gen seed, generator + revision, prompt hash,
  generation config, device/library segments); dataset name, config and split
  sizes; seed; deterministic flags and `CUBLAS_WORKSPACE_CONFIG`; torch
  device and GPU name; the quantum device actually used (`None` for arms
  without a circuit); compared and recorded package versions with the
  pip-freeze hash and pin check; model and training config; parameter
  counts. Record only: resume checks are unchanged. No timestamp, so
  same-seed and resumed runs keep bit-identical histories (E10).
- **Test evidence:** `tests/test_manifest.py` (5 tests): every field present
  and equal to the history's; commit = `git rev-parse HEAD`; dirty flag set by
  an uncommitted `.py` change in a temp repo; each cache's `file_sha256`
  equals a direct sha256 of the file, its header equals the header file and
  its generation record equals the cache's provenance; a grid arm without a
  circuit records `quantum_device = None`. Six mutations each fail ≥ 1 test.
  Full suite: 239 passed, 1 skipped in 143.10 s.
- **Process note:** the first commit (`fa17749`, never pushed) went in with
  3 failing determinism tests because a piped `pytest | tail` hid the exit
  code. The cause was a `created_utc` timestamp that made same-seed histories
  differ; it was removed and the unpushed commit amended to `eb855a2`.
- **Cache-build manifest (2026-10-28, Step 6):** `build_cache_manifest` in
  `training/manifest.py` and `scripts/cache_manifest.py` write the manifest
  of a cache build without a training run (`"kind": "cache_build"`: commit,
  dirty flag, code hash, environment, one `cache_record` per split). It
  refuses an incomplete cache and suppresses the loader's cache-quality
  blocks, so it never shows test-split numbers. Notebook cell 10;
  `tests/test_manifest.py` +2 tests.

### E7 — A capped cache can silently restart hours of LLM generation
- **Severity:** Medium→High for the cache build · **Status:** **fixed** in `6aa20a5` · **Phase 1, item 2**
- **Evidence:** `training/dataset.py:157-163` (documented in the code) and
  `evaluation/baselines.py:304-309`: calling with `max_samples=None` against a
  capped (`complete=False`) cache resumes generation. Any caller other than
  `build_cache.py` (training, baselines, input ablation, notebooks) can trigger it.
- **Proposed fix:** An explicit `allow_build` flag, False everywhere except
  `scripts/build_cache.py`; otherwise raise with the cap the cache was built with.
- **Verify:** Loading a capped smoke cache with `max_samples=None` raises and
  never loads the LLM.
- **Wider than first written:** a **missing** cache also built silently, so
  training on a fresh VM without the cache would have started hours of
  generation. Callers that could reach the build path: `training/runner.py`
  (train/val/test), `evaluation/baselines.py` (5 call sites),
  `evaluation/input_ablation.py`, the v46 notebook.
- **Fix (2026-10-18):** `QAIRDataset(..., allow_build=False)`. Any request the
  stored cache cannot satisfy (missing; capped below `max_samples`; incomplete
  with `max_samples=None`) raises before the lock, the LLM or the dataset is
  touched. The message gives the path, state (stored count, `complete`,
  `raw_index`, `gen_seed`), the request and the exact `scripts/build_cache.py`
  command. Only `scripts/build_cache.py` passes `allow_build=True`. A complete
  cache asked for more than it holds now raises instead of being downgraded
  (A6 will make it return the full cache).
- **Test evidence:** `tests/test_cache_build_guard.py` (generator, encoder and
  dataset loader replaced by functions that fail if called during reads):
  5 passed; full suite 13 passed, 1 skipped (slow) in 3.8 s. Mutation check:
  with `allow_build` defaulting to True (old behaviour) the three refusal
  tests fail.

### E8 — No v45+ timing data; no tests
- **Severity:** Medium · **Status:** **fixed 2026-10-27** in `6f99a49` (script); tests exist since Phase 1. **Measurement pending:** the T4 run in the Step 6 notebook
- **Evidence:** Only the v44 rate is recorded (`1.24s/it`, batch 16, K=4,
  `qAIR_v44.ipynb` cell 21); v46 tqdm bars were saved at 0%. No `tests/`.
  `models/full_model.py:317,348` runs the reasoner and circuit twice per forward
  pass when validator feedback is on.
- **Fix:** `tests/` now has 251 tests (Phase 1, Step 5). `scripts/smoke_timing.py`
  builds a cache of 20 train + 10 validation ARC-Easy questions (decision
  2026-10-27) in a new directory and trains 1 epoch, timing generation,
  scoring, embedding, model loading, train steps, evaluation and checkpoint
  I/O by wrapping the real calls (CUDA-synchronized). It writes
  `smoke_timing.json` with per-question and per-step rates and a [DERIVED]
  projection: full cache (5197 questions), one epoch (141 steps + 570
  validation questions) and a 20-epoch run. It refuses `config.CACHE_DIR`
  and any directory that already holds a cache.
- **Test evidence:** `tests/test_smoke_timing.py` (4 tests, fake generator and
  encoder, real model): every stage timed with > 0 s, call counts, quantum
  device and the 141-step projection; all wrappers removed afterwards; a
  second run in the same directory and a run in `CACHE_DIR` are refused with
  nothing generated. Five mutations fail ≥ 1 test; not restoring the
  build-stage wrappers passes because the later restore of the same timer
  reverts them (equivalent). Full suite: 251 passed, 1 skipped in 143.26 s.

### E9 — Stale docs and names
- **Severity:** Low · **Status:** **fixed 2026-10-27** in `6802e75`
- **Evidence:** `evaluation/stats.py:4-7` (869 = merged-ARC validation);
  `training/runner.py:125,136` ("v45.1"); `scripts/build_cache.py:149`
  (`qair_v45_best.pt`); `docs/history/CLAUDE_v46.md` says "Current version: v45".
- **Fix (decisions 2026-10-27):**
  - stale statements of current behaviour fixed: the training banner
    ("qAIR v46 baseline"), checkpoint names in `scripts/build_cache.py`,
    `evaluation/input_ablation.py` and the README (`qair_ARC-Easy_s42_best.pt`),
    the validation sizes in `evaluation/stats.py`, `training/ablations.py` and
    `config.py` (ARC-Easy 570, SE up to ~2.1 points), the `QAIRvNext`
    docstring version, the README benchmark table and defaults line;
  - defaults changed: `ARC_CONFIG = "ARC-Easy"`, `EPOCHS = 20` (PD-7);
    `main.py --epochs` added. Reproducing the capstone configuration needs
    `python main.py --arc-config ARC-Challenge --epochs 8` (README, `config.py`,
    CLAUDE.md); the capstone run also capped data at 80 train / 20 validation
    in its notebook, which `main.py` did not expose. **Follow-up
    2026-10-28:** `main.py --train-samples N --val-samples N` (both modes;
    default all), so the capstone configuration is one command:
    `python main.py --arc-config ARC-Challenge --epochs 8 --train-samples 80 --val-samples 20`;
  - kept: the "v45:" history comments, historical measurements (e.g. 294/869
    in `input_ablation.py`), the README's v45 sections (marked as history) and
    `docs/history/CLAUDE_v46.md` (untouched, historical evidence).
- **Test evidence:** `tests/test_defaults.py` (5 tests): config and both entry
  points default to ARC-Easy / 20 / 5; the CLI passes the defaults and the
  capstone flags in both modes; none of the fixed stale strings appears in
  tracked code outside `tests/`. Five mutations each fail ≥ 1 test. Full
  suite: 256 passed, 1 skipped in 144.45 s.

## D. Data / input problems

### D1 — 0.5B hypotheses carry no signal beyond chance
- **Severity:** Critical (for the research question; it does not block "working") · **Status:** open; **revised 2026-10-28** — on the full ARC-Easy cache the support channel carries signal (0.3350 / 0.3298 vs chance 0.25, z +9.3 / +4.4) while the margin is at chance because the attack channel is anti-informative (dossier D1, D5)
- **Evidence:** Notebook cell 14, `[CACHE QUALITY train / ARC-Challenge] 80 samples`:
  support 0.1750, attack 0.1875, margin 0.2000, LLM likelihood 0.3375,
  `[SIGNAL WARNING]`. Chance ≈ 0.25.

### D2 — Hypotheses less diverse than options; support ≈ attack per option
- **Severity:** High · **Status:** open
- **Evidence:** Cell 14: `cos(H_sup) 0.6644`, `cos(H_att) 0.6116` vs `cos(O) 0.4949`;
  `cos(sup_n, att_n) 0.7412`.

### D3 — Each generation prompt sees one option only
- **Severity:** Medium · **Status:** open · **Evidence:** `models/generator.py:136-161`.

### D4 — Encoder truncates at 128 tokens; one sampled hypothesis per (option, mode)
- **Severity:** Medium (High for MedQA later) · **Status:** open
- **Evidence:** `models/encoder.py:47`; `models/generator.py:198,306-314`.

## C. Design limitations — **needs supervisor discussion**

### C1 — Born readout is sign-symmetric / equivalent to additive scoring
- **Severity:** High · **needs supervisor discussion**
- **Evidence:** `models/interference.py:193,217-220`; `models/full_model.py:226,390`.
  [VERIFIED algebra] With equal circuit phases, θ≈0 and attack phase +π, the
  score is (S_n − T_n)², which ranks by |S−T|. [HYPOTHESIS] Any symmetry-breaking
  reference term makes it monotone in S−T (additive scoring).

### C2 — Phase channel barely learns; `destructive_fraction` saturated by init
- **Severity:** High · **needs supervisor discussion** (fix B8 first)
- **Evidence:** `polarity_phase` drift ≤ 0.0059 rad; `destructive_fraction` is 1.0
  every epoch (`qair_v46_history.pt`); `config.py:219-234`; notebook cell 27
  shows `[INTERFERENCE SATURATED]` ×3.

### C3 — K = 2N hypotheses unpruned; no abstention
- **Severity:** Medium · **needs supervisor discussion**
- **Evidence:** No pruning in `models/full_model.py`; `training/evaluate.py:124-128`
  always takes the argmax; collapse entropy 2.0763 vs log 8 ≈ 2.0794.

### C4 — All LLM knowledge passes through frozen 384-d MiniLM embeddings
- **Severity:** High · **needs supervisor discussion**
- **Evidence:** `config.py:88-94`; `models/full_model.py:416-424`. F0 best
  validation 0.35 vs cached LLM 0.40 on the same 20 items (cells 39, 36).

### C5 — Redundant / doubled computation
- **Severity:** Low · **needs supervisor discussion**
- **Evidence:** `models/answer_selector.py:185-187`; `models/full_model.py:336-358`.

---

## 8. CLAUDE.md "Known issues" → verification (against v46)

| CLAUDE.md claim | Result | ID |
|---|---|---|
| McNemar p=0.6875, n=20, CI [30%,70%] | VERIFIED (nb cell 41), v46 run on validation | B1, B3 |
| 80 train / 20 val, one seed, ~40 steps | VERIFIED (`qair_v46_config.pt`; 5 × 8 = 40) | B1, B2, B8 |
| 0.5B hypotheses no signal | VERIFIED (train cache 0.175 / 0.1875 / 0.20 vs ~0.25) | D1 |
| Readout squared norm, additive / sign-symmetric | VERIFIED algebraically; additive equivalence [HYPOTHESIS] | C1 |
| Phases ≤0.006 rad; one temperature never moved | VERIFIED: 0.0059 rad max; `hypothesis_temp` drift 0 | C2, A1 |
| K=8 unpruned, no abstention | VERIFIED | C3 |
| Constant-circuit bug stays fixed | VERIFIED absent in v46 (numerical test), present in v40 | A0 |
| (new) validator target k-invariant | VERIFIED in v46 `models/validator.py:142` | A7 |

No label leakage at evaluation: `training/evaluate.py:122` calls `model(**inputs)`
without `y`; `y` reaches only the validator BCE target (`models/validator.py:140-142`,
`training/losses.py:115-133`).

---

## 9. Issues added by the issue dossier (2026-10-17)

Full analysis, mathematics, fixes and tests: `docs/baseline/issue_dossier.md`.
Fix-order placement: dossier §3 (A8 with A3; E10/E11 with E6; B11 in Phase 3
as measurement only; D5 in Phase 4; C6/C7 for the supervisor).

### A8 — Ablation grid resumes stale checkpoints from earlier sessions
- **Category:** A · **Severity:** High · **Status:** **fixed going forward** with A3 in `f605c55`.
  The v46 grid numbers stay invalid; nothing re-runs them. The per-version
  `CKPT_DIR` was **not** added (decision 2026-10-19): with the dataset in the
  run key and the code fingerprint checked, a stale checkpoint raises
  instead of being hidden in another directory.
- **Evidence:** v46 run cell 39: `[RESUME A1_baseline_s42] Resuming from epoch 8 …
  [SKIP] A1_baseline_s42 already finished 8 epochs.`; A3 "Resuming from epoch 7",
  P0 "Resuming from epoch 6", F0 resumed; all from
  `/content/drive/MyDrive/qAIR_V45/ckpt/`. Code: `training/ablations.py:227,265-272`.
- **Proposed fix:** shared with A3 (provenance in checkpoints; refuse mismatches),
  plus a per-version, per-dataset checkpoint directory.
- **Verify:** a grid run in a directory holding a checkpoint with a different
  commit raises.

### B11 — Calibration is neither controlled nor measured with adequate power
- **Category:** B · **Severity:** Low–Medium · **Status:** open
- **Evidence:** ECE 0.161 → 0.385 (cell 27), but a calibrated model has an
  expected 15-bin ECE of 0.241 at n=20 (`scripts/verify_dossier_math.py`
  check 10), so the observation is weak. Mean confidence is not logged
  (`training/evaluate.py:212-242`).
- **Proposed fix:** log confidence vs accuracy per epoch; ECE with a bootstrap CI
  and a reliability diagram on full splits.
- **Verify:** at n=570 ECE is either clearly above the ≈0.05 noise floor or not.

### E10 — Training is non-deterministic by default
- **Category:** E · **Severity:** Medium · **Status:** **fixed on CPU** in `bbea237`; T4 check pending (Step 6)
- **Evidence:** `[SEED] Set to 42 (deterministic=False)` (cells 10, 39);
  `training/seed.py:8`; no caller passes `deterministic=True` (grep).
- **Proposed fix:** `deterministic=True` for reported runs; record it in the manifest.
- **Verify:** two same-seed T4 smoke runs give identical predictions.
- **Fix (2026-10-19, decisions approved 2026-10-19):**
  1. `config.DETERMINISTIC = True`, `config.DETERMINISTIC_WARN_ONLY = False`;
     `main.py`, `run_training` and `run_ablation_suite` call
     `set_seed(seed, deterministic=True, warn_only=…)`, i.e.
     `torch.use_deterministic_algorithms(True)`: an op without a
     deterministic implementation **raises**. `--deterministic-warn-only`
     (and `deterministic_warn_only=` on both functions) switches to
     `warn_only=True`;
  2. `set_seed` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` first, only if unset,
     before any cuBLAS handle exists; it always sets
     `cudnn.deterministic` / `benchmark=False` and the deterministic-algorithms
     flag explicitly (both ways), so modes do not leak between calls;
  3. `training_config` records `deterministic` and `deterministic_warn_only`
     (compared on resume, visible in every checkpoint and result);
     `cublas_workspace_config` is recorded for information.
  - Evaluation scripts (`input_ablation`, `baselines`) keep
    `set_seed(0)` with the default `deterministic=False`.
- **Test evidence:** `tests/test_determinism_resume.py` (CPU): strict and
  warn-only modes set the torch flags; `CUBLAS_WORKSPACE_CONFIG` is set when
  unset and kept when set; two same-seed 2-epoch runs give identical
  histories, bit-identical weights and identical per-example validation
  records; the mode is recorded, and resuming with
  `deterministic_warn_only=True` raises.
- **Step 6 notebook (T4) must:** run two same-seed 1-epoch smoke runs and
  diff their per-example validation predictions; print any operation that
  raises under strict deterministic mode (then decide whether to rerun with
  `--deterministic-warn-only`, which is recorded and compared). The
  PennyLane `lightning.gpu` device is outside torch's determinism controls
  and may have its own nondeterminism [ASSUMPTION]; that run checks it.
- **Prepared 2026-10-28 (not yet run on a T4):** `scripts/t4_checks.py`
  `determinism` (two same-seed 1-epoch runs in separate folders; compares
  histories, best-checkpoint parameters and per-example validation
  predictions and scores; exits 2 and records the error if strict mode
  raises, no retry) and `quantum` (the circuit twice on `lightning.gpu`,
  outputs and gradients bit for bit, plus the largest difference from
  `lightning.qubit`; exits 3 if the device cannot be built). Notebook cells
  5–6. `tests/test_t4_checks.py` (10 tests, CPU); 9 mutations each fail ≥ 1.
- **CUDA 13 fallback (2026-10-28):** Colab now runs CUDA 13; the
  pennylane-lightning-gpu 0.42.0 build targets CUDA 12. `_build_device` runs
  one gate on `lightning.gpu` before using it (`_probe`), so a device that
  constructs but cannot execute also falls back to `lightning.qubit` under
  `QUANTUM_DEVICE="auto"` (explicit `"lightning.gpu"` still raises). The
  manifest records `quantum_device` and the new `quantum_device_fallback`
  (the reason). The notebook's GPU-simulator install no longer stops the
  notebook when it fails. `tests/test_quantum_device_fallback.py` (5 tests);
  6 mutations each fail ≥ 1. The CPU forward pass is unchanged (no probe on
  `lightning.qubit`; the full suite, including the bit-identical forward
  tests, passes).

### E11 — Environment not pinned; resolver conflicts in the v46 run
- **Category:** E · **Severity:** Medium · **Status:** **fixed** in `4c4a766`; torch pinned 2026-10-28 to `2.11.0+cu130` (Colab T4, CUDA 13.0)
- **Evidence:** cell 8: `huggingface-hub 0.33.4` incompatible with
  `diffusers 0.39.0` / `gradio 6.20.0`; cell 10: PennyLane warns about JAX
  0.7.2 > 0.6.2; `requirements.txt:1` leaves torch unpinned.
- **Proposed fix:** pin torch; record `pip freeze` in the manifest (E6); install
  reported runs into a clean environment.
- **Verify:** identical `pip freeze` hash across installs; no resolver errors in the log.
- **Also verified (2026-10-19):** the executed v46 notebook never printed
  its torch or CUDA version (grep), so there is no recorded version to pin.
- **Fix (2026-10-19, decisions approved 2026-10-19):** new
  `training/environment.py`:
  1. `check_pins` runs first in `run_training` and `run_ablation_suite`:
     every **runtime** package (torch, transformers, accelerate, datasets,
     huggingface_hub, pennylane, pennylane-lightning) that has an `==` pin
     in `requirements.txt` must be installed at that version, else
     `EnvironmentMismatchError` before any loader, model or file is
     created. The plotting/analysis pins (scikit-learn, matplotlib,
     seaborn, umap-learn, plotly, networkx) are not imported by training
     and are recorded only. `--ignore-pin-mismatch` (and
     `ignore_pin_mismatch=`) overrides; the result (`pin_check`:
     mismatches, overridden) goes into the provenance;
  2. torch is still unpinned (decision: pin from the Step 6 T4 run;
     `requirements.txt` says so), so it is recorded but not checked
     against `requirements.txt`;
  3. provenance `environment` = torch, CUDA, pennylane,
     pennylane-lightning versions, **compared on resume**;
     `environment_info` = python, pennylane-lightning-gpu, transformers,
     accelerate, datasets, huggingface_hub, numpy, and sha256 of the full
     installed-package list, **recorded only**; the old informational
     `torch_version` field is replaced by `environment.torch`;
  4. each run writes `{run}_env.txt` (all installed `name==version`,
     from `importlib.metadata`) next to its checkpoint;
  5. `describe(provenance)` (versions + pin-check state, `pins OVERRIDDEN …`
     when overridden) is printed at the start and end of `--mode train`,
     per grid arm, in the grid summary, stored in each grid result entry
     (`environment`), and `history["provenance"]` is saved in every
     history and returned result.
- **Step 6 notebook (2026-10-28):** cells 2–3 print torch, `torch.version.cuda`,
  the GPU and the full `pip freeze` before installing; uninstall diffusers,
  gradio, jax, jaxlib; install `requirements.txt` with torch held at Colab's
  version by a constraint (never reinstalled; a changed torch stops the
  notebook); run `pip check`; print the pip-freeze sha256 and
  `packages_sha256`. The torch pin is committed from that output.
- **Torch pin (2026-10-28):** Colab cell 2 reported torch `2.11.0+cu130`,
  `torch.version.cuda` 13.0, Tesla T4; `requirements.txt` pins
  `torch==2.11.0+cu130`. `matches_pin`: a CUDA build must match exactly; a
  CPU-only build (`torch.version.cuda is None`) passes on the public version,
  so the CPU test suite runs on torch 2.11.0 (283 passed, 1 skipped).
  `tests/test_environment.py` +9 cases.
- **Colab pip conflicts (2026-10-28) [DERIVED]:** simulated from the cell-2
  freeze and PyPI metadata (Linux markers): cell 3's install downgrades
  transformers 5.20.2→4.52.4, huggingface_hub 1.33.0→0.33.4, accelerate
  1.15.0→1.7.0, datasets 4.8.5→3.6.0, tokenizers 0.23.3→0.21.4, fsspec,
  dill, multiprocess, scikit-learn, networkx, plotly, umap-learn, matplotlib.
  Remaining conflicts, all unused by qAIR (not in the dependency closure of
  its packages): gcsfs (fsspec), gradio-client (hub ≥ 1.16), then bigframes,
  hf-gradio; dopamine-rl needs flax. ipython lacks `jedi` before anything is
  installed (Colab kernel, kept). sentence-transformers 5.7.0 accepts
  `transformers>=4.41,<6` and `huggingface-hub>=0.23`; qAIR does not import it
  (the encoder uses `transformers.AutoModel`), so it is not pinned. Cell 3
  now uninstalls every unused package `pip check` complains about, keeps
  and documents Colab kernel packages, and stops if a package qAIR needs
  complains. To confirm with the real cell 3 output.
- **Test evidence:** `tests/test_environment.py` 7 passed: pin parsing
  (extras, comments, bare names); the repo leaves torch unpinned; matching
  pins pass; a runtime mismatch raises and the override records it, while
  a mismatched non-runtime pin is ignored; compared/recorded split;
  `run_training` with a mismatched pin raises before creating the
  checkpoint directory; with the override, the checkpoint, the history,
  the env file, the printed log and the grid entry and summary all carry
  it. `tests/test_checkpoint_provenance.py`: a change of torch, CUDA,
  pennylane or pennylane-lightning refuses a resume; python, package hash
  and pin-check changes do not. Full suite: 85 passed, 1 skipped.
  Mutations (never raise; environment not compared; all pins checked;
  grid summary not printed; pin check compared) each fail ≥ 1 test.
- **Step 6 notebook (T4) must, before training:** print torch version,
  `torch.version.cuda` and the full `pip freeze`; run `pip check`;
  uninstall the unused conflicting packages (diffusers, gradio, jax);
  never force-reinstall torch. Then pin `torch==<that version>` in
  `requirements.txt`, which turns on its startup check.

### D5 — Attack hypotheses do not reliably attack
- **Category:** D · **Severity:** High · **Status:** open
- **Evidence:** cell 43: the attack on the correct option "Carbon dioxide" reads
  "Plants absorb carbon dioxide (CO2) through their stomata … This absorption
  process is crucial…" (it supports). Polarity is assigned by mode, not content
  (`models/generator.py:347`). The rate is unmeasured.
- **Proposed fix:** audit 100 attacks and report the non-compliance rate η with a
  CI. Filter or regenerate only with approval.
- **Verify:** the audit; the dossier derives that the attack channel's expected
  signal scales with (1 − 2η).
- **Full-scale finding (2026-10-28):** attack disagreement 0.1804 / 0.1719,
  below chance (z −7.6 / −4.3, train / validation): anti-informative.
  First-order implied η ≈ 0.91 ± 0.07 (train), 0.99 ± 0.16 (validation);
  details and assumptions in dossier D5. The audit is still needed.

### C6 — The collapse over hypotheses is near-uniform
- **Category:** C · **Severity:** High · **Status:** open · **needs supervisor discussion**
- **Evidence:** collapse entropy 2.0704 → 2.0763 vs log 8 = 2.0794, i.e. KL to
  uniform 0.0090 → 0.0031 nats; peak 0.149 → 0.141 (cell 27). The implied
  temperature is ≈1.0, its initial value (verification script check 1).
- **Proposed fix:** none before A7 and full-data runs; then discuss a
  temperature or selectivity change.

### C7 — Hypotheses' option-energy profiles converge
- **Category:** C · **Severity:** High · **Status:** open · **needs supervisor discussion**
- **Evidence:** energy-row pairwise cosine 0.769 → 0.838 (main run), 0.9706 (A3),
  while hypothesis cosine stays 0.87–0.88 (cells 27, 39). At near-uniform
  amplitudes the gradient on option energies is identical within a channel
  (verification script check 13).
- **Proposed fix:** measure within- vs across-channel cosine; no change before C6/D1.

**Checked, not added as separate issues:** "learned scalars never move" is
explained by the Adam step budget (B8; check 5), with `hypothesis_temp` as the
one true zero-gradient case (A1). "Sanity check passes on a saturated metric" is
part of C2.

---

## 10. Issues found during the Phase 1 fixes

### A9 — Any cache-load error is treated as a truncated file: the cache is moved aside and rebuilt
- **Category:** A · **Severity:** High · **Status:** **fixed** in `0313a15` (found 2026-10-18 while fixing E3)
- **Evidence:** `training/dataset.py` (load block, `except Exception` after
  `torch.load(cache_path, map_location="cpu")`) renames the file to
  `.corrupt` and rebuilds from scratch for *every* exception, not only a
  truncated archive. Under torch ≥ 2.6, `torch.load` defaults to
  `weights_only=True`, so any header value that is not an allow-listed type
  raises `UnpicklingError` and triggers the rebuild. Observed in the E3 test
  run: a `TorchVersion` object in the header made a valid cache be moved to
  `.corrupt` and regenerated. (E3 now stores plain strings, so E3's own header
  is safe; the handler remains dangerous for any other load error, e.g. a
  torch version change.)
- **Proposed fix:** treat only a genuine truncation (zip central-directory
  error) as recoverable; re-raise every other error with the path; never
  rename a file silently. Combined with E7, a build must never start
  implicitly.
- **Verify:** unit tests: a cache with a non-allow-listed header value raises
  with the path and is left in place; a truncated file is still detected.
- **Fix (2026-10-18):**
  - the load handler never renames, deletes or rebuilds; it raises
    `CacheLoadError` with the path, the exception and the cache header;
  - `safe_load_cache`: `torch.load(..., weights_only=True)` inside
    `torch.serialization.safe_globals([TorchVersion])` (a `str` subclass,
    safe to allow-list); never `weights_only=False`;
  - `_save` also writes a plain-JSON sidecar `<cache>.header.json` (version,
    arc_config, raw_index, complete, n_samples, gen_seed, generator and
    revision, device, device name, torch version, hypotheses_sha256), so the
    header is readable when the cache itself is not. Old caches without a
    sidecar say so in the error.
- **Test evidence:** `tests/test_cache_load_errors.py` 5 passed: a truncated
  cache and a garbage file both raise and stay **byte-identical** (sha256
  before = after; no `.corrupt` file); the truncated case reports version 46,
  gen_seed 0 and device cpu from the sidecar; a `TorchVersion` header loads;
  a pickled object whose `__reduce__` would create a file is refused and
  **the file is not created**; the sidecar matches the cache. Full suite:
  18 passed, 1 skipped. Mutations: restoring the rename handler fails 3 tests;
  `weights_only=False` fails the unsafe-pickle test.

### A10 — An early-stopped run trains again when it is re-run
- **Category:** A · **Severity:** Medium · **Status:** **fixed** with A4 in `4083ecf` (found 2026-10-19 while proposing A4)
- **Evidence:** before the fix, `training/train.py` saved the latest
  checkpoint at the stop epoch $e_s < E-1$ and then `break`; nothing recorded
  that the run had stopped. Re-running the same run passed the A3 check,
  resumed at $e_s+1$ with `epochs_no_improve = 0`, and trained up to
  `patience` more epochs. The grid's `[SKIP]` tested only
  `start_epoch >= epochs` (`training/ablations.py`, before A4).
- **Root cause:** early-stopping state was not part of the checkpoint.
- **Fix:** checkpoints store `stopped_early`; `is_finished(state, epochs)`
  = `stopped_early or start_epoch >= epochs`; `Trainer.train_from` skips a
  finished run and returns its saved history, in both `runner.py` and the grid.
- **Test evidence:** `tests/test_resume_state.py`: a run that stopped
  early after 3 of 6 epochs, re-run, makes 0 validation calls, prints
  `[SKIP] … already finished (stopped early)`, returns the same history and
  leaves the checkpoint at epoch 2; the same for a grid arm (A1_baseline,
  `best_val_acc` 0.5).

### A11 — A resumed run is not RNG-continuous with an uninterrupted one
- **Category:** E · **Severity:** Medium · **Status:** **fixed** with E10 in `bbea237` (found 2026-10-19 during A4). **Decision 2026-10-19:** fixed together with E10, keeps its own ID; scope: python, numpy, torch CPU/CUDA RNG and the DataLoader generator/epoch, so a resumed run equals an uninterrupted one.
- **Evidence:** 4-epoch CPU smoke run (fake cache, 16 train / 8 val, seed 42,
  `deterministic=False`, scripted validation), interrupted after epoch 2 and
  resumed: epoch losses uninterrupted `[1.8829, 1.3686, 1.0157, 0.8861]`,
  resumed `[1.8829, 1.3686, 0.9016, 0.8123]` (probe outside the repo,
  2026-10-19).
- **Root cause [from the code]:** no RNG state is checkpointed.
  `build_loaders` creates the train loader with a fresh
  `seeded_generator(seed)` on every start, so the first epoch after a
  resume replays epoch 1's batch order; the global torch RNG (dropout,
  `collate_fn`'s per-batch option `randperm`) restarts from `set_seed`.
- **Proposed fix (not implemented):** save the train loader generator state
  and `torch.get_rng_state()` (plus `torch.cuda.get_rng_state_all()` on
  CUDA) in the checkpoint at the end of each epoch and restore them on resume;
  all are tensors, so `weights_only=True` still loads them.
- **Verify:** the probe above gives identical losses on CPU with
  `deterministic=True`.
- **Fix (2026-10-19):**
  1. `training.seed.capture_rng_state(loader_generator)` /
     `restore_rng_state`: python `random`, numpy (state array stored as an
     int64 tensor), torch CPU, CUDA (when available) and the train loader's
     generator; all reload under `weights_only=True`;
  2. every checkpoint stores `rng`, captured in `save_checkpoint` after
     validation (which also draws from the global torch RNG);
     `load_or_resume` restores it; a checkpoint without `rng` is refused
     (format stays 2, approved);
  3. the grid re-seeds the shared train loader's generator per arm, so an
     arm's batch order no longer depends on which arms ran or were skipped
     before it in the same session (found while implementing: the loaders
     are built once per seed and shared by every arm).
- **Test evidence:** `tests/test_determinism_resume.py`: the A4 probe
  (4 epochs, interrupted at the third validation, resumed) now gives a
  history **equal** to the uninterrupted run in every metric, and
  bit-identical final weights (real validation, not scripted); RNG state
  saved and reloaded with `weights_only=True` reproduces python, numpy,
  torch and generator draws; a checkpoint without `rng` is refused;
  `A2_validator` after `A1_baseline` has the same history as alone.
  Mutations (restore skipped; loader state not saved; torch state not
  restored; numpy not restored; no per-arm re-seed; warn-only not recorded;
  `DETERMINISTIC = False`) each fail ≥ 1 test. Full suite: 74 passed,
  1 skipped.

### A12 — Shuffling misaligns hypotheses when K is not a multiple of N
- **Category:** A · **Severity:** Low (latent) · **Status:** **fixed 2026-10-27** in `59f4974` (found 2026-10-19 while writing the DoD §2.6 test; fix approved 2026-10-27)
- **Evidence:** `training/dataset.py` `collate_fn`, `shuffle_options=True`:
  the options and LLM scores are always permuted, but when `k % n != 0` the
  hypotheses, `polarity`, `is_fallback` and `hyp_option` are left in their
  original order. `align` is then built from the unpermuted `hyp_option`, so
  each hypothesis points at whatever option now sits at its old index. Probe
  (outside the repo, `torch.manual_seed(3)`, N = 4, K = 6): option codes after
  shuffling `[2, 3, 1, 0]`; hypotheses about options `[0, 1, 2, 3, 0, 1]` are
  aligned to options `[2, 3, 1, 0, 2, 3]`. The code comment says such a sample
  is "left alone rather than silently misaligned"; it is misaligned.
- **Reachability [VERIFIED: `models/generator.py:384-446`,
  `training/dataset.py:318-340`]:** caches built by this code always have
  K = len(modes) × N in mode-major order, so `k % n == 0` and the path is not
  reached. Only a hand-built or legacy sample can reach it.
- **Fix:** `collate_fn` raises `ValueError` when asked to shuffle a sample
  with `k % n != 0` or whose `hyp_option` is not mode-major. Unshuffled
  batches still accept such samples (their alignment is correct). The check
  runs before `randperm` and draws no random numbers, so valid batches
  shuffle exactly as before.
- **Test evidence:** `tests/test_option_shuffle_alignment.py` (+5 tests):
  shuffling a K = 6, N = 4 sample and an option-major sample raises;
  unshuffled, both stay aligned; a seeded shuffle of valid samples gives the
  same permutations as `torch.randperm` from the same seed. Against the
  pre-fix `dataset.py` 2 tests fail; checking only `k % n` fails 1. Full
  suite: 234 passed, 1 skipped in 137.14 s.

### Regression tests DoD §2.5 and §2.6 (2026-10-19, `0997a99`)
- **§2.5 no label leakage:** `tests/test_no_label_leakage.py`. For all 14
  arms, in eval and in train mode (same dropout seed), every model output is
  bit-identical with `y` omitted, correct or wrong, except
  `validator.relevance_target`, which equals the one-hot of `y` broadcast over
  K. Mutations: an energy offset when `y` is given, `relevance += 1e-3 ×
  target` and a target of 0.5 each fail ≥ 1 test.
- **§2.6 option shuffling:** `tests/test_option_shuffle_alignment.py`. A
  hand-built ragged batch (3, 4, 5 options; every tensor carries an option /
  mode code), 50 seeded shuffles: each hypothesis stays aligned to its option,
  with its polarity and fallback flag; each LLM score stays with its option;
  the label points at the same option; masks and padding are correct. On 16
  cached samples, shuffled vs unshuffled, the label's option and the
  hypotheses about it are the same. Mutations: not permuting H, not remapping
  `y`, not permuting the LLM scores and not permuting `is_fallback` each fail
  ≥ 1 test. Not permuting `polarity` passes, and is an equivalent mutant:
  polarity is constant within a mode block and the shuffle only reorders rows
  within blocks.
- **Full suite:** 199 passed, 1 skipped in 122.09 s.

### Regression test DoD §2.3 collapse correctness (2026-10-19, `f355965`)
- `tests/test_collapse_correctness.py`: `CoherentCollapse` on a hand-built
  batch (B = 2, K = 4, N = 3; one padded option, one padded hypothesis)
  against values computed independently with `math`. Valid-option
  probabilities sum to 1 (±1e-5); padded options and hypotheses get ≤ 1e-12;
  `"classical"` = Σ_k r_k² t_kn², `"zero"` = (Σ_k r_k t_kn)², `"learned"` =
  |Σ_k r_k t_kn e^{iφ_kn}|² (±1e-6 on the distribution); support/attack phases
  0/π give (S_n − T_n)² (C1 documented, not changed); hypothesis weights are
  r_k² normalized over valid hypotheses. The full model's scores are a
  distribution over valid options for all 14 arms (padded mass e^−30).
- Mutations: wrong classical, in-phase or imaginary part, option temperature
  offset, padding floor −20 and no hypothesis mask on r each fail ≥ 1 test.
  Removing any one padding mask alone passes because each is backed by a
  second mask (`invalid` in `log_t`, the `weights` fill, the mask on r):
  equivalent mutants. Removing both option masks fails 7 tests.
- Full suite: 221 passed, 1 skipped in 122.99 s.

### DoD §2.2 and §2.7 per-epoch logging (2026-10-27, `402808e`)
- **Change:** `training/diagnostics.py` (new), called from
  `training/train.py` every epoch; values go into the history (saved in every
  checkpoint and the run's history file) and three lines are printed.
  - **§2.2:** per top-level module (`in_proj`, `reasoner`, `quantum`,
    `validator`, `selector`, `fusion`, `collapse`, `interference`,
    `phase_head`; parameters on the model itself under `model`), the mean and
    max per-step L2 gradient norm, after unscaling and before clipping; the
    total norm, the fraction of steps clipped at `GRAD_CLIP = 2.0` and the
    number of AMP-skipped steps (excluded from the means). A missing gradient
    counts as 0.
  - **§2.7:** every 0-d parameter (phases, temperatures, gates; 13 in F0) and
    its drift from the run's start; per top-level module the relative weight
    change ‖W − W0‖ / ‖W0‖.
  - W0 and the initial scalars are stored in the history once, at the run's
    start, so a resume measures from the true start. W0 adds one copy of the
    weights to each checkpoint and history file (≈ 4.2 MB for F0's 1,055,707
    parameters in fp32 [DERIVED]). Grid result entries leave W0 out, so
    `ablation_results.pt` does not grow by one copy per run.
  - No checkpoint format change: any `.py` change already refuses resume
    (A3 code fingerprint), and evaluation loading is unaffected.
- **Test evidence:** `tests/test_training_diagnostics.py` (new, 8 tests):
  hand-set gradients give the expected per-module mean/max, clipped fraction
  and skipped count; drift and relative change on a hand-built model; W0 kept
  on a second `start_of_run`; on a real 2-epoch run every key is logged every
  epoch, the logged scalars and drift equal the checkpoint's values, the
  `in_proj` change equals ‖W − W0‖/‖W0‖ recomputed from the checkpoint, the
  three lines print once per epoch, and grid entries omit W0 while the run's
  history file keeps it. Per step, √Σ module² equals the unclipped total
  (rel 1e-4), with clipping active. `tests/test_determinism_resume.py` now
  compares histories including the new keys and W0: resumed = uninterrupted.
  Ten mutations (measure after clip, skip test inverted, no per-epoch reset,
  line not printed, W0 overwritten on resume, drift from 0, absolute instead
  of relative change, max = last, clipping not counted, W0 kept in the grid)
  each fail ≥ 1 test. Full suite: 229 passed, 1 skipped in 141.59 s.

- **Follow-up 2026-10-28:** the T4 smoke run printed a relative change of
  7.07e+08 for one module whose W0 is (near) zero. The denominator is now
  `max(‖W0‖, NORM_FLOOR = 1e-3)`; such modules are marked `*` in the printed
  line (`floored_modules`). `tests/test_training_diagnostics.py` +1 test; 2
  mutations each fail it.

### E12 — A training step takes 69.9 s on the T4 (PennyLane per-row circuit execution)
- **Category:** E · **Severity:** Critical (blocks Step 7: 20 epochs ≈ 58.5 h per run) · **Status:** **closed 2026-10-29**: merged `b548095` (follow-up `2f30e39`); T4 measured at 1.514 s/step (was 69.9 s)
- **Evidence (T4, user, commit `44fc12a`):** `smoke_timing.py` measured
  70 s per train step (batch 16) and 0.58 s per question for generation,
  i.e. one epoch (141 steps) ≈ 176 min [DERIVED there].
- **Profile (CPU, Apple M4, torch 2.11.0, `lightning.qubit`, B = 16, N = 4,
  K = 8; scratch script, two identical runs) [VERIFIED]:**

  | Part | Seconds |
  |---|---|
  | full training step (forward + backward) | 0.86 (0.77 + 0.09) |
  | same step, circuit replaced by a stand-in of the same shape | 0.019 |
  | circuit alone, 128 rows, forward / backward | 0.32 / 0.02 |
  | eval forward, one batch, no grad | 0.68 |

  - The circuit is ≈ 98 % of the step. It runs **twice per forward**: once
    after the first reasoner pass and again in the validator-feedback pass
    (`models/full_model.py` `_run_quantum`, E8 note), each time on B·K = 128
    rows.
  - `qml.Tracker`: 512 executions and 512 derivatives per step. `TorchLayer`
    splits the batch into one tape per row; `diff_method="best"` resolves to
    lightning's **adjoint** method, computed together with the forward pass
    (`execute_and_derivative`), which is why backward is cheap and forward
    is not. The cost is per-row tape overhead, linear in rows (1 row 4 ms,
    8 rows 19 ms, 32 rows 73 ms, 128 rows 0.32 s).
  - Not a cost: the rest of the model (19 ms), the reasoner's 3 persistent
    steps, the per-module diagnostics.
- **Why 80× slower on the T4 than on this CPU — not measured; hypotheses:**
  (1) `lightning.gpu` pays a fixed per-tape cost (cuStateVec setup, host ↔
  device copies, synchronisation) on 512 tiny 6-qubit tapes per step; (2)
  the smoke run times only 2 steps, so one-time start-up (CUDA context,
  cuStateVec, first-call overheads) is counted in the per-step mean; (3)
  Colab's 2 vCPUs are slower than an M4 for the per-tape Python work. The
  next cell-6 run reports seconds per 128-row pass for `lightning.gpu` and
  `lightning.qubit` (`t4_checks.py quantum`, now with timings), which
  separates (1) from (2) and (3).
- **Proposed fix (needs approval: same math, not bit-identical):** a
  torch-native statevector of the same circuit (Hadamard, `AngleEmbedding`
  Y, `StronglyEntanglingLayers` with PennyLane's `Rot` and CNOT ranges,
  Pauli X/Y/Z expectations), batched over all rows, differentiated by
  autograd on the model's device; PennyLane kept behind a flag as the
  reference. Prototype (scratch, CPU) against `TorchLayer`: max |Δ output|
  1.5e-8, |Δ input grad| 1.2e-7, |Δ weight grad| 1.9e-6 in complex128 (the
  reference itself returns float32); 128 rows forward + backward in 8 ms
  (complex128) / 6.6 ms (complex64), against ≈ 340 ms for PennyLane.
- **Estimate after the fix [DERIVED on the M4 / HYPOTHESIS for the T4]:**
  CPU step ≈ 0.019 + 2 × 0.008 ≈ 0.035 s. On the T4, the circuit becomes a
  few hundred small kernels per step; estimated 0.05–0.15 s per step, i.e.
  141 × 20 steps ≈ 2.5–7 min of training per 20-epoch run plus validation,
  to be measured by the smoke run.
- **CPU vs T4 (recorded 2026-10-28):** 0.75 s/step on CPU (M4,
  `lightning.qubit`) vs 69.9 s/step on the T4 (`lightning.gpu`). Reviewer's
  reading: per-call GPU launch overhead of 512 per-row circuits, not compute
  [HYPOTHESIS until the cell-6 seconds-per-pass measurement].
- **Implementation (branch `e12-torch-statevector`):** `models/statevector.py`
  (`run_circuit`, `TorchStatevectorCircuit`); `config.QUANTUM_SIMULATOR =
  "torch"` (default) or `"pennylane"` (reference / fallback);
  `QuantumEvolutionLayer(quantum_simulator=…)`; `device_name =
  "torch.statevector"` in the manifest. Same parameter name (`weights`) and
  initial draw as `TorchLayer`, so same-seed models and checkpoints are
  interchangeable. Both engines are exact classical simulations of the same
  circuit; the model's mathematics is unchanged.
- **Tests on the branch:** `tests/test_torch_statevector.py` (float64 ≤ 1e-6
  and float32 ≤ 1e-5 on outputs and input / weight gradients, 3 seeds each;
  identical same-seed weights; full forward of all 16 arms within 1e-5);
  `tests/test_circuit_input_dependence.py` under both engines, with the
  X-rotation negative control; the determinism and resume suite and the v46
  bit-identical forward tests (`test_forward_unchanged`, with the v46
  checkpoints present) pass on the torch default; `t4_checks quantum`
  compares and times the torch engine against `lightning.qubit`. 8 mutations
  of the circuit maths each fail ≥ 1 test.
- **Follow-up after the merge (2026-10-28):** `quantum_simulator` is a
  provenance field compared on resume (and in the manifest), so a run never
  continues under the other engine; `tests/test_determinism_resume.py`: a run
  interrupted under `"pennylane"` refuses to resume under `"torch"` with
  exactly that one mismatch, and resumes under the same engine. Notebook cell
  7 prints a third verdict line for the torch engine (max difference from
  `lightning.qubit`, seconds per pass for all three).
- **Interim option without code change:** `QUANTUM_DEVICE="lightning.qubit"`
  on Colab. At the M4's 0.86 s/step and a 2–4× slower Colab CPU
  [ASSUMPTION], ≈ 2–4 s/step, ≈ 2.5–4.5 h per 20-epoch run.
- **T4 measurement (user, commit `2f30e39`, session
  `20261028_170721_2f30e39`, smoke cache 20 train / 10 validation)
  [VERIFIED by the user's cell 4–7 output]:**
  - train step **1.514 s** (was 69.9 s); [DERIVED there] one epoch ≈ 3.6 min,
    20 epochs ≈ 1.21 h. The smoke run times only 2 steps, so start-up is in
    the mean. The estimate above (0.05–0.15 s) was 10× too low. By the
    cell-6 timing below, the circuit is now ≲ 0.1 s of the step [DERIVED:
    2 passes × 0.024 s plus backward]; where the rest goes is not profiled
    [HYPOTHESIS: the rest of the model and the per-step diagnostics on
    Colab's 2 vCPUs].
  - determinism pair: IDENTICAL under the torch engine (strict mode).
  - torch statevector vs `lightning.qubit`: max difference 1.9e-6 (weight
    gradient; the reference is float32). Seconds per 128-row pass: torch
    0.024, `lightning.gpu` 41.1, so the 69.9 s step was `lightning.gpu`'s
    per-row cost (hypothesis (1) above).
  - end to end: the 1-epoch smoke run reproduces the PennyLane-engine run
    (train loss 1.6301 vs 1.6302, validation 0.40, identical diagnostics).
  - E10 follow-up: the fusion weight-change metric reads 0.707 with the
    norm floor (was 7.07e+08).
- **Closed:** Step 7 is no longer blocked; its run plan uses 1.514 s/step.

### E13 — The torch circuit engine fails on Apple-Silicon MPS (float64)
- **Category:** E · **Severity:** Low (local development only; Colab uses CUDA) · **Status:** open, found 2026-10-29
- **Evidence [VERIFIED: local dry run of the Step 7 notebook, torch 2.11.0, Apple M4]:**
  `python main.py --mode ablation …` on a Mac raises `TypeError: Cannot
  convert a MPS Tensor to float64 dtype` in `models/statevector.py`
  `run_circuit` (`inputs.double()`), because `config.resolve_device()`
  prefers `mps` and the engine computes in float64. The tests do not see it:
  they force `device = "cpu"`. CUDA supports float64, so the T4 is unaffected
  (smoke run above).
- **Also stale:** the `resolve_device` docstring still says the circuit always
  runs on `lightning.qubit` on the CPU; since E12 it runs on the model's device.
- **Options (not implemented; a choice of numerics on one device):** run the
  circuit on the CPU when the model is on `mps`, or compute in complex64 on
  `mps` only (float32 tolerances, already tested at ≤ 1e-5). Or skip `mps` in
  `resolve_device`. Workaround now: run locally with the CPU device.

### Note for every training run
**Always commit before starting a training run; the code fingerprint ignores untracked .py files.**
