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
6. **A5** `input_ablation` rebuilds the wrong architecture without a saved config.
7. **A1, A2** Parameters that never receive gradient. The fix approach (remove vs exclude from the optimizer) needs confirmation because of checkpoint compatibility (rule 9).
8. **A7** Validator relevance target has no hypothesis axis. **Blocked: supervisor approval.** A candidate fix exists in the v47 patch.
9. Regression tests from E8: circuit input-dependence (A0) and gradient coverage (DoD §2.1–2.2).

**Phase 2 — remaining engineering**
10. **E6** Run provenance manifest (commit, cache hash, versions, device, seed).
11. **E5** `ablation_results.pt` overwritten per invocation.
12. **E8** (rest) Timed 20-question smoke run on T4 to replace the cost assumptions; `tests/`.
13. **E9** Stale docs and names.
→ Then **build the ARC-Easy cache**. This is a >1 h T4 run, so it needs your go-ahead (CLAUDE.md).

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
- **Follow-up:** Commit this test as a regression test (Phase 1, item 9). Do not
  train from local `main`.

### A1 — `CoherentCollapse.hypothesis_temp` is a dead parameter
- **Category:** A · **Severity:** Medium · **Status:** open
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
- **Category:** A · **Severity:** Low–Medium · **Status:** open
- **Evidence:** `models/collapse.py:55-60,85`: the `confidence` head is only
  returned (`models/full_model.py:470`) and never used in `training/losses.py`.
  `models/full_model.py:203,384-387`: `phase_head` is unused when
  `phase_source="circuit"` and a bottleneck exists (the default).
  `llm_prior_weight` is unused when `use_llm_prior=False` (expected in the
  ablation arms).
- **Proposed fix / verify:** As for A1. Record trainable-parameter counts per arm.

### A3 — Single-run name lacks the seed; resume ignores the config
- **Category:** A · **Severity:** High · **Status:** open
- **Evidence:** `training/runner.py:125` sets `name="qair_v45.1"`.
  `training/checkpoint.py:15,25-46` loads `{name}_latest.pt` with no seed or
  config check. `--mode train --seed 43` after a seed-42 run therefore resumes
  seed 42. `training/ablations.py:227` does include the seed.
- **Proposed fix:** Put seed, arm and dataset in the run name; store the config in
  the checkpoint; refuse to resume on a mismatch.
- **Verify:** Two 1-epoch smoke runs (≤20 examples, separate cache dir) with
  different seeds → two checkpoint files, different initial loss.

### A4 — Resumed training history discarded and overwritten
- **Category:** A · **Severity:** Medium · **Status:** open
- **Evidence:** `training/runner.py:175` discards the resumed history
  (`start_epoch, best_acc, _ = load_or_resume(…)`). `training/train.py:132-145`
  starts a fresh `history`, and `:203` overwrites the file every epoch.
  `training/ablations.py:265-278` uses `resumed_history` only for finished runs.
  `epochs_no_improve` resets on resume (`training/train.py:147`). Observed:
  `A3_persistent_s42_latest.pt` is at epoch 7, but its history file holds
  **1** epoch; `qair_v45_history.pt` holds 2.
- **Proposed fix / verify:** Merge histories and persist `epochs_no_improve`.
  An interrupted-then-resumed smoke run should have one history entry per epoch.

### A5 — `input_ablation` can rebuild the wrong architecture
- **Category:** A · **Severity:** High · **Status:** open
- **Evidence:** `evaluation/input_ablation.py:252-258` falls back to a default
  config, and `:232` defaults `use_llm_prior=True`. Ablation runs never write
  `_config.pt` (it is written only at `training/runner.py:188-206`); `ckpt/` has
  config files only for `qair_v45` and `qair_v46`. All arms share one parameter
  set, so `load_state_dict` succeeds silently.
- **Proposed fix / verify:** Store the config in every checkpoint and refuse to
  run without it. The "intact" accuracy should equal the logged validation
  accuracy.

### A6 — Cache edge cases silently change state or data
- **Category:** A · **Severity:** Medium · **Status:** open
- **Evidence:** (1) `training/dataset.py:154-172` + `:437-439`: a complete cache
  smaller than `max_samples` is re-saved as `complete=False`.
  (2) `:203-209`: a missing split is silently replaced by `validation`.
  (3) `:256-260`: a scoring failure stores zeros for `llm_logprob`;
  `:235-254`: generation failure → template fallbacks, with no per-sample flag
  for the scoring failure.
- **Proposed fix / verify:** Treat `complete` as final, raise on a missing split,
  add a per-sample `score_failed` flag. Unit tests with a fake generator.

### A7 — v46 validator relevance target has no hypothesis axis *(new)*
- **Category:** A · **Severity:** High · **Status:** open · **fix needs supervisor approval**
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
- **Severity:** High · **Status:** open
- **Evidence:** `training/train.py:325-329` saves `best` on the max validation
  accuracy. The reported 0.50 is the max of
  `[0.35, 0.35, 0.40, 0.45, 0.40, 0.40, 0.45, 0.50]` (`qair_v46_history.pt`).
  `training/runner.py:87-95` skips test when its cache is missing; the v46
  history has no `test` key.

### B4 — No "LLM letter pick" baseline (mandatory)
- **Severity:** High · **Status:** open
- **Evidence:** `models/generator.py:357-451` is cloze-style option likelihood
  only; `evaluation/baselines.py` has no lettered-options MCQ prompt.

### B5 — No majority-vote-over-hypotheses baseline (mandatory)
- **Severity:** High · **Status:** open
- **Evidence:** `evaluation/baselines.py` modes: `llm`, `asymmetry`, `probe`,
  `direct`.

### B6 — Pooling records across seeds makes McNemar/bootstrap anti-conservative
- **Severity:** High · **Status:** open
- **Evidence:** `evaluation/stats.py:95-100` pools `test_records` across seeds;
  `:127` runs McNemar on the pooled list; `:120` bootstraps (item, seed) cells.
  Holm is only printed (`:138-140`).

### B7 — Probe "data ceiling" selects its best epoch on validation
- **Severity:** Medium · **Status:** open · **Evidence:** `evaluation/baselines.py:333-348`.

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
- **Severity:** Medium · **Status:** open · **Evidence:** `training/ablations.py:212,318`.

### E6 — Run provenance not recorded
- **Severity:** High · **Status:** open
- **Evidence:** `training/runner.py:189-205` saves no commit, cache hash, dataset
  revision or package versions; ablation runs save no config (A5).
  `requirements.txt:1` leaves `torch` unpinned; the circuit device is `"auto"`
  (`config.py:141`).

### E7 — A capped cache can silently restart hours of LLM generation
- **Severity:** Medium→High for the cache build · **Status:** open · **Phase 1, item 2 (blocks the ARC-Easy cache)**
- **Evidence:** `training/dataset.py:157-163` (documented in the code) and
  `evaluation/baselines.py:304-309`: calling with `max_samples=None` against a
  capped (`complete=False`) cache resumes generation. Any caller other than
  `build_cache.py` (training, baselines, input ablation, notebooks) can trigger it.
- **Proposed fix:** An explicit `allow_build` flag, False everywhere except
  `scripts/build_cache.py`; otherwise raise with the cap the cache was built with.
- **Verify:** Loading a capped smoke cache with `max_samples=None` raises and
  never loads the LLM.

### E8 — No v45+ timing data; no tests
- **Severity:** Medium · **Status:** open
- **Evidence:** Only the v44 rate is recorded (`1.24s/it`, batch 16, K=4,
  `qAIR_v44.ipynb` cell 21); v46 tqdm bars were saved at 0%. No `tests/`.
  `models/full_model.py:317,348` runs the reasoner and circuit twice per forward
  pass when validator feedback is on.

### E9 — Stale docs and names
- **Severity:** Low · **Status:** open
- **Evidence:** `evaluation/stats.py:4-7` (869 = merged-ARC validation);
  `training/runner.py:125,136` ("v45.1"); `scripts/build_cache.py:149`
  (`qair_v45_best.pt`); `docs/history/CLAUDE_v46.md` says "Current version: v45".

## D. Data / input problems

### D1 — 0.5B hypotheses carry no signal beyond chance
- **Severity:** Critical (for the research question; it does not block "working") · **Status:** open
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
- **Category:** A · **Severity:** High · **Status:** open
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
- **Category:** E · **Severity:** Medium · **Status:** open
- **Evidence:** `[SEED] Set to 42 (deterministic=False)` (cells 10, 39);
  `training/seed.py:8`; no caller passes `deterministic=True` (grep).
- **Proposed fix:** `deterministic=True` for reported runs; record it in the manifest.
- **Verify:** two same-seed T4 smoke runs give identical predictions.

### E11 — Environment not pinned; resolver conflicts in the v46 run
- **Category:** E · **Severity:** Medium · **Status:** open
- **Evidence:** cell 8: `huggingface-hub 0.33.4` incompatible with
  `diffusers 0.39.0` / `gradio 6.20.0`; cell 10: PennyLane warns about JAX
  0.7.2 > 0.6.2; `requirements.txt:1` leaves torch unpinned.
- **Proposed fix:** pin torch; record `pip freeze` in the manifest (E6); install
  reported runs into a clean environment.
- **Verify:** identical `pip freeze` hash across installs; no resolver errors in the log.

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
- **Category:** A · **Severity:** High · **Status:** open (found 2026-10-18 while fixing E3)
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

