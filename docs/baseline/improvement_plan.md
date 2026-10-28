# qAIR improvement plan — Tiers 0–4 (PROPOSAL, for supervisor review)

> **Decision status (2026-10-19).** The supervisor has not reviewed this plan
> yet. Work proceeds under the **provisional decisions** in
> [`supervisor_decisions.md`](supervisor_decisions.md) (each "provisional — pending supervisor
> review"). The rule for all of them: no change replaces v46 behaviour by
> default; every new behaviour is a flagged, default-off arm, and every
> result table reports the v46 baseline alongside. "Supervisor approval:
> Yes" below means the item still needs the supervisor's review; until then
> the provisional decision named next to it applies.

Branch `baseline` (code byte-identical to v46 `9552498`). Written 2026-10-18.
**Planning document only: no code is changed and no run is made by it.** It
builds on `docs/baseline/issues.md` (issue IDs, fix order) and
`docs/baseline/issue_dossier.md` (root causes and mathematics). Where this plan
repeats a mechanism, the dossier section is named so the derivation can be
checked there.

**Labels** (same as the dossier). [VERIFIED: …] = a file:line or a command run
while writing this plan. [DERIVED] = arithmetic shown here; the script that
produced every derived number is in Appendix A. [OBSERVED: v46 run] = saved
output of `notebooks/qAIR_v46.ipynb`. [HYPOTHESIS] = plausible, untested.
[ASSUMPTION] = taken as given until measured. **Every T4 cost in this plan is
an [ASSUMPTION]** until the timed smoke run of E8 replaces it.

---

## 0. Ground rules for every tier

1. **One variable per comparison** (CLAUDE.md rule 4). Each change below is
   compared with a *reference run* that differs from it in that one change
   only.
2. **The reference is the Tier 0 baseline.** If a Tier 1 cache is adopted, the
   reference becomes "Tier 0 code and recipe on the adopted cache" (written
   **Tier 0′**). That run is the control for every later tier, so each later
   comparison still changes one thing.
3. **Same split, same seeds.** Full ARC-Easy (train 2251 / validation 570 /
   test 2376 [VERIFIED: `config.py:58`]), seeds 42, 43, 44 (from
   `config.py:286` `SEEDS = (42, 43, 44, 45, 46)`).
4. **Validation selects, test reports once.** Every gate, prompt choice,
   generator choice, threshold and epoch count is decided on train or
   validation items. Test is evaluated once per (arm, seed) (B3; rule 5).
5. **Statistics per DoD §4.** Per-seed exact McNemar, plus an item-level
   paired bootstrap of the seed-averaged difference, with Holm across the
   pre-registered contrasts (B6).
6. **Predictions are written before the run.** Each "falsifiable prediction"
   below is the pre-registration. If a result contradicts it, the plan records
   that; it does not re-tune and re-run on the same items.
7. **Arms used throughout** [VERIFIED: `training/ablations.py:34-44,109-145`]:
   `A3_persistent` (quantum + validator, no LLM prior), `F0_full_with_llm_prior`
   (A3 + LLM prior; the default single-run config, `config.py`
   `USE_LLM_PRIOR = True`), `P0_classical_collapse` and `P1_phase_zero`
   (controls for A3, no prior).

### 0.1 Numbers used throughout (all verified or derived; Appendix A)

| Quantity | Value | Source |
|---|---|---|
| Steps per epoch on full ARC-Easy train | ⌈2251/16⌉ = **141** (loader has no `drop_last`) | [VERIFIED: `config.py:268` `BATCH_SIZE = 16`; `training/runner.py:61-67`] |
| LR groups | `quantum.*` and `validator.*` at 2e-4; everything else at 5e-4; AdamW, wd 0.05, β = (0.9, 0.999) | [VERIFIED: `training/train.py:47-59`; `config.py:272-277`] |
| Schedule | `CosineAnnealingLR(T_max = epochs, eta_min = 1e-6)`, stepped once per epoch | [VERIFIED: `training/train.py:128-130,194`] |
| Adam step budget Σ lr_t, base group (5e-4) | 8 epochs: **0.318** rad · 15: **0.565** · 30: **1.095** | [DERIVED: A.1] |
| Same, 2e-4 group | 8: 0.127 · 15: 0.227 · 30: 0.439 | [DERIVED: A.1] |
| Destructive threshold for N = 4 | ρ* = arccos(−3/4) = **2.419** rad; ρ* − π/2 = **0.848** rad; π − ρ* = 0.723 rad | [DERIVED: A.1; dossier C2] |
| Parameters per training example | 1,056,382 / 2251 = **469** (v46 run: 13,205 at n = 80) | [DERIVED; parameter count OBSERVED cell 22] |
| Gate threshold, chance 0.25, exact Clopper–Pearson lower bound > 0.25 | n = 500: ≥ 145 correct (0.290) · n = 570: ≥ 164 (0.288) · n = 2251: ≥ 604 (0.268) | [DERIVED: A.2] |
| Minimum detectable paired difference at n = 2376, power 0.8 | 2.0 / 2.8 / 3.5 points for a discordant rate of 0.1 / 0.2 / 0.3 (α = 0.025, Holm's first step for 2 contrasts) | [DERIVED: A.3; the discordant rate is an ASSUMPTION] |

### 0.2 Cost basis (all [ASSUMPTION], from `docs/baseline/dataset_options.md`)

Measured inputs: cloze scoring at **0.88 s/question** (`qAIR_v44.ipynb`
cell 31) and training at **1.24 s/step** for v44 (cell 21). Assumed: cache
generation 1.4–2.9 s per 4-option question with the 0.5B generator, and
training at 1–3 s/step. Derived from those: full ARC-Easy cache (5197 questions)
**2.0–4.2 h**; one training epoch (141 steps) **2.4–7 min**; one 8-epoch run
**19–56 min**. E8's timed smoke run replaces all of these before any long run.

**Epoch count under PD-7.** The run costs and step budgets in this plan were
written for 8-epoch runs. Under provisional decision PD-7 a Tier 0 or Tier 0′
run trains for at most 20 epochs (48–140 min), so every "× 8 epochs" cost
below is at most 2.5 × larger, and less when early stopping triggers. The
8-epoch step budgets are kept where they explain a mechanism (e.g. T2c).

---

## Tier 0 — trustworthiness (Phases 1–3 of the current fix order)

**Goal.** Measure the v46 model correctly. **Expected accuracy change: none.**
Tier 0 fixes bookkeeping, reproducibility and statistics. With one exception
(A1/A2), it does not change the forward pass, and A1/A2 remove only parameters
that receive no gradient. Any difference from the capstone's 0.50 comes from
the dataset and the evaluation, not from a model change.

**Output.** The true v46 baseline on full ARC-Easy **test**, 3 seeds, for arms
A3, F0, P0 and P1, plus the three generator-only baselines (cloze option
likelihood, letter pick, majority vote). Every number has a manifest and paired
tests (DoD §1–§5).

**A7 is not in Tier 0.** Phase 1 lists it, but it changes the training objective
and needs approval, so it is in Tier 2 (T2a).

### Tier 0 items

Mechanisms, mathematics and tests for each item are in the dossier (§2,
same ID). The table gives the plan-level fields.

| Phase | Item | Fixes | Falsifiable check (pass/fail) | Approval |
|---|---|---|---|---|
| 1 | **E3** seeded, per-batch-reseeded generation; seed and decoding config in cache header | non-reproducible cache | two 20-question builds with the same seed are byte-identical; a resumed build equals an uninterrupted one | No |
| 1 | **E7** `allow_build=False` default | silent hours-long regeneration | loading a capped stub cache with `max_samples=None` raises; the generator constructor is never called | No |
| 1 | **A6** complete flag final; missing split raises; `score_failed` flag | silent state/data changes | three CPU unit tests with a stub generator | No |
| 1 | **A3 + A8** run key `{arm}_{dataset}_s{seed}`; provenance in checkpoints; refuse mismatched resume; per-commit ckpt dir | seeds and stale checkpoints silently resumed | seeds 42/43 give two files and different epoch-1 losses; a fake-commit checkpoint raises | No |
| 1 | **A4** merge resumed history; persist `epochs_no_improve` | truncated curves | interrupted-and-resumed run has one history entry per epoch | No |
| 1 | **A5** config in every checkpoint; `input_ablation` refuses without it | wrong architecture rebuilt | "intact" accuracy equals the logged validation accuracy exactly | No |
| 1 | **A1/A2** remove `hypothesis_temp` (option a), the unused `confidence` head and the unused `phase_head`; loader drops old keys | 675 parameters that cannot learn (dossier A2) | gradient-coverage test lists 0 parameters with `grad is None` | No (option a) |
| 1 | Regression tests: A0 circuit input-dependence, gradient coverage, E3/E7 | silent regressions | `pytest` passes on CPU; the A0 test fails on v40 | No |
| 2 | **E6 + E10 + E11** manifest (commit, cache hash, `pip freeze` hash, devices, seeds, determinism flag) | untraceable numbers | manifest has all keys; commit = `git rev-parse HEAD`; two same-seed T4 smoke runs give identical predictions | No |
| 2 | **E5** one result file per (arm, seed) | overwritten results | two separate calls both appear in the aggregate | No |
| 2 | **E8** timed 20-question smoke build + 1 epoch on T4 | all costs are assumptions | s/question and s/step are in the manifest | No |
| 2 | **E9** stale names and docs | readability | `grep v45` finds no stale reference | No |
| 2 | **Build the full ARC-Easy cache** | B1, B10 | `complete=True` for train/validation/test; counts match the split sizes minus documented filtered rows | **Go-ahead** (> 1 h) |
| 3 | **B4** letter-pick baseline; **B5** evidence-weighted majority vote | missing mandatory baselines | B4 deterministic on 20 items; B5 unit test on a hand-built item | No |
| 3 | **B6** per-seed McNemar + item-level paired bootstrap + Holm | anti-conservative pooled tests | simulated identical arms reject at ≈ 5% | No |
| 3 | **B3, B7** select on validation, report test once; fixed-epoch probes | optimistic numbers | mean (best-val − test) ≥ 0 across arms and seeds | No |
| 3 | **B11** log mean confidence vs accuracy; ECE with bootstrap CI | unmeasured calibration | at n = 570, ECE either clearly exceeds the ≈ 0.05 noise floor or does not | No |
| 3 | **C2 measurement only**: replace the `destructive_fraction > 0` sanity check; report coherent_acc − classical_acc with a paired CI | a metric true by construction | the check no longer passes on an untrained model | No |
| 3 | **Full-split, 3-seed runs** of A3, F0, P0, P1 (closes B1, B2, B8 logging, B10) | n = 20, one seed | every test CI half-width ≤ 0.021 (dossier B1) | **Go-ahead** (> 1 h) |

**Recipe for the Tier 0 runs (provisional decisions PD-7, PD-10 in
[`supervisor_decisions.md`](supervisor_decisions.md)).** Tier 0 is **the v46
architecture trained under the pre-registered protocol**: full ARC-Easy
train, at most **20 epochs**, early stopping on validation accuracy with
patience 5, all other hyperparameters as in v46. Step budget Σ lr_t at 20
epochs: 0.742 (base group) and 0.297 (2e-4 group) if no early stop
[DERIVED: A.1 formula]. Cost 48–140 min per run [ASSUMPTION, §0.2].

*Explained deviation from the capstone.* The v46 run used `EPOCHS = 8` on 80
training items (notebook cell 12 [OBSERVED]), about 40 optimizer steps
(dossier B8). That was a development budget, not a protocol, so Tier 0 does
not copy it; the capstone's number is reported as the capstone's, never as
Tier 0. The originally proposed 8-epoch recipe (D-2) is kept in §2 for the
supervisor's review.

**Diagnostics logged in every Tier 0 run** (they are the reference values that
later tiers are predicted against): collapse entropy and KL to uniform,
energy-row cosine split *within* and *across* channels (C7), drift and gradient
norm of every scalar parameter (B8), coherent_acc vs classical_acc, mean
confidence vs accuracy, input-ablation deltas with CIs (A5 fixed), and the
full-split cache gate (support agreement, attack disagreement, support−attack
margin, cloze accuracy) with exact CIs.

**Tier 0 cost [ASSUMPTION].** Cache 2.0–4.2 h; 12 training runs (4 arms × 3
seeds × 8 epochs) 3.8–11.2 h; cloze and letter-pick baselines on validation +
test (2946 questions) ≈ 0.7 h each at 0.88 s/question; majority vote and stats
on CPU in minutes. **Total ≈ 7–17 h of T4**, spread over several Colab
sessions. Resume correctness (A3/A4) is a precondition for splitting it.

**Tier 0 risks.** (i) The true baseline may sit at or below the generator-only
baselines. That is a valid result (DoD: "working" ≠ "winning"). (ii) If D1 holds
on ARC-Easy (the gate's CI includes chance), Tiers 2–4 cannot be expected to
move accuracy. Tier 1 comes first for this reason. (iii) Colab session limits
interrupt long runs; A3/A4 must be fixed first.

---

## Tier 1 — evidence quality (D1–D5)

**Why this tier comes before any architecture change.** By the
data-processing argument (dossier D1), if the hypotheses carry no information
about the answer beyond the question and options, no aggregator can extract
any. The v46 cache gate found all three hypothesis statistics consistent with
chance on 80 items: 0.175 / 0.1875 / 0.200 vs 0.25 [OBSERVED cell 14; dossier
check 12]. Tiers 2–4 are tested only on a cache that passes the gate below.

> **Priority note (2026-10-28, measured on the full Tier 0 cache, train /
> validation only; dossier D1, D5).** Support agreement is above chance
> (0.3350 / 0.3298, z +9.3 / +4.4), attack disagreement is **below** chance
> (0.1804 / 0.1719, z −7.6 / −4.3), and the v46 **support − attack margin is
> at chance** (0.2554 / 0.2509; 143/570 on validation, below the gate's 164).
> The attack channel is anti-informative and cancels the support signal
> (first-order implied η ≈ 0.9). This makes **T1c attack verification the
> top-priority evidence fix**. **Adopted provisionally (PD-15, 2026-10-28):**
> T1c moves to the front, T1c → T1a → T1b → T1d, a deviation from the
> pre-registered order under PD-3 justified by this train/validation finding.

### Tier 1 pre-training gate (applies to every Tier 1 cache)

> **Gate.** On the full ARC-Easy **validation** split (n = 570 ≥ 500), the
> support−attack margin statistic (`training/dataset.py:711`; the B5 majority
> vote) must beat chance with its exact 95% CI excluding chance. With chance
> 0.25 that means **≥ 164 / 570 correct (0.288)** [DERIVED: A.2]. Also reported:
> the same on train (n = 2251, threshold ≥ 604, 0.268), and support agreement
> and attack disagreement with their CIs.

- **Chance level.** ARC-Easy is "mostly 4 options" (dataset_options.md), not
  all 4. The gate therefore uses the per-item chance Σᵢ 1/Nᵢ / n and an exact
  Poisson-binomial tail, not 0.25. Computed 2026-10-28 from the ARC-Easy
  option counts (train {3: 6, 4: 2241, 5: 4}, validation {3: 1, 4: 567, 5: 2}):
  chance 0.2501 / 0.2500; exact thresholds **605 / 2251** (train) and
  **164 / 570** (validation) (`scripts/gate_significance.py`).
- **Gate is on validation, never test** (rule 5).
- **A cache that fails the gate is not trained on.** Its gate numbers are
  reported as the D1 finding for that generator configuration.

### Staging (to keep one variable per step without four full caches)

Each step changes one thing and is judged first at the gate level on the
**validation split only** (570 questions; 0.2–0.5 h per cache build at 0.5B
rates [ASSUMPTION]). A step is kept only if its gate statistic improves on the
previous step's, by a paired test on the same 570 items (exact McNemar on
per-item margin-vote correctness). Order: **T1c → T1a → T1b → T1d**, cumulative
(PD-15, 2026-10-28; pre-registered order was T1a → T1b → T1c → T1d).
Only the final adopted configuration gets a full cache (train + validation +
test) and training runs.

**What this does and does not attribute.** The gate-level paired comparisons
attribute evidence quality to each component. The training-level comparison
(Tier 0′ vs Tier 0) tests the adopted *bundle* and cannot assign its accuracy
change to individual components. The plan states this rather than claiming
per-component accuracy effects it did not measure.

### T1a — Larger generator (Qwen2.5-1.5B or 3B in fp16, or 7B in 4-bit)

- **Fixes:** D1 (no signal) and, indirectly, D5 (instruction following).
- **Mechanism.** The hypotheses can contain only what the generator can
  articulate about each option. The dossier (D3) shows that per-option
  generation is bounded by the generator's per-option judgement, which the cloze
  baseline measures (0.3375 on 80 ARC-Challenge train items [OBSERVED cell 14]).
  A larger generator raises that bound [HYPOTHESIS: the size of the rise on
  ARC-Easy is unmeasured here]. It should also follow the "argue this answer
  fails" instruction more reliably, lowering η (D5) [HYPOTHESIS].
- **Pre-screen (cheap, no cache).** Zero-shot cloze and letter-pick accuracy of
  each candidate on 300 **validation** questions (≈ 4–5 min each at the 0.5B
  rate; slower for larger models [ASSUMPTION]). Choose the generator on
  validation only. Do not cite published benchmark numbers for the larger models
  until they are fetched and checked.
- **Falsifiable prediction.** For the chosen generator vs 0.5B on the same 570
  validation items: support−attack margin rises and passes the gate; cloze
  accuracy rises. If the margin statistic does not rise, generator size is not
  the limit (then T1b/T1c are the next candidates).
- **Single-variable comparison.** Gate level: 0.5B cache vs larger-generator
  cache, same prompts, same items, same seed. Training level: Tier 0 recipe on
  the new cache vs Tier 0, 3 seeds.
- **Consequence for the success criterion.** The generator also produces the
  LLM-prior feature and both direct LLM baselines. With a larger generator, the
  baselines **must be recomputed with that generator**: qAIR has to beat the
  generator it uses, not the 0.5B model. This raises the bar, and the plan
  accepts that.
- **T4 cost [ASSUMPTION].** Weights in fp16 take about 2 bytes per parameter:
  ≈ 3.1 GB (1.5B), ≈ 6.2 GB (3B); 7B in 4-bit ≈ 4 GB plus overhead. All fit a
  16 GB T4. Parameter counts are from memory of the Qwen2.5 model cards and are
  not fetched this session. Generation time is assumed to scale roughly with
  parameter count: × 2–3 (1.5B) or × 4–6 (3B) on the 2.0–4.2 h cache, i.e.
  ≈ 4–25 h for a full cache. The 20-question timed smoke build (E8) must be
  repeated per generator before deciding. A T4 has no bf16 support, so fp16 it
  is [ASSUMPTION].
- **Risks.** Full-cache builds exceed one Colab session; resumable seeded
  builds (E3) are required. fp16 overflow in some layers [ASSUMPTION: check for
  NaN log-probs in the smoke build]. Licence: check each candidate's model card
  (one size may be under a research-only licence [ASSUMPTION]; fine for
  coursework but must be stated). Contamination rises with model size
  [ASSUMPTION]; it shifts generator and qAIR equally in paired comparisons.
- **Supervisor approval:** **Yes** (changes the model's input evidence).
  Provisional decision PD-3: proceed behind a flag as a separate arm; the v46
  cache stays the default.

### T1b — Contrastive prompts that see all options

- **Fixes:** D3 (each prompt sees one option), and through it D1 and D2.
- **Mechanism (dossier D3).** An attack on option m is sampled from
  p(· | q, o_m). Without the alternatives it cannot say "B is wrong because C",
  so the attack channel is bounded by the generator's judgement of o_m in
  isolation. Showing all options lets it contrast. Alignment is kept by naming
  one **target** option per prompt ("argue that option B fails, given that the
  options are A–D"). K stays 2N, so no tensor shape changes.
- **Falsifiable prediction.** On the same 570 validation items: the
  attack-disagreement statistic rises (paired test), and the support−attack
  margin rises. If neither moves, the generator's knowledge, not the prompt, is
  the limit (dossier D3).
- **Alignment check (must pass, or the step is rejected).** The fraction of
  hypotheses whose MiniLM embedding is closest to their *target* option must
  not fall relative to the previous step. Pre-v44 alignment drift is the known
  failure (generator docstring, `models/generator.py:45-50`).
- **Single-variable comparison.** The previous step's cache vs the same with
  the contrastive prompt, same generator and seed.
- **T4 cost [ASSUMPTION].** Prompts are longer (all options), so generation
  costs about × 1.2–1.5. Validation-only gate build ≈ 0.3–0.7 h at 0.5B rates.
- **Risks.** Alignment drift. The prompt could leak position cues. No label
  leakage: the generator never sees the answer key.
- **Supervisor approval:** **Yes.** Provisional decision PD-3: flagged
  separate arm.

### T1c — Attack verification by LLM scoring; filter or regenerate non-compliant attacks

- **Fixes:** D5 (attacks that support their option).
- **Mechanism (dossier D5).** If an attack actually supports its option with
  probability η while keeping the −1 label, the attack channel's expected
  signed evidence is (1 − 2η) times its ideal value. It carries no signal at
  η = 0.5 and is anti-informative above it. Lowering η raises the channel's
  signal linearly.
- **Procedure.**
  1. **Measure η first.** Manually audit 100 random attacks from the train cache
     with a fixed rubric; report η̂ with an exact CI.
  2. **Verifier.** Score each attack with a stance prompt to the LLM: "Does this
     text argue that answer X is wrong? yes/no", using the log-prob of the
     answer token. The verifier judges *stance*, never correctness, and never
     sees the answer key, so no label enters the cache. Calibrate its threshold
     against the 100-item manual audit (report precision and recall), on train
     items.
  3. **Regenerate** a failing attack up to R = 3 times [ASSUMPTION: R]. If it
     still fails, keep it and set a per-hypothesis `stance_failed` flag; mask it
     out of the collapse only in a separate arm. Regenerating keeps K = 2N, so
     shapes are unchanged. Masking uses the existing (B, K) hypothesis mask.
- **Falsifiable prediction.** η̂ after verification is below η̂ before, with
  non-overlapping CIs. Attack disagreement and the margin statistic rise on the
  same 570 items. Dossier prediction: η̂ before is > 0.2 [HYPOTHESIS]. If
  η̂ < 0.05 before, T1c is dropped as immaterial.
- **Single-variable comparison.** Previous step's cache vs the same with
  verification and regeneration.
- **T4 cost [ASSUMPTION].** 4 attacks × 5197 questions = 20,788 stance checks.
  At the measured cloze rate (0.88 s per question of 4 scored sequences) that is
  ≈ 1.3 h with the 0.5B model, plus regeneration of the failures. It scales with
  verifier size.
- **Risks.** The same model as generator and verifier makes errors correlated
  and may only filter what it can already judge. A larger verifier is a further
  variable, so keep the verifier fixed across steps. Regeneration biases attacks
  towards phrasings the verifier likes.
- **Supervisor approval:** **Yes** (filtering or regeneration). Provisional
  decision PD-3: flagged separate arm.

### T1d — 2–3 samples per hypothesis

- **Fixes:** D4 (one sampled hypothesis per (option, mode)).
- **Mechanism (dossier D4).** One draw at T = 0.7 carries the full generation
  variance. Averaging the embeddings of M draws reduces its standard deviation
  by 1/√M: × 0.71 for M = 2, × 0.58 for M = 3 [DERIVED]. **Shape:** embed M·K
  texts, (M·K, 384) → mean over the M samples of each (option, mode) →
  (K, 384). The model input is unchanged; only the cache stores M texts per
  slot.
- **Falsifiable prediction.** The margin statistic rises on the same items, and
  the per-item variance of hypothesis embeddings across samples is reported. If
  the margin does not rise, sampling noise is not the limiting term.
- **Single-variable comparison.** M = 1 vs M = 2 (then 3), same prompts,
  generator, verifier and seed.
- **T4 cost [ASSUMPTION].** Generation × M. Embedding cost is small by
  comparison [ASSUMPTION].
- **Risks.** Averaging dilutes distinctive samples. Cost × M on top of a larger
  generator may exceed a reasonable budget; drop T1d first if so.
- **Supervisor approval:** **Yes** (D4: changing M). Provisional decision
  PD-3: flagged separate arm.

**Tier 1 training cost [ASSUMPTION].** Adopted full cache: 4–25 h depending on
the generator (T1a), × M. Tier 0′ runs on it: F0 and A3 × 3 seeds × 8 epochs =
1.9–5.6 h. Generator-only baselines recomputed with the new generator: about
0.7 h each at 0.5B speed, more for larger models.

---

## Tier 2 — learning signal

Run on the adopted cache (Tier 0′ reference). If no cache passes the gate,
Tier 2 runs on the 0.5B cache **as a mechanism check only**, with the
pre-stated expectation of no accuracy change.

### T2a — A7 soundness-gated validator target (v47 patch)

- **Fixes:** A7 (the relevance target has no hypothesis axis); first suspect for
  C6.
- **Mechanism (dossier A7, check 3).** With the v46 target, every hypothesis is
  told to score the true option 1, so the loss gradient at the true option is
  negative for every hypothesis, sound or not. The quantity that reaches the
  collapse, the option-mean of the target, is 1/N for every hypothesis: zero
  information distinguishing them even at zero loss. The gated target
  multiplies by soundness s_k (1 for a support of the true option or an attack
  on a wrong option, else 0). Its option-mean s_k/N then varies with k, and
  unsound hypotheses are pushed down at the true option. Shapes unchanged
  (target (B, K, N) from `align` (B, K, N) and `polarity` (B, K)); no new
  parameters; checkpoints compatible.
- **Scope of the change.** Only the `_relevance_target` hunk of
  `models/validator.py` from `~/qair_backup/v47_uncommitted.patch` (sha256
  `59a0d595…6010437` [VERIFIED: `shasum -a 256` this session]). The patch's
  other hunks (CLAUDE.md text; the `amp_entropy` metric in
  `training/evaluate.py` and `training/train.py`) are not part of this
  variable. `amp_entropy` is measurement-only; *decision D-5* asks whether it
  may be added in Tier 0 so the reference run logs it too. Provisional
  decision PD-9: logged as a diagnostic only.
- **Falsifiable prediction.** (1) Mean collapse weight on sound hypotheses
  exceeds that on unsound ones (soundness is known from labels; used only as a
  diagnostic). (2) Collapse KL to uniform rises clearly above the Tier 0′ value,
  which v46 measured at 0.0031 nats on 20 items. (3) Accuracy: a small rise
  **only if** the cache passed the gate [HYPOTHESIS]; no accuracy change on a
  gate-failing cache. If KL does not rise, A7 is not the bottleneck and C6/C7/D1
  are (dossier C6).
- **Single-variable comparison.** Tier 0′ F0 and A3 vs the same with the gated
  target, 3 seeds, paired.
- **T4 cost [ASSUMPTION].** No extra compute per run. 6 runs (F0, A3 × 3 seeds)
  × 8 epochs: 1.9–5.6 h.
- **Risks.** Labels are used in a new way during training (still none at eval:
  `training/evaluate.py:122` passes no `y`). Collapse metrics are not comparable
  with the capstone.
- **Supervisor approval:** **Yes** (training objective). Provisional decision
  PD-2: an extra arm behind a flag; the v46 target stays the default and the
  main baseline. The v47 patch is not applied (CLAUDE.md).

### T2b — Training budget: a longer epoch cap than Tier 0′ (re-specified for PD-7)

- **Fixes:** B8 (40 optimizer steps in v46) beyond what full data already gives.
- **Mechanism (dossier B8).** Adam moves a parameter by at most about lr per
  step for a consistent gradient. The total movement budget Σ lr_t for the base
  group is 0.318 rad at 8 epochs, 0.565 at 15 and 1.095 at 30 [DERIVED: A.1],
  against 0.0113 in the v46 run (dossier check 5). Full train data also lowers
  capacity per example from 13,205 to 469 parameters [DERIVED].
- **Falsifiable prediction.** Train loss is lower at the end. Scalar drifts are
  ≫ 0.01 rad, or their logged gradients are ≈ 0 with alternating sign (the C2
  stationary-point signature). Validation accuracy: no predicted direction
  [HYPOTHESIS]. With 469 parameters per example, longer training may overfit;
  `PATIENCE = 5` (`config.py:270`) applies, and best-val vs test is reported.
- **Single-variable comparison (re-specified 2026-10-19 for PD-7).** Tier 0′
  (cap 20 epochs, patience 5) vs the same with a **30-epoch cap** (patience
  5), same seeds. Step budget Σ lr_t without early stopping: 0.742 vs 1.095
  (base group), 0.297 vs 0.439 (2e-4 group) [DERIVED: A.1]. Note: `T_max`
  equals the epoch cap, so the LR schedule stretches with the cap even when a
  run stops early. That is part of "the budget" as one variable. The
  15-epoch arm of the original plan is dropped: it is below the Tier 0′ cap.
  The mechanism and prediction above were written against 8 epochs; the
  prediction still applies to the 20 → 30 step.
- **T4 cost [ASSUMPTION].** One 30-epoch run: 72–210 min. 6 runs: 7–21 h.
  Runs over 1 h are expected; they are run on Colab by me (PD-7).
- **Risks.** Overfitting; cost. **Supervisor approval:** no architecture
  approval; go-ahead for > 1 h runs. Provisional decision PD-4: separate
  flagged arms.

### T2c — Separate LR group for phase and temperature scalars

- **Fixes:** B8/F6 (scalars cannot move within the budget); a precondition for a
  meaningful C2 test (T3b).
- **Group membership** [VERIFIED: `grep nn.Parameter( models/`]:
  `polarity_phase` (`models/full_model.py:226`, base group 5e-4),
  `interference.option_temp` (`models/interference.py:105`, 5e-4),
  `answer_selector.temperature` (`models/answer_selector.py:119`, 5e-4),
  `validator.temperature` (`models/validator.py:88`, 2e-4 group),
  `quantum.phase_gain` (`models/quantum_layer.py:219`, 2e-4 group). The
  collapse temperature is the output of a `Linear(32, 1)` head
  (`models/collapse.py:53,83`), not a scalar, so it stays in its group.
  `hypothesis_temp` is gone after A1 (option a).
- **Step count for ρ, from the actual LR [DERIVED: A.1].** Moving ρ by 0.848 rad
  (π/2 → ρ* = 2.419, the T3b crossing) takes at least 0.848 / 5e-4 =
  **1,696 steps ≈ 12.0 epochs** at a constant 5e-4, even with a perfectly
  consistent gradient sign. Under the actual cosine schedule:
  - 8 epochs: total budget 0.318 rad. **Cannot reach.**
  - 15 epochs: 0.565 rad. **Cannot reach.**
  - 30 epochs: 1.095 rad. Reaches 0.848 only **at the end of epoch 15**, and
    only if every update has the same sign.
  So at the current LR, the T3b test is decided by the budget, not by the data.
- **Proposal.** A scalar group whose budget over the **first half** of the run
  is 3 × 0.848 rad. The factor 3 is an [ASSUMPTION] margin for gradients of
  inconsistent sign. This needs a peak LR of **2.96e-3 at 15 epochs** or
  **1.44e-3 at 30 epochs**, same cosine shape [DERIVED: A.1]. Weight decay stays
  0.05 to keep this one variable. Decoupled decay pulls ρ towards 0: from π/2
  that is about −0.044 rad over 15 epochs or −0.084 over 30 at 5e-4 [DERIVED].
  This is small against 0.848 but works against an upward crossing, so it is
  reported.
- **Falsifiable prediction.** At the π init, ρ's movement grows with the LR but
  stays sign-inconsistent, a random walk around a stationary point (dossier C2),
  so its net drift is ≪ its path length. Temperature scalars move ≫ 0.01.
  Accuracy: no change predicted on its own [HYPOTHESIS].
- **Single-variable comparison.** Tier 0′ at the T2b budget vs the same plus the
  scalar group.
- **T4 cost [ASSUMPTION].** None per run. 6 runs at the chosen budget.
- **Risks.** Instability. The temperatures are floored (`0.3 + softplus` at
  `models/interference.py:168`; `0.5 + softplus` at `models/validator.py:177`
  and `models/answer_selector.py:201` [VERIFIED]), so they cannot reach 0, but
  a large LR can still make them oscillate. Log every scalar per epoch.
- **Supervisor approval:** no architecture change; **inform the supervisor**
  because it changes training dynamics. *Decision D-6.* Provisional
  decision PD-4: separate flagged arm.

### T2d — Re-measure C6 and C7 before changing them

- **Fixes:** nothing directly. It decides whether C6/C7 need a change.
- **Mechanism (dossier C6, C7).** Collapse KL ≈ Var_k(E)/(2T²): a near-uniform
  collapse means tiny energy differences at an unmoved temperature. Energy-row
  convergence is predicted by common-mode gradients when amplitudes are
  near-uniform.
- **Falsifiable prediction.** If A7 and the budget were the causes, KL after
  T2a–T2c is clearly above the Tier 0′ value, and within-channel energy-row
  cosine stops rising. If not, the dossier's prediction applies: within-channel
  cosine rises faster than across-channel, and C6/C7 are discussed as design
  items. No change is proposed before this measurement.
- **Comparison.** Logged diagnostics of the T2a–T2c runs vs Tier 0′.
- **Cost:** none (logging). **Approval:** no (measurement); any change after it:
  yes.

---

## Tier 3 — readout (C1, C2, C3; all need approval; provisional decision PD-5)

### T3a — C1 signed-readout arm

- **Fixes:** C1 (the Born readout ranks by the size of support − attack, not its
  sign).
- **Mechanism (dossier C1, check 2).** At ρ = π with phase-aligned channels, the
  Born score is (S − R)². An option with support 0.6 and attack 0.1 ties one
  with support 0.1 and attack 0.6. The signed arm scores
  score_n = Re(A_n · e^{−iψ}) with a learned reference phase ψ (one scalar), and
  takes a softmax over options for probabilities. Shapes unchanged ((B, N)
  scores); one new scalar key, defaulting to 0 for old checkpoints.
- **Honest statement of what this arm is.** With aligned phases, ρ = π and
  ψ = 0, the signed score equals S_n − R_n exactly: **plain additive
  scoring**. It differs from additive scoring only through the spread of the
  circuit and compatibility phases. If those phases stay near-aligned (v46
  measured phase spread 0.11 [OBSERVED cell 24]), this arm *is* additive scoring
  and must be reported as such, not as interference. The dossier's
  reference-amplitude variant, |c e^{iψ} + A_n|², ranks like additive scoring
  by proof (dossier C1). So neither variant demonstrates interference by its
  ranking.
- **Falsifiable prediction.** (1) CPU unit test: on the constructed two-option
  example, the signed arm ranks the supported option first (Born gives a tie).
  (2) On data, if C1 is binding: signed-arm accuracy > Born-arm accuracy
  (paired). If signed ≈ Born, the sign symmetry was not the binding constraint
  (D1 is). (3) Report the learned phase spread. If it stays < ~0.2 rad
  [ASSUMPTION threshold], label the arm "additive scoring" in every table.
- **Single-variable comparison.** Tier 0′ A3 (Born) vs A3 with the signed
  readout, 3 seeds; P0 and P1 reported alongside as controls.
- **T4 cost [ASSUMPTION].** As one Tier 0 arm: 3 runs, 1–3 h.
- **Risks.** It changes the central claimed mechanism; it is not comparable
  with the capstone. A win would be a win for signed or additive evidence
  aggregation, not for "quantum" interference, and must be described that way.
- **Supervisor approval:** **Yes.** *Decision D-7.* Provisional decision PD-5:
  separate pre-registered arm; the Born readout stays the default.

### T3b — C2 ρ = π/2 initialization arm (pre-registered; π arm kept)

- **Fixes:** C2 (π is a stationary point; `destructive_fraction` saturated
  at init).
- **Mechanism (dossier C2, checks 4–5).** At π with aligned channels, the
  gradient of the readout in ρ is exactly 0. At π/2 it is maximal in magnitude,
  and the destructive fraction starts at 0 because π/2 < ρ* = 2.419. Any learned
  cancellation then shows as ρ crossing 2.419.
- **Pre-registration (written now, before any run).**
  - Both arms are run: ρ₀ = π (as v46) and ρ₀ = π/2, same seeds, same budget,
    **with the T2c scalar LR group** (without it, the 0.848 rad crossing is not
    reachable; see T2c).
  - **H1 (cancellation is learned):** in ≥ 2 of 3 seeds, ρ ends above 2.419,
    destructive_fraction rises from ≈ 0, and coherent_acc − classical_acc > 0
    with a paired CI excluding 0.
  - **H0 (not learned):** ρ stays near π/2 or falls, and destructive_fraction
    stays ≈ 0.
  - Any other outcome is reported as inconclusive.
  - The π arm is kept as the comparison, and no third init is tried on the same
    data after seeing the results.
- **Falsifiable prediction.** H1 vs H0 as above. No accuracy direction is
  predicted.
- **Single-variable comparison.** ρ₀ = π vs ρ₀ = π/2, all else equal (both on
  Tier 0′ + T2c).
- **T4 cost [ASSUMPTION].** 3 extra runs at the T2b budget (the π arm is the T2c
  run).
- **Risks.** It changes the init; not comparable with the capstone. A
  "learned cancellation" result still rests on D1 being resolved.
- **Supervisor approval:** **Yes.** *Decision D-8.* Provisional decision PD-5:
  separate pre-registered arm; the ρ = π init stays the default.

### T3c — C3 abstention with risk–coverage reporting

- **Fixes:** C3 (always argmax; no abstention).
- **Mechanism (dossier C3).** If confidence is informative, rejecting the
  lowest-confidence items raises accuracy on the rest. This trades coverage for
  risk. It needs calibrated confidence (B11).
- **Procedure (evaluation-only).** Confidence = max_n P(n) (the margin between
  the top two is a secondary choice, fixed now). Report the risk–coverage curve
  and its area (AURC) on test. Any operating threshold (e.g. 80% coverage) is
  chosen on validation. Hypothesis **pruning** stays deferred until the collapse
  is selective (T2d), because pruning a near-uniform collapse removes signal at
  random (dossier C3).
- **Falsifiable prediction.** Accuracy at 80% coverage > full-coverage accuracy
  with a paired bootstrap CI excluding 0, if confidence is informative. Compare
  the curves of qAIR and the cloze baseline (its softmaxed scores), because the
  generator alone may already rank its own confidence as well.
- **Single-variable comparison.** No model change. Curves from the same Tier 0′
  checkpoints.
- **T4 cost:** none beyond evaluation (minutes).
- **Risks.** Abstention can make a weak model look good. Always report coverage
  next to accuracy. **Supervisor approval:** curve (measurement) **no**;
  pruning **yes**. Provisional decision PD-5: abstention as a separate
  pre-registered arm; pruning is not covered.

---

## Tier 3b — path Hamiltonian: a joint quantum state over all reasoning paths (needs approval; provisional decision PD-5)

Full design, mathematics, references and numerical checks:
[`docs/baseline/path_hamiltonian_design.md`](path_hamiltonian_design.md)
(checks: `scripts/verify_path_hamiltonian.py`, 33/33 pass). This tier is not
the same as item T3b above (the ρ = π/2 init arm); it is a separate readout.

- **Fixes:** the core qAIR idea was never implemented. v46's circuit processes
  each hypothesis separately (`models/quantum_layer.py:250-254` reshapes to
  (B·K, D) [VERIFIED]). This readout replaces the interference collapse whose
  limits are C1, C2, C6 and C7. The per-hypothesis circuit stays upstream
  (rule 6).
- **Mechanism.**
  - One qubit per path (K = 2N = 8 for 4 options); a basis state is a "world"
    of paths holding or failing.
  - Energy terms come from a unary head and a **directed pair head** in the 0/1
    basis. The pair head has symmetric co-occurrence b and one-directional
    "i requires j" penalties r·x_i(1 − x_j). These map exactly to Ising fields
    and couplings (design §2.1; check 13).
  - H1 = Σ h_i Z_i + Σ J_ij Z_iZ_j − Γ Σ X_i; H2 = H1 + Σ K_ij X_iX_j with
    unconstrained sign.
  - Readout: the thermal state at learned T (**decision P-2 accepted**).
  - The answer is the probability that an option's support holds and its
    attack fails, normalized; cross-entropy as in v46.
  - Data enters only through the energy terms and the basis encoding.
- **Arms** (thermal readout, A3 base):

  | Arm | What it is | Isolates |
  |---|---|---|
  | **H0** | Γ ≡ 0: classical Ising Gibbs distribution over worlds | — |
  | **H1** | Γ learned: superposition, but stoquastic, so provably no interference between paths (Perron–Frobenius; non-negative Gibbs matrix, Bravyi et al. 2008) | H1 vs H0: superposition (claim Q) |
  | **H2** | H1 + K_ij X_iX_j with sign unconstrained: non-stoquastic in the world basis, so amplitudes can carry signs and paths can cancel | H2 vs H1: interference (claim I) |
  | H2s | K ≤ 0: same capacity as H2 but stoquastic | H2 vs H2s: sign vs extra capacity |
  | H1-sym | r ≡ 0, symmetric pair terms only | directed terms (ablation) |
  | H1-GS | ground-state readout (secondary) | readout choice |

- **Falsifiable predictions (pre-registered, design §6).**
  - **Q is supported** only if all of these hold:
    - H1 − H0 > 0 with the paired CI excluding 0, ahead in all 3 seeds;
    - Γ stays in use;
    - the ground state is entangled.

    Q is falsified if Γ collapses or an adequately powered test shows no gain.
  - **I is supported** only if all of these hold:
    - H2 > H1 (CI excluding 0, all seeds);
    - H2 > H2s;
    - the learned H2 really has sign structure: median negative-amplitude mass
      ≥ 0.01 and sign-stripped readout distance ≥ 0.01.

    **H2 ≈ H1 falsifies I.** So does unused sign structure, or H2 ≈ H2s.
  - Expected [HYPOTHESIS]: H1 ≥ H0 by < 2 points; H2 ≈ H1.
  - Positive results show that a transverse-field relaxation, or sign structure
    in the world basis, helps. They are not quantum advantage.
  - Stoquastic is not the same as efficiently simulable in general (Bravyi 2015;
    Hastings 2013). At K ≤ 10 everything is computed exactly anyway.
- **Single-variable comparisons.**
  - Primary: H1 vs H0 (Γ) and H2 vs H1 (the K term), each at α = 0.05.
  - Required attribution: H2 vs H2s.
  - Holm family: H1 vs H1-sym; arms vs Tier 0′ A3, LLM cloze and letter pick;
    H1-GS vs H1.
  - All use 3 seeds, full ARC-Easy, and test once.
- **Prerequisites.** Tier 0, and a Tier 1 cache that passes the gate. Every
  energy term is a function of the path embeddings, so with no hypothesis
  signal (D1) all arms tie for reasons unrelated to Q or I. Recommended first:
  T2a and T2c.
- **T4 cost [ASSUMPTION].** CPU (Apple M4, numpy) eigh forward: 47 ms per batch
  of 16 at K = 8 and 1.15 s at K = 10 [VERIFIED: `--bench`]. On a T4, K = 8 adds
  little to 1–3 s/step; K = 10 batches may double. Primary arms H0, H1, H2 × 3
  seeds × 8 epochs ≈ 3–10 h; H2s, H1-sym and H1-GS add ≈ 3–10 h. A CPU torch
  micro-benchmark comes first (design §8).
- **Risks.**
  - near-degenerate ground states (H1-GS);
  - pair terms not learning without signal;
  - Γ collapsing or H2 learning K ≤ 0 (pre-registered outcomes, not
    prevented);
  - H2 gains from capacity (the H2s control);
  - basis-dependence of stoquasticity (Marvian et al. 2019);
  - overfitting the ~33.5 k-parameter heads;
  - over-claiming.
- **Integration.** New `models/path_hamiltonian.py`; a default-off
  `readout="path"` flag in `models/full_model.py`; arms in
  `training/ablations.py`; logging in `training/evaluate.py`. The v46 path is
  unchanged. That is 4 files, over the 3-file threshold.
- **Supervisor approval:** **Yes.**
  - Decisions: D-12 (H0/H1), D-13 (H2 + H2s), D-14 (directed head +
    H1-sym), D-15 (4-file integration).
  - P-2 (thermal readout) is already accepted.
  - Provisional decision PD-5 (`supervisor_decisions.md`): H0/H1/H2 proceed
    as separate pre-registered arms behind the default-off flag. D-14 is
    still open.

---

## Tier 4 — bottleneck (C4, C5; all need approval; provisionally deferred, PD-6)

### T4a — C4 encoder options

- **Fixes:** C4 (all generator knowledge reaches the model through frozen 384-d
  mean-pooled MiniLM, except the scalar LLM prior; `config.py:81-96` states
  this).
- **Mechanism (dossier C4).** For a frozen encoder e, I(y; e(H)) ≤ I(y; H): the
  encoder can only lose information. A better representation loosens the bound;
  it cannot exceed what the hypotheses contain (so Tier 1 comes first).
- **Probe-first gate (cheap, no qAIR training).** For each option, train the
  B7-fixed probes on the cached representation (train) and evaluate on
  validation. Only an option whose probe beats the MiniLM probe on validation
  (paired test) gets a qAIR training run.
- **Options.**
  1. **bge-base encoder** (`BAAI/bge-base-en-v1.5`, 768-d [ASSUMPTION: from
     memory]). Shape: `EMBEDDING_DIM` 384 → 768, so the input projection
     becomes 768 → 128 and its weight shape changes. The cache must be
     re-embedded (a dim-mismatched cache is refused, per the comment at
     `config.py:81-86`); the
     checkpoint format needs a version bump (rule 9).
  2. **LLM log-prob features.** Per-hypothesis scalars from the generator,
     cached: e.g. log P(hypothesis | q, o_m), and the T1c stance score. Shape:
     the answer-selector feature vector (`models/answer_selector.py:142-199`)
     grows by f scalars per (k, n). This is a narrow bypass of MiniLM, like the
     existing option-level prior.
  3. **Joint encoding.** Encode (question, option, hypothesis) together, using a
     cross-encoder or the generator's last hidden state. Shape: H (B, K, 384) →
     (B, K, d), with d the chosen model's width. This is the largest change and
     the highest cost.
- **Falsifiable prediction.** Per option: probe accuracy on validation improves
  over MiniLM. qAIR accuracy improves only if the probe does. If no probe
  improves, the encoder is not the binding constraint.
- **Single-variable comparison.** Tier 0′ vs the same with one encoder option,
  3 seeds; one option at a time.
- **T4 cost [ASSUMPTION].** (1) Re-embedding 5197 questions × 2N texts: small
  next to generation. (2) One scoring forward per hypothesis: ≈ 1–2 h at 0.5B
  rates. (3) A full forward pass per hypothesis text with hidden-state
  extraction: ≈ cache-scoring cost, more for larger generators. Plus 3 training
  runs per option kept.
- **Risks.** Pipeline change; option 3 breaks checkpoint compatibility; larger
  inputs raise overfitting risk at 2251 training items.
- **Supervisor approval:** **Yes.** *Decision D-9.*

### T4b — C5 cleanup

- **Fixes:** C5 (redundant hamiltonian and cosine energies; a doubled reasoner
  and circuit pass when validator feedback is on).
- **Mechanism (dossier C5).** After per-sample standardization, the two energies
  differ only by a per-option factor 1/‖O′_n‖. They are near-collinear, so their
  fusion weights are weakly identified. The notebook states that disabling
  feedback "roughly halves per-epoch time" [OBSERVED cell 12 note].
- **Changes (each a separate arm).** (i) Drop the cosine energy. (ii) Make
  validator feedback an ablation instead of the default.
- **Falsifiable prediction.** (i) Accuracy change within seed noise (|Δ| below
  ~2× the seed std). (ii) Accuracy within seed noise, per-epoch time down by
  ≈ 40–50% [ASSUMPTION based on the notebook note]. If either costs accuracy
  beyond seed noise, keep it.
- **Single-variable comparison.** Tier 0′ vs each change, 3 seeds.
- **T4 cost [ASSUMPTION].** 3 runs per change; (ii) makes later runs cheaper.
- **Risks.** Parameter counts change (matched-parameter claims). **Supervisor
  approval:** **Yes.**

---

## Overall sequence and stopping rules

```
Tier 0 (bugs, cache, baselines, 3-seed runs) ── true v46 baseline
   │
Tier 1 gate-level staging on validation: T1c → T1a → T1b → T1d (PD-15)
   ├─ no configuration passes the gate ─→ STOP architecture work.
   │      Report D1 as the finding; Tier 2 only as a mechanism check.
   └─ adopted cache passes ─→ full cache ─→ Tier 0′ runs (new reference)
          │
Tier 2: T2a (A7) → T2b (budget) → T2c (scalar LR) → T2d (re-measure C6/C7)
          │
Tier 3: T3a (signed readout) ∥ T3b (π/2 vs π, pre-registered) ∥ T3c (risk–coverage)
          │
Tier 3b: path Hamiltonian H0 → H1 → H2 (+ H2s, H1-sym, H1-GS), pre-registered; gate-passing cache only
          │
Tier 4: T4a probes → T4a training for options that pass ∥ T4b
```

Each tier's adopted change becomes part of the reference only if its
single-variable comparison favoured it on **validation**. Test numbers are
reported, never used to adopt. Under the provisional decisions
([`supervisor_decisions.md`](supervisor_decisions.md)), "reference" means the comparison
reference for later arms only: the code default stays v46, and every result
table also reports the v46 baseline.

**Rough total [ASSUMPTION; replaced by E8 measurements].** Tier 0 ≈ 7–17 h;
Tier 1 ≈ 6–30 h (dominated by the generator choice); Tier 2 ≈ 10–30 h (30-epoch
runs dominate); Tier 3 ≈ 3–10 h; Tier 3b ≈ 6–20 h; Tier 4 ≈ 3–15 h. These are T4 hours over many
Colab sessions; every run over 1 h needs the go-ahead.

---

## 1. One-page summary for my supervisor

**Where we are.** The capstone result (50% vs the language model's 40%, on 20
questions) is not evidence either way. The issue dossier lists the bugs and
measurement errors behind that, and this plan orders the work that turns qAIR
into an honest test of its idea.

**Tier 0: make the measurement trustworthy.** Fix the bookkeeping bugs (runs
resuming the wrong checkpoint, lost training history, unreproducible
hypothesis generation, parameters that never learn), add the two missing
baselines, and correct the statistics. Then run the unchanged v46 model on the
full ARC-Easy dataset with three seeds. No accuracy change is expected from
this tier. Its output is the true v46 number on 2,376 test questions, next to
the language model's own answers. Roughly 7–17 GPU hours (estimate).

**Tier 1: give the model evidence worth reasoning about.** In the capstone
run, the generated support and attack arguments told us nothing about the
right answer: they scored at chance. No reasoning architecture can recover
information that is not in its input. So before touching the architecture, we
improve the arguments: a larger language model; prompts that show all the
options; checking that "attacks" actually attack, and regenerating those that
don't; and two or three samples per argument. Each step is kept only if it
measurably improves the arguments on the validation questions. There is a hard
gate: we do not train on a new set of arguments unless they beat chance clearly
on at least 500 questions. If nothing passes, that is itself the main finding,
and we report it.

**Tier 2: let the model learn.** Fix the validator's training target, which
currently tells every argument to say the same thing (needs your approval).
Train longer. Give the phase and temperature settings a learning rate at which
they can actually move: at the current rate, the phase cannot reach the point
where cancellation begins even in 15 epochs. Then re-measure whether the model
selects between arguments before changing anything else.

**Tier 3: test the quantum-style readout honestly.** (a) A readout that keeps
the sign of the evidence; we will say plainly that it reduces to simple
"support minus attack" scoring when the phases are aligned. (b) A
pre-registered test that starts the phase where learning is possible and checks
whether cancellation is learned, with the original setting kept for
comparison. (c) Letting the model abstain on low-confidence questions,
reported as accuracy against coverage.

**Tier 3b: build the idea qAIR was named for.** The capstone circuit looked at
each argument on its own. This design gives every argument a qubit, so the
quantum state covers every combination of arguments being true or false, and
pairs of arguments can exclude, require or support each other. Three
versions differ in one switch at a time: classical; quantum superposition
without cancellation; and quantum with signed amplitudes that let arguments
cancel. A control rules out "it only helped because it had more parameters". What counts
as success and as failure is fixed in advance (see
`path_hamiltonian_design.md`).

**Tier 4: widen the bottleneck.** Everything the language model knows reaches
qAIR through a small frozen sentence encoder. We test better options (a larger
encoder, extra language-model scores, joint encoding), each first with a cheap
probe, and remove redundant computation.

**Success** means qAIR beats both direct uses of the same language model
(scoring each option, and picking a letter) on the full ARC-Easy test set,
across three seeds, with a significant paired test. With 2,376 test questions,
differences smaller than about 3 points cannot be shown reliably (estimate).

**Honesty commitments.** No accuracy claim from 20 questions. No "quantum
advantage" claim (the circuit is simulated). If a readout reduces to additive
scoring, we call it additive scoring. Test data is never used to make choices.

---

## 2. Decisions needed from my supervisor

The last column is the provisional decision recorded in
[`supervisor_decisions.md`](supervisor_decisions.md) while the supervisor is away; each is
provisional — pending supervisor review. "Open" means no provisional
decision was taken.

| # | Decision | My recommendation | Provisional decision |
|---|---|---|---|
| D-1 | Approve Tier 0 as scoped (bugs, baselines, statistics), and the go-ahead for the ARC-Easy cache build and the 3-seed runs (each > 1 h on T4) | Approve | PD-1: proceed |
| D-2 | Tier 0 epoch count: 8 (v46 as run) or 30 with patience 5 (`config.py` default) | 8, so Tier 0 is "v46 as trained"; longer budgets are tested in T2b | PD-7: at most 20 epochs, patience 5; Tier 0 = v46 architecture under this protocol |
| D-3 | Which arm is "qAIR" for the success criterion: F0 (with the LLM prior; deployed system) or A3 (no prior) | F0 as primary, A3 always reported: F0 contains the cloze score as a feature, so A3 shows whether reasoning adds anything without it | PD-8: F0 primary, A3 secondary |
| D-4 | Tier 1: allow cache rebuilds with (a) a larger generator (which sizes; licences), (b) contrastive prompts, (c) stance verification with regeneration, (d) M = 2–3 samples; accept that baselines are recomputed with the new generator | Approve the staged, gate-checked procedure; pre-screen 1.5B, 3B and 7B-4bit on validation | PD-3: flagged separate arms; v46 cache stays the default |
| D-5 | A7: apply the soundness-gated target (validator hunk of the v47 patch only) in Tier 2; may the `amp_entropy` metric (measurement only) be added in Tier 0? | Approve both | PD-2: A7 as an extra flagged arm, v46 target the default; PD-9: `amp_entropy` diagnostic only |
| D-6 | Separate LR group for phase and temperature scalars (peak 1.44e-3 to 2.96e-3 depending on budget) | Approve; it is a precondition for D-8 | PD-4: flagged separate arm |
| D-7 | C1: is the Born readout part of the claim to defend, or may a signed-readout arm be tested (and labelled additive scoring if it reduces to that)? | Test it as an arm; keep Born as the reference | PD-5: pre-registered arm; Born readout the default |
| D-8 | C2: approve the pre-registered ρ = π/2 vs π comparison | Approve with T2c | PD-5: pre-registered arm; ρ = π the default |
| D-9 | C3/C4/C5: risk–coverage reporting (no approval needed) now; pruning deferred; which encoder options to probe; C5 cleanup | Probe all three C4 options cheaply; train only those that pass; defer pruning | PD-5 for C3 (pruning not covered); PD-6: C4/C5 deferred |
| D-10 | Stopping rule: if no Tier 1 cache passes the gate, report D1 as the result instead of continuing architecture work | Agree | PD-13: as written (no gate pass → report "no signal", stop architecture work, write up the negative result). PD-10 is the separate training stopping rule |
| D-11 | Confirm the primary success criterion in §3 (arms, test, α, Holm over 2 contrasts) | Confirm before any test-set evaluation | PD-11: F0 beats both LLM baselines, full test, 3 seeds, paired bootstrap + Holm, p < 0.05, same direction in all 3 seeds, per-seed McNemar reported, v46 baseline alongside |
| D-12 | **Approve the path Hamiltonian arms (H0/H1)**, with H1-GS as secondary (Tier 3b; thermal readout P-2 already accepted) | Approve, to run only after a Tier 1 cache passes the gate | PD-5: pre-registered arms |
| D-13 | Approve arm **H2** (non-stoquastic X_iX_j couplings, sign unconstrained) with the **H2s** capacity control; XX only or XX + YY | Approve, XX only; H2s required for any interference claim | PD-5: pre-registered arms (H2 with H2s) |
| D-14 | Directed (a, b, r) pair head as the default, with the symmetric-only **H1-sym** ablation | Approve | PD-12: directed pair terms; symmetric-only head as ablation |
| D-15 | The 4-file integration for Tier 3b (`path_hamiltonian_design.md` §9) | Approve, behind a default-off flag | PD-5: behind the default-off flag |

---

## 3. Primary success criterion

**qAIR beats both direct LLM baselines on the full ARC-Easy test split across
3 seeds with a significant paired test.** Operationally (fixed before any test
evaluation; provisional decision PD-11 in
[`supervisor_decisions.md`](supervisor_decisions.md), pending supervisor review):

1. **Arms.** qAIR = the arm chosen in D-3 (proposed: F0; provisional decision
   PD-8: F0 primary, A3 secondary). Baselines = (a) LLM
   option likelihood (cloze) and (b) LLM letter pick, both **computed with the
   same generator that produced qAIR's cache and prior**.
2. **Data.** Full ARC-Easy test, 2376 items [VERIFIED: `config.py:58`].
   Checkpoint selected on validation; test evaluated once per seed.
3. **Seeds.** 42, 43, 44, each a full training run with its own manifest.
4. **Test.** For each baseline, the item-level paired bootstrap of the
   seed-averaged accuracy difference (resample items; average over seeds inside
   each resample). Both 95% CIs must lie above 0 after Holm correction over the
   2 contrasts. In addition, qAIR's accuracy exceeds each baseline's in **all 3
   seeds**, and per-seed exact McNemar p-values are reported.
5. **Power, stated in advance.** At n = 2376 and power 0.8, the smallest
   detectable difference is about 2.0–3.5 points, depending on the discordant
   rate (0.1–0.3 [ASSUMPTION]; measured in Tier 0) [DERIVED: A.3]. A true gain
   below this will likely not meet the criterion. That is a property of the
   dataset, not a reason to relax the test.
6. **Reporting.** The same table always shows A3, P0, P1, majority vote and
   both baselines. If only F0 meets the criterion and A3 does not, the claim is
   limited to "the system with the LLM prior", not "multi-hypothesis
   reasoning".

---

## Appendix A — how every derived number was computed

Computed 2026-10-18 with Python 3 standard library (no numpy/scipy). The code
is reproduced here so the numbers can be checked; it is not part of the
repository's code.

```python
import math
from math import cos, pi, acos, sqrt
from statistics import NormalDist

# A.1 -- Adam step budget under the actual schedule
steps = math.ceil(2251 / 16)                      # 141; BATCH_SIZE=16, no drop_last
def lr_e(base, e, E, eta=1e-6):                   # CosineAnnealingLR, stepped per epoch
    return eta + (base - eta) * (1 + cos(pi * e / E)) / 2
def budget(base, E, upto=None):                   # sum of lr over all steps
    return steps * sum(lr_e(base, e, E) for e in range(E if upto is None else upto))
rho_star = acos(-3 / 4)                           # 2.4189 ; rho_star - pi/2 = 0.8481
# budget(5e-4, 8/15/30) = 0.3177 / 0.5650 / 1.0948 ; budget(2e-4, ...) = 0.1274 / 0.2266 / 0.4391
# 0.848 / 5e-4 = 1696 steps = 12.03 epochs ; at E=30 the cumulative budget passes 0.848 at end of epoch 15
# scalar peak LR so that the first-half budget = 3 * 0.848:
#   3*0.848 / budget(1.0, E, E//2)  ->  E=15: 2.96e-3 ; E=30: 1.44e-3   (x1 margin: 9.88e-4 / 4.80e-4)
# weight-decay-only drift from pi/2, wd=0.05: -(pi/2)*(1-exp(-0.05*budget)) -> E=15: -0.044 ; E=30: -0.084

# A.2 -- gate: smallest k with exact (Clopper-Pearson) lower bound > 0.25, i.e. P(X >= k | n, 0.25) < 0.025
def upper_tail(k, n, p=0.25):
    lp = lambda j: (math.lgamma(n+1) - math.lgamma(j+1) - math.lgamma(n-j+1)
                    + j*math.log(p) + (n-j)*math.log(1-p))
    return sum(math.exp(lp(j)) for j in range(k, n + 1))
# n=500 -> 145 (0.2900) ; n=570 -> 164 (0.2877) ; n=2251 -> 604 (0.2683) ; n=2376 -> 637 (0.2681)

# A.3 -- minimum detectable paired difference, McNemar normal approximation, power 0.8
nd = NormalDist()
def mde(d, alpha, n=2376):
    return (nd.inv_cdf(1 - alpha/2) + nd.inv_cdf(0.8)) * sqrt(d / n)
# alpha=0.025: d=0.1 -> 0.0200 ; d=0.2 -> 0.0283 ; d=0.3 -> 0.0346
# alpha=0.05 : d=0.1 -> 0.0182 ; d=0.2 -> 0.0257 ; d=0.3 -> 0.0315

# Other: 1,056,382 / 2251 = 469.3 params/example ; 1/sqrt(2) = 0.707, 1/sqrt(3) = 0.577
# Costs: 12 runs x 8 epochs x 2.4-7 min = 3.8-11.2 h ; 6 runs = 1.9-5.6 h ;
#        (570+2376) x 0.88 s = 0.72 h ; 5197 x 0.88 s = 1.27 h ; 15/30 epochs x 2.4-7 min = 36-105 / 72-210 min
```
