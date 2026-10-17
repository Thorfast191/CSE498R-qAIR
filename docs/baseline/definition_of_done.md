# Definition of done — "baseline working" (PROPOSAL, to confirm with supervisor)

**"Working" means correct and honestly measured.** Every stage does what it
claims, the full pipeline runs on full splits, and every number has a seed, a
CI and a paired comparison against simple baselines on the same items.
**Beating the baselines is the research question, not a requirement.** A
correctly measured result where qAIR ties or loses to majority vote or to the
generator alone still counts as a working baseline.

Every criterion below is pass/fail and comes with the command or artifact that
shows it. Issue IDs refer to `docs/baseline/issues.md`.

## 1. Runs end-to-end on full splits, reproducibly

| # | Criterion | Evidence required |
|---|---|---|
| 1.1 | Code is the v46 capstone (`9552498`, tag `capstone-final`) plus only approved fixes, on branch `baseline`, committed and clean. The v47 patch is not applied unless the supervisor approves A7 | `git status` clean; commit hash in the run manifest (E1, E2) |
| 1.2 | Cache built for **full** ARC-Easy train / validation / test (2251 / 570 / 2376), and logged sample counts equal the split sizes minus documented filtered items. Built only after E3 and E7 are fixed | `[CACHE SAVED] … complete=True` per split; counts in the manifest (B1) |
| 1.3 | Cache is reproducible: rebuilding 20 questions with the same seed gives byte-identical hypotheses | hash comparison (E3) |
| 1.4 | Training + test evaluation finishes without error for every arm in §3 × 3 seeds | one manifest per (arm, seed): commit, cache hash, config, seed, device, package versions (E6) |
| 1.5 | Same seed, same device ⇒ same validation accuracy and same per-example predictions (run twice, smoke size ≤ 20 examples) | diff of two prediction files is empty |
| 1.6 | Resume is correct: an interrupted run resumed from `latest` gives the same history length and final metrics as an uninterrupted one; a different seed never resumes another seed's checkpoint | smoke test (A3, A4) |
| 1.7 | No checkpoints or data committed | `git ls-files ckpt cache` lists only `.gitkeep` (E4) |

## 2. Each stage verified to do what it claims

Written as small tests under `tests/`, plus a diagnostics log for every run.

| # | Stage | Pass criterion |
|---|---|---|
| 2.1 | Quantum circuit | Output std across ≥ 32 distinct random inputs is > 1e-3 per measured observable. Already measured once by a scratch test (issues.md A0): v46 min std 3.6e-02; v40 ≈ 1e-16, so the test catches the bug. Must exist as a committed regression test. |
| 2.2 | Gradient coverage | After one backward pass on a real batch, **every** parameter in the optimizer has a finite, non-zero `grad` for that arm's config. Unused parameters are either removed from the optimizer or listed explicitly with a reason (A1, A2). Per-module grad norms are logged every epoch. |
| 2.3 | Collapse correctness | For a hand-built batch: probabilities over valid options sum to 1 (±1e-5); padded options get ≤ 1e-12 mass; `phase_mode="classical"` equals Σ_k \|c_k\|²\|v_kn\|² computed by hand; `"zero"` equals \|Σ_k c_k v_kn\|²; with phases at 0/π the coherent sum equals (S−T)² (C1 documented, not changed). |
| 2.4 | Hypothesis dependence | `evaluation/input_ablation.py` is run with the correct per-checkpoint config (A5 fixed), and reports per-channel deltas with bootstrap CIs. **Reporting it is the criterion**; a FAIL verdict is a finding, not a blocker. |
| 2.5 | No label leakage | `y` reaches only the validator BCE target. A test asserts that predictions are identical whether `y` is passed or not, in eval mode. |
| 2.6 | Option shuffling | A test checks that `collate_fn` with shuffling preserves (option, hypothesis, label) alignment. |
| 2.7 | Parameter drift | Drift of every scalar parameter (phases, temperatures, gates) is logged per epoch (B8, C2). |

## 3. Simple baselines on the same split and items

Every baseline emits per-example records in the same item order as qAIR, so
paired tests are possible.

**Mandatory** (decided 2026-10-17):

| Baseline | Definition | Exists in v46? |
|---|---|---|
| LLM option likelihood | argmax of length-normalized log P(option \| question) (cloze) | Yes: `evaluation/baselines.py --mode llm` |
| LLM letter pick | Qwen sees the question + lettered options; argmax over the letter-token log-probs; deterministic | **No** (B4) |
| Majority vote over hypotheses | support votes +1 for its option, attack −1; argmax with a fixed tie-break | **No** (B5) |
| P0 — classical collapse | `P0_classical_collapse`: decohered mixture Σ\|c\|²\|v\|² in place of the coherent sum | Yes |
| P1 — phase-ablated | `P1_phase_zero`: phases pinned to 0 | Yes |

These are compared against the qAIR arm under test. [ASSUMPTION — confirm]
That is `A3_persistent` (reasoning result, no LLM prior) and
`F0_full_with_llm_prior` (deployed system).

**Optional:** literal softmax arm (Born collapse replaced by a softmax; a
C-level change, so it needs approval before implementation); parameter-matched
classical aggregator (`A1c` / `A3c` exist); chance / majority-class position.

## 4. Statistics

| # | Criterion |
|---|---|
| 4.1 | ≥ 3 seeds per arm (proposed 42, 43, 44); report mean ± std of test accuracy |
| 4.2 | 95% CI from a paired **item-level** bootstrap (resample items; within a resample, average over seeds) — not pooled (item, seed) cells (B6) |
| 4.3 | Paired test vs each baseline: McNemar per seed **and** the paired bootstrap of the accuracy difference; Holm correction across the pre-registered contrasts |
| 4.4 | Pre-registered contrasts (fixed before running): qAIR vs LLM option likelihood, vs LLM letter pick, vs majority vote, A3 vs P0, A3 vs P1, F0 vs A3. Optional: A1b vs A1c, vs literal softmax |
| 4.5 | Checkpoint selected on validation; test evaluated once per (arm, seed); nothing tuned on test (B3) |
| 4.6 | Minimum detectable effect stated. [ASSUMPTION] At ARC-Easy's 2376 test items and accuracy near 0.5, one-run SE ≈ √(0.25/2376) ≈ 1.0 point, so differences < ~2–3 points are not claimable. |

## 5. Deliverables

- `docs/research_log.md` entry per run (config, seed, dataset, split, commit).
- One results table: arms × {mean ± std test acc, 95% CI, p vs each baseline}.
- The §2 diagnostics for the main arm: collapse entropy / log K, coherent_acc vs
  classical_acc, drift, grad norms, input-ablation deltas.

## Explicitly not required for "done"

- qAIR beating any baseline.
- Hypotheses showing above-chance signal (D1 is a research finding).
- Any C-category change (C1–C5); these are discussed separately.

## Decided (2026-10-17)

- Baseline code: v46 (`9552498`, tag `capstone-final`).
- Baseline dataset: ARC-Easy; cache built only after E3 and E7 are fixed.
- Mandatory baselines: LLM option likelihood, LLM letter pick, majority vote,
  P0, P1. Literal softmax arm optional.

## Still to confirm with the supervisor

1. The criteria above (thresholds such as 2.1's 1e-3; 3 seeds).
2. Which qAIR arms are measured against the baselines (proposal: A3 and F0).
3. A7: whether the v47 validator-target change may be applied (it changes the
   training objective).
4. C1–C5 (issues.md).
