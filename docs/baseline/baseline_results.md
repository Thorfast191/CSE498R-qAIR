# Tier 0 baseline results — v46 on full ARC-Easy (Step 7)

**Source.** `results_table.txt` of session `runs/20261029_193129_2966775_step7s5`
(Drive), commit `2966775`, run by the user on a Colab T4 (torch 2.11.0+cu130,
CUDA 13.0, pennylane 0.42.3, lightning 0.42.0, pins ok, torch circuit engine).
Every number below comes from that file unless labelled otherwise.

**Data.** Canonical Tier 0 cache, hashes verified in the same session: train
`4bc0031289a03e4f…` (2251), validation `0760535c445ad6dc…` (570), test
`5faa6d62c82612d6…` (2376), gen_seed 0.

**Protocol.** Arms A3 (v46 reasoning arm), F0 (A3 + LLM prior), P0, P1; seeds
0, 1, 2; at most 20 epochs, early stopping on validation accuracy with patience
5 (PD-7 / PD-10), deterministic. Test was evaluated once per run, from the
checkpoint with the best validation accuracy (B3). The protocol was not
changed during Step 7.

*Deviation from the plan, fixed before the runs:* seeds 0, 1, 2 (the Step 7
instruction) instead of 42, 43, 44 (`improvement_plan.md` §0 rule 3, §3 item 3).
It was chosen before any run and is reported here, not hidden.

---

## Observation

### Test accuracy (n = 2376)

| Arm / baseline | Test acc (mean ± std, 3 seeds) | 95% CI [DERIVED] | Best val acc (mean ± std) | pw cos | phase | destr |
|---|---|---|---|---|---|---|
| **F0** full, with LLM prior | 0.5824 ± 0.0080 | [0.5626, 0.6022] | 0.5632 ± 0.0066 | 0.9149 | 0.3603 | 0.0001 |
| A3 persistent (v46) | 0.4364 ± 0.0078 | [0.4165, 0.4563] | 0.4380 ± 0.0081 | 0.5436 | 0.9871 | 1.0000 |
| P0 classical collapse | 0.4231 ± 0.0112 | [0.4032, 0.4430] | 0.4351 ± 0.0043 | 0.6105 | 0.2475 | 0.0000 |
| P1 phase zero | 0.4297 ± 0.0030 | [0.4098, 0.4496] | 0.4374 ± 0.0086 | 0.6294 | 0.0000 | 0.0000 |
| LLM letter pick (B4) | **0.6953** (1652/2376) | [0.6768, 0.7138] | — | | | |
| LLM option likelihood | 0.5976 (1420/2376) | [0.5779, 0.6174] | — | | | |
| Evidence-weighted vote (B5) | 0.2618 (622/2376) | [0.2441, 0.2795] | — | | | |

- The CI column is $p \pm 1.96\sqrt{p(1-p)/n}$ at n = 2376 on the
  seed-mean (arms) or on the deterministic baseline [DERIVED; item sampling
  only, not the item-level bootstrap]. The paired CIs below are the
  pre-registered ones.
- std is the population std over the 3 seeds (`training/ablations.py`
  `summarize`). pw cos, phase, destr are test-split means over seeds of
  `pairwise_cos`, `phase_effect` and `destructive_fraction`.

### PRIMARY family (PD-11, PD-14): Holm over 2

Item-level paired bootstrap of the seed-averaged difference (10,000
resamples), per-seed exact McNemar.

| Contrast | diff | 95% CI | p | p Holm | higher in all seeds | McNemar p s0 / s1 / s2 |
|---|---|---|---|---|---|---|
| F0 − LLM option likelihood | −0.0153 | [−0.0318, +0.0013] | 0.0695 | 0.0695 | no | 0.0434 / 0.0285 / 0.695 |
| F0 − LLM letter pick | −0.1129 | [−0.1357, −0.0902] | 0.0001 | 0.0002 | no | 1.66e-19 / 3.32e-20 / 4.35e-15 |

**PD-11 success criterion: NOT MET.** F0 is below both LLM baselines. Its
difference from the option likelihood is not significant (the CI includes 0);
it is 11.3 points below the letter pick (significant, in the wrong direction).

### SECONDARY family (PD-14): Holm over 6, reported only

| Contrast | diff | 95% CI | p Holm |
|---|---|---|---|
| A3 − option likelihood | −0.1612 | [−0.1843, −0.1373] | 0.0006 |
| A3 − letter pick | −0.2588 | [−0.2818, −0.2367] | 0.0006 |
| P0 − option likelihood | −0.1745 | [−0.1982, −0.1503] | 0.0006 |
| P0 − letter pick | −0.2722 | [−0.2945, −0.2506] | 0.0006 |
| P1 − option likelihood | −0.1679 | [−0.1916, −0.1437] | 0.0006 |
| P1 − letter pick | −0.2656 | [−0.2877, −0.2433] | 0.0006 |

Every arm is below both LLM baselines on the seed average; all per-seed
McNemar p ≤ 9.2e-28. (Per-seed accuracies are not in the results table.)

### Exploratory, uncorrected: against A3

| Contrast | diff | 95% CI | p | McNemar p s0 / s1 / s2 |
|---|---|---|---|---|
| F0 − A3 | +0.1459 | [+0.1261, +0.1655] | 0.0001 | 3.16e-31 / 1.33e-24 / 6.13e-30 |
| P0 − A3 | −0.0133 | [−0.0265, +0.0000] | 0.0514 | 0.849 / 0.00659 / 0.613 |
| P1 − A3 | −0.0067 | [−0.0198, +0.0066] | 0.3189 | 0.854 / 0.536 / 0.168 |

### Training dynamics (Tier 2 findings; reported by the user from the Step 7 logs, not in `results_table.txt`)

- Every F0 seed selected epoch 1 and early-stopped at epoch 6.
- Train loss fell while validation accuracy fell.
- Gradient clipping hit 100 % of steps from epoch 3.
- F0's selected checkpoints have `destructive_fraction` ≈ 0 (test mean 0.0001):
  its phases are inert. A3's is 1.0000: saturated, the C2 signature.

None of these led to a protocol change during Step 7.

---

## Interpretation

1. **The trained system does not beat its own generator.** F0 (0.5824) is
   not distinguishable from the generator's option likelihood alone (0.5976;
   paired diff −0.0153, CI [−0.0318, +0.0013]) and is clearly below the letter
   pick (0.6953). F0's prior feature is that same option likelihood: the cached
   `llm_logprob`, whose argmax is the baseline [VERIFIED:
   `evaluation/baselines.py` `run_cached_llm_baseline`]. So F0's
   accuracy is consistent with "the prior, slightly diluted"
   [HYPOTHESIS: the input ablation's `llm_logprob := zeros` would test it].
2. **The prior carries most of F0.** F0 − A3 = +0.1459 (exploratory). Under
   plan §3 item 6, any statement about F0 is about "the system with the LLM
   prior", and here it does not reach its own prior.
3. **F0 learns nothing useful after epoch 1.** Best validation at epoch 1 in
   every seed, with train loss falling and validation falling after it, means
   that more training moves F0 away from the prior-dominated starting point,
   and the validation-selected checkpoint is close to that starting point
   [HYPOTHESIS: overfitting the training objective, or an objective that does
   not reward accuracy (A7 / T2a); not separated by these data]. Clipping on
   100 % of steps from epoch 3 means the updates are norm-limited throughout
   most of training. Whether this is cause or symptom is not known
   [HYPOTHESIS].
4. **A3 is far above chance without the prior, and we do not yet know why.**
   A3 = 0.4364 against a 4-option chance near 0.25 [ASSUMPTION: test is
   mostly 4-option, like train / validation, chance 0.2501 / 0.2500]. The
   parameter-free vote over the same evidence is at chance (0.2618). So A3's
   signal comes from learned use of something: the support channel (which
   carries signal on train / validation, gate 2026-10-28), the attack channel
   used with its sign learned, or the question / option embeddings alone
   [HYPOTHESIS; all three open]. The proposed input ablation separates them.
5. **No evidence for or against interference.** P0 − A3 = −0.0133
   (p 0.051) and P1 − A3 = −0.0067 (p 0.32) are exploratory and uncorrected,
   and under the B2 rule (withdraw arm differences smaller than ~2× the
   per-arm std: 2 × 0.0112 and 2 × 0.0078) they are withdrawn. In addition, the
   comparison is not a fair test of interference: A3's `destructive_fraction`
   is saturated at 1.0 (C2), so A3 vs P0 / P1 compares saturated
   interference with none, not learned interference with none.
6. **The B5 vote matches the gate finding.** The margin statistic was at
   chance on train / validation (0.2554 / 0.2509, recorded 2026-10-28 before
   the test run); the vote on test is 0.2618. The hypotheses' combined
   support − attack evidence, unweighted, does not answer questions.

### Pre-registered predictions: held or failed

| Prediction (source) | Result |
|---|---|
| PD-11 / plan §3: F0 beats both LLM baselines (Holm CI > 0, higher in all 3 seeds) | **Failed.** Both differences negative. |
| B1 (dossier): test CI half-width ≤ 0.021 for every arm | **Held** [DERIVED]: 0.0198–0.0199 per arm. (Paired-difference CIs are wider, up to ±0.024; a different quantity.) |
| B2 (dossier): per-arm std reported; arm differences < ~2× std withdrawn | **Held / applied.** P0 − A3 and P1 − A3 withdrawn (interpretation 5). |
| B3 (dossier): mean (best-val − test) across arms and seeds ≥ 0 | **Held narrowly** on the arm means: +0.0005 [DERIVED from the table; per-seed values not in it]. F0 alone is −0.0192. |
| B5 (dossier): if D1 holds on ARC-Easy, the vote is within ±0.04 of chance (validation) | **Held** on validation (margin 0.2509) and on test (0.2618). |
| D1 (dossier): the margin exceeds chance at n = 570 | **Failed** on validation (2026-10-28, before this run). By its own consequence, any qAIR gain had to come from the prior; F0 did not beat the prior. |
| Tier 0: "expected accuracy change: none" vs the capstone 0.50 | **Not testable:** the capstone was ARC-Challenge, n = 20 validation; Tier 0 is ARC-Easy test. |
| Tier 0 risk (i): the baseline may sit at or below the generator-only baselines | **Realised.** A valid result (DoD: working ≠ winning). |
| B8: `polarity_phase` moves ≫ 0.01 rad, or its gradient stays ≈ 0 with alternating sign | **Not evaluated here** (per-run histories on Drive, not read). |
| B11: ECE vs the ≈ 0.05 noise floor | **Not evaluated here** (not in the results table). |
| Plan §3 item 5: discordant rate measured in Tier 0 | **Not computed yet** (needs the per-item records). |

---

## Recommendation

1. **Tier 0 is closed with PD-11 not met.** Report it as the result: on full
   ARC-Easy, v46 trained under the pre-registered protocol is below both
   generator-only baselines. Do not re-tune or re-run Tier 0 on these items
   (plan §0 rule 6).
2. **The letter pick, 0.6953 on test, is the bar for every later claim.** A
   later tier that does not beat it, under the same paired test, has not
   shown that the architecture adds anything to its generator.
3. **Do not read A3 vs P0 / P1 as evidence about interference** while A3's
   `destructive_fraction` is saturated (C2). That needs the supervisor's C2
   discussion first.
4. **Training dynamics** (F0 best at epoch 1, falling validation, 100 %
   clipping) go to Tier 2 (T2a objective, T2b budget, T2c scalar LR) as
   observations to predict against. No change to the Tier 0 protocol.

## Next Experiment

1. **Validation-only input ablation** of the 6 trained A3 and F0 best
   checkpoints (proposed separately, awaiting approval): does A3's
   accuracy come from the hypotheses or from the question / option
   embeddings? The test split is not used again.
2. **T1c attack verification** (PD-15 order T1c → T1a → T1b → T1d), informed by
   the ablation: if A3 does not use the hypotheses, Tier 1 evidence fixes are
   predicted to change nothing until the aggregator uses them.
3. **Still open:** per-seed records → discordant rate (plan §3 item 5), B8
   drift and B11 calibration from the run histories, D5 attack audit (100
   attacks).
