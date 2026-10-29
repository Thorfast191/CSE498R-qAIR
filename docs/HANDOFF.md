# HANDOFF

Builder: Claude Code · Reviewer: Claude on claude.ai (reads GitHub) · Runs
and decisions: the user. Updated at the end of every session (CLAUDE.md).

**Last updated:** 2026-10-30 · Step 7 / Tier 0 **closed** at `2966775` · no Colab run pending

## Result (full report: `docs/baseline/baseline_results.md`)
Test n = 2376, 3 seeds: F0 0.5824 ± 0.0080, A3 0.4364, P0 0.4231, P1 0.4297; letter pick **0.6953** (the bar for every later claim), option likelihood 0.5976, B5 vote 0.2618. **PD-11 not met**: F0 − option likelihood −0.0153 (Holm p 0.0695), F0 − letter pick −0.1129 (Holm p 0.0002).

## Done this session
| Item | Commit |
|---|---|
| `baseline_results.md` (Observation / Interpretation / Recommendation / Next Experiment; pre-registered predictions held / failed) | `4d68936` |
| issues.md: Tier 0 close-out; B1, B2, B3 closed, B8 budget part closed, B10 closed; research log entry | `4d68936` |

No code changed; suite unchanged at 365 passed, 2 skipped.

## Decisions needed
1. **Approve the validation-only input ablation** (proposal in the session reply; summary): 6 best checkpoints (A3, F0 × seeds 0/1/2); corruptions H := zeros, H from another question (existing "H rolled across batch"), attack := 0, support := 0, **Q := zeros (new)**, plus the existing cheap ones (noise, H := O, llm_logprob := 0, O := 0, polarity, align); paired item-bootstrap CI of each delta, seeds averaged; `--split test` refused in code. Eval only, ≈ 15 min on Colab including setup [DERIVED / ASSUMPTION]. Recommendation: approve.
2. E13 (decided: CPU circuit on MPS), after the ablation.

## Next step
After approval: implement and test on CPU, add a cell to the Step 7 notebook (reads `step7_2966775/ckpt`), give you the new PINNED_COMMIT. Then T1c (PD-15).

## Colab
Nothing to run now.
