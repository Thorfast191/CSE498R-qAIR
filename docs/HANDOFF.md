# HANDOFF

Builder: Claude Code · Reviewer: Claude on claude.ai (reads GitHub) · Runs
and decisions: the user. Updated at the end of every session (CLAUDE.md).

**Last updated:** 2026-10-28 · code at the commit after `10f37d6` (this file's commit)

## Canonical Tier 0 cache (ARC-Easy)
Built at commit `44fc12a`, session `20261028_135552_44fc12a`, gen_seed 0. Drive: `qAIR_CSE498R/cache/arc_easy/`. Manifest: `runs/20261028_135552_44fc12a/cache_manifest.json`.

| Split | n | complete | sha256 (prefix) |
|---|---|---|---|
| train | 2251 | True | `4bc0031289a03e4f` |
| validation | 570 | True | `0760535c445ad6dc` |
| test | 2376 | True | `5faa6d62c82612d6` (never gated) |

**Gate finding** (train / val; dossier D1, D5): support 0.335 / 0.330 is above chance (z +9.3 / +4.4). Attack 0.180 / 0.172 is **below** chance (z −7.6 / −4.3), with implied η ≈ 0.9. The margin 0.255 / 0.251 is **at chance**, and validation's 143/570 is below the Tier 1 gate threshold of 164. LLM likelihood is 0.572 / 0.583.

## Done this session
| Item | Commit | Tests |
|---|---|---|
| E12 profile; weight-change floor; quantum check records devices and timing | `9b9c302` | 284 passed |
| B4, B5, B7 | `62a1800` | real-Qwen slow test passed |
| B3 | `63156e2` | 304 passed, 2 skipped |
| Gate finding recorded (dossier D1/D5, issues, plan, log); `scripts/gate_significance.py` | this commit | 308 passed, 2 skipped |

## Training speed (E12, the Step 7 blocker): fix ready, waiting for approval
- **Cause:** on CPU, 98 % of a step is PennyLane running 512 per-row circuits; the rest of the model takes 19 ms.
- **Fix:** the torch statevector is implemented and tested, **uncommitted** (local worktree; the patch applies cleanly to main).
- **Accuracy:** it matches PennyLane ≤ 1e-6 (float64) and ≤ 1e-5 (float32) on outputs and gradients, and across all 16 arms.
- **Speed (CPU):** 0.033 s vs 0.75 s per step.
- **T4:** to be measured after the commit; estimate 0.05–0.15 s/step [HYPOTHESIS], versus 69.9 s measured.

## Decisions needed
1. **E12: approve the commit** with `QUANTUM_SIMULATOR = "torch"` as the default (recommended). I can push the review branch `e12-torch-statevector` first if the reviewer wants to read it.
2. **B6 statistics** (spec in the previous HANDOFF, `10f37d6`): McNemar per seed; item-level paired bootstrap averaging the 3 seeds; Holm over {v46, F0, P0, P1} × {option likelihood, letter pick}; PD-11's 3-seed rule reported alongside.
3. **Tier 1 order:** move T1c (attack verification) first? This changes the pre-registered staging (PD-3). Recommended, given the finding.

## Next step
After decision 1: commit E12, then a T4 smoke timing at the new commit. After decision 2: B6. Then the Step 7 Colab section, with a measured time estimate.

## Colab (now)
Nothing to run until E12 is committed. Then: a fresh runtime, `PINNED_COMMIT` = the E12 commit, run cells 0–7, and paste cell 4's timing and cell 7's two verdict lines.
