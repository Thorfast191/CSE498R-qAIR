# HANDOFF

Builder: Claude Code · Reviewer: Claude on claude.ai (reads GitHub) · Runs
and decisions: the user. Updated at the end of every session (CLAUDE.md).

**Last updated:** 2026-10-29 · **PINNED_COMMIT = `2966775`** (Step 7; keep it for all five sessions)

## Canonical Tier 0 cache (ARC-Easy)
Commit `44fc12a`, session `20261028_135552_44fc12a`, gen_seed 0; manifest `runs/20261028_135552_44fc12a/cache_manifest.json`. sha256 prefixes: train (2251) `4bc0031289a03e4f`, validation (570) `0760535c445ad6dc`, test (2376) `5faa6d62c82612d6`. The Step 7 notebook checks all three before training.

## Done this session
| Item | Commit | Tests |
|---|---|---|
| E12 closed: T4 1.514 s/step (was 69.9 s), determinism IDENTICAL, torch vs `lightning.qubit` 1.9e-6 | `2966775` | |
| `scripts/step7_runs.py`, `main.py --cache-dir/--ckpt-dir`, `notebooks/cse498r_step7_baseline.ipynb`; local 5-session dry run incl. a kill + resume | `2966775` | **365 passed, 2 skipped** |
| Found E13: torch engine raises on Mac MPS (float64); CUDA unaffected | `2966775` | |

## Time estimate [DERIVED from 1.514 s/step; 141 steps + validation ≈ 3.6 min/epoch]
| Session | Runs | Training | With setup (~10 min [ASSUMPTION]) |
|---|---|---|---|
| 1 | F0 × seeds 0, 1, 2 | 1.1–3.6 h | **1.3–3.8 h** |
| 2 | A3 × 3 | 1.1–3.6 h | 1.3–3.8 h |
| 3 | P0 × 3 | 1.1–3.6 h | 1.3–3.8 h |
| 4 | P1 × 3 | 1.1–3.6 h | 1.3–3.8 h |
| 5 | baselines + stats | letter pick ≤ 23 min (upper bound: 2376 × 0.58 s) | ≤ 0.6 h |

Per run: 6 epochs (earliest early stop) = 22 min to 20 epochs = 72 min; test eval adds seconds. Total 5.6–15.8 h of T4. [HYPOTHESIS] A3/P0/P1 cost the same per step as the measured v46 config (same circuit calls); the 1.514 s mean covers only 2 steps.

## Decisions needed
None open. Approved 2026-10-29 (reviewer read `2966775`; user): the Step 7 runs, `RUNS_APPROVED = True` for all five sessions. E13 decided: on MPS run the circuit on the CPU; implemented after Step 7.

## Colab now — each session
1. Fresh **T4** runtime; open `notebooks/cse498r_step7_baseline.ipynb` fresh from GitHub `main`.
2. Cell 0: `PINNED_COMMIT = "2966775"`, `SESSION_NUMBER = 1` (then 2, 3, 4, 5), `RUNS_APPROVED = True`.
3. Sessions 1–4: run cells **0–6**. Session 5: cells **0–5, 7, 8**.
4. Cell 4 shows finished runs and `NEXT: session N`. After a disconnect, run 0–6 again with the same number: finished runs are skipped, an interrupted run resumes from its last epoch.
5. Stop and paste to Claude Code if cell 3, 5 or 6 stops (e.g. `CheckpointMismatchError`, strict determinism); leave the checkpoint files where they are.
6. Paste after each session: cell 4's status table. After session 5: everything from `RESULTS TABLE` down.
