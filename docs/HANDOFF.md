# HANDOFF

Builder: Claude Code · Reviewer: Claude on claude.ai (reads GitHub) · Runs
and decisions: the user. Updated at the end of every session (CLAUDE.md).

**Last updated:** 2026-10-28 · **PINNED_COMMIT = `2f30e39`** (contains the E12 merge `b548095` plus the resume follow-up)

## Canonical Tier 0 cache (ARC-Easy)
Built at commit `44fc12a`, session `20261028_135552_44fc12a`, gen_seed 0. Manifest: `runs/20261028_135552_44fc12a/cache_manifest.json`. sha256 prefixes: train (2251) `4bc0031289a03e4f`, validation (570) `0760535c445ad6dc`, test (2376) `5faa6d62c82612d6` (never gated). Unchanged by E12: the engine affects training only, not the cache.

## Done this session
| Item | Commit | Tests |
|---|---|---|
| E12 merged into `baseline` (merge commit) and pushed to `cse498r/main` | `b548095` | 350 passed, 2 skipped |
| Follow-up: `quantum_simulator` compared on resume and in the manifest; cell 7 gets a torch-engine verdict line | `2f30e39` | **352 passed, 2 skipped** |

## Decisions needed
None open. After the smoke timing I'll bring the Step 7 run plan and its time estimate for your go-ahead.

## Next step
1. You run the T4 smoke timing below.
2. I estimate the time of the Step 7 runs from it and write the Step 7 Colab section: v46/A3, F0, P0, P1 × seeds 0/1/2 plus the LLM option likelihood, letter pick and vote baselines on test, then `stats.py` with the PD-14 families.

## Colab (now): T4 smoke timing, cells 0–7
1. Runtime → Disconnect and delete runtime; connect a fresh **T4**.
2. Open `notebooks/cse498r_baseline.ipynb` fresh from GitHub `main` (it changed in `2f30e39`; don't reuse an open copy).
3. In cell 0, set `PINNED_COMMIT = "2f30e39"` and keep `SECTION_B_APPROVED = False`.
4. Run cells **0, 1, 2, 3**. Cell 3 should end with `torch pinned (2.11.0+cu130) and installed; continue`. If it stops on "packages qAIR needs", paste that and stop.
5. Run cells **4, 5, 6, 7**:
   - Cell 4 builds a new 20 + 10-question smoke cache in `runs/<session>/smoke/`; it doesn't touch `cache/arc_easy/`.
   - Cell 5 is the determinism pair, now on the torch engine.
   - Cell 6 compares and times `lightning.gpu`, `lightning.qubit` and the torch engine.
6. Don't run cells 8–10; the cache is already built.
7. Paste:
   - cell 3's `ENVIRONMENT` block;
   - cell 4's `SMOKE TIMING` block (per stage, train step, eval/question, [DERIVED] projections);
   - cell 7's three lines: `determinism:`, `lightning.gpu:`, `torch statevector:`.
