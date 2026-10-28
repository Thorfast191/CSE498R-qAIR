# HANDOFF

Builder: Claude Code · Reviewer: Claude on claude.ai (reads GitHub) · Runs
and decisions: the user. Updated at the end of every session (CLAUDE.md).

**Last updated:** 2026-10-28 · main (`baseline` → `cse498r/main`) at `6f28f31` · **review branch `e12-torch-statevector` at `c0c6b2d`** (not merged)

## For the reviewer: E12 branch
`https://github.com/Thorfast191/CSE498R-qAIR/tree/e12-torch-statevector`, two commits on top of `81015be`:
- **`7960dc5` (code):**
  - New `models/statevector.py`.
  - `config.QUANTUM_SIMULATOR = "torch"` is the default; `"pennylane"` stays as the reference and fallback.
  - The quantum layer gains the switch, and `t4_checks` compares and times both engines.
  - Tests: `tests/test_torch_statevector.py` (outputs and gradients, float64 ≤ 1e-6 and float32 ≤ 1e-5, all 16 arms) and A0 under both engines, plus the determinism and v46 bit-identical forward tests on the default.
- **`c0c6b2d` (docs):**
  - README and dossier: both engines are exact classical simulations of the same circuit.
  - issues.md E12: the CPU-vs-T4 finding.
- **Branch suite:** 339 passed, 2 skipped.

## Canonical Tier 0 cache (ARC-Easy)
Built at commit `44fc12a`, session `20261028_135552_44fc12a`, gen_seed 0. Manifest: `runs/20261028_135552_44fc12a/cache_manifest.json`.

| Split | n | sha256 (prefix) |
|---|---|---|
| train | 2251 | `4bc0031289a03e4f` |
| validation | 570 | `0760535c445ad6dc` |
| test | 2376 | `5faa6d62c82612d6` (never gated) |

## Done this session
| Item | Commit | Tests |
|---|---|---|
| E12 pushed as a review branch | `7960dc5`, `c0c6b2d` (branch) | 339 passed (branch) |
| B6 (PD-14 families); PD-14, PD-15 recorded; B5 note (support-only = exploratory) | `6f28f31` | 319 passed, 2 skipped (main) |

## Decisions needed
1. **Merge E12** once the reviewer has read the branch. I'll merge it into `baseline` with a merge commit (no rebase of the pushed branch, no force).

## Next step
After the merge:
1. You run a T4 smoke timing (cells 0–7) at the merge commit, which measures the real step time.
2. I write the Step 7 Colab section (v46/A3, F0, P0, P1 × seeds 0/1/2 plus the baselines on test, then `stats.py` with the PD-14 families) with a T4 time estimate from that smoke run, and ask before you run it.

## Colab (now)
Nothing until the merge. Then use a fresh runtime with `PINNED_COMMIT` set to the merge commit, run cells 0–7, and paste cell 4's timing, cell 6's lines and cell 7's two verdict lines.
