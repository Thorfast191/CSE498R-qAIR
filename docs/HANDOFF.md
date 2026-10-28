# HANDOFF

Builder: Claude Code · Reviewer: Claude on claude.ai (reads GitHub) · Runs
and decisions: the user. Updated at the end of every session (CLAUDE.md).

**Last updated:** 2026-10-28 · code at `63156e2` · Section B keeps running at **`44fc12a`** (no cache-related file changed since: `git diff 44fc12a..63156e2` of dataset, generator, encoder, build/manifest scripts, config, requirements is empty)

## Done this session
| Item | Commit | Tests |
|---|---|---|
| E12 profile (issues.md E12); weight-change floor `max(‖W0‖, 1e-3)`; quantum check records both QNode devices and seconds per pass | `9b9c302` | 284 passed |
| B4 letter pick, B5 evidence-weighted vote (pre-registered weighting), B7 probe epochs on a train holdout, `--out-dir` records | `62a1800` | + real-Qwen slow test passed |
| B3 test only from the validation-selected checkpoint, `test_from` recorded | `63156e2` | **304 passed, 2 skipped** (torch 2.11.0 CPU) |

**E12 finding (CPU, M4):** a step is 0.86 s; 98 % is PennyLane running 512 per-row tapes (2 circuit calls × 128 rows, adjoint). The rest of the model takes 19 ms. Why the T4 is 80× slower is not measured (hypotheses in E12).
**Lightning 0.0 diff:** the reference really is `lightning.qubit` (explicit, no fallback). TorchLayer returns float32, so float64-level differences round away.

## Decisions needed
1. **E12 torch statevector: approve to commit?** It's implemented and tested, **uncommitted**, in a local worktree (patch: 10 files, applies cleanly).
   - **Results:** outputs and gradients match PennyLane ≤ 1e-6 in float64 and ≤ 1e-5 in float32. All 16 arms' full forward match. A0 passes under both simulators. Same-seed weights are identical, so checkpoints are interchangeable. The v46 bit-identical forward tests still pass. 8 math mutations each fail.
   - **Speed (CPU):** 0.033 s vs 0.75 s per step; 0.008 vs 0.66 s per eval batch.
   - **T4 estimate [HYPOTHESIS]:** 0.05–0.15 s/step, i.e. minutes per 20-epoch run.
   - Recommendation: approve, with `QUANTUM_SIMULATOR = "torch"` as the default and `"pennylane"` as the reference. Optional: let me push the branch `e12-torch-statevector` (not `main`) so the reviewer can read it first.
2. **B6 statistics (needs approval before I commit it).** Proposal:
   - McNemar (exact) per seed.
   - Paired bootstrap over items, averaging each item over the 3 seeds inside each resample (10,000 resamples, seed 0).
   - Holm over the pre-registered family: {v46/A3, F0, P0, P1} × {LLM option likelihood, LLM letter pick} = 8 tests.
   - PD-11's same-direction-in-all-3-seeds rule reported alongside.
   - A CPU simulation test confirms the pooled test over-rejects and the new one stays near 5 %.
3. **Torch pin rule for CPU builds** (from the last session): keep it (recommended).

## Next step
I write B6 once you approve it, then the Step 7 Colab section. Its T4 time estimate comes from a new smoke timing at the E12 commit.

## Colab (now)
1. Let Section B (cell 8) finish at `44fc12a`. Then run cells 9 and 10.
2. Paste: the cell 9 output (train and validation quality blocks plus sizes) and the cell 10 lines.
3. If you approve E12: after I commit it, rerun cells 0–7 at the new commit (fresh runtime), then paste cell 4's timing and cell 6's lines.
