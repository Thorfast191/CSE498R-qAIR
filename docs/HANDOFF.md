# HANDOFF

Builder: Claude Code · Reviewer: Claude on claude.ai (reads GitHub) · Runs
and decisions: the user. Updated at the end of every session (CLAUDE.md).

**Last updated:** 2026-10-28 · **PINNED_COMMIT = `44fc12a`** (this file is committed after it)

## Done this session
| Item | Commit | Tests |
|---|---|---|
| Approval policy, HANDOFF; `main.py --train-samples/--val-samples` | `ff290dc`, `05a48bf` | 256 passed |
| Step 6 notebook, `t4_checks.py`, `cache_manifest.py` | `9974664` | 268 passed |
| Torch pin `2.11.0+cu130` (CPU build of 2.11.0 passes); no test-split quality block; `lightning.gpu` probed before use, fallback reason in manifest; cell 3 classifies every `pip check` complaint; cell 7 verdict lines | `44fc12a` | **283 passed, 1 skipped on torch 2.11.0 CPU** |

Expected Colab conflicts [DERIVED, simulated from your freeze + PyPI]: our pins downgrade transformers, hub, accelerate, datasets, tokenizers, fsspec; then gcsfs, gradio-client, bigframes, hf-gradio, dopamine-rl are unused by qAIR and get uninstalled; ipython's missing `jedi` was already there (kept). sentence-transformers 5.7.0 already accepts our pins and qAIR doesn't import it, so it isn't pinned. Details in issues.md E11.

## Decisions needed
1. **Torch pin rule for CPU builds** (`training/environment.py matches_pin`): a CUDA build must equal `2.11.0+cu130` exactly; a CPU-only build passes on `2.11.0`. Without this rule, no CPU machine can run the tests. Recommendation: keep it.

## Next step
You rerun cells 0–3 at `44fc12a`, then Section A (4–7), and paste me cell 3's ENVIRONMENT block and cell 7's last lines. Meanwhile I start Step 7 (B4, B5, B3/B7; B6 waits for approval).

## Colab (now)
1. Runtime → Disconnect and delete runtime (cell 3 already ran once in the old one). Then connect a fresh T4.
2. Cell 0: `PINNED_COMMIT = "44fc12a"`, `SECTION_B_APPROVED = False`.
3. Run cells 0–7. If cell 3 stops on "packages qAIR needs", paste that and stop.
4. Paste: the `ENVIRONMENT` block (cell 3), plus the `determinism:` and `lightning.gpu:` lines and the JSON (cell 7).
