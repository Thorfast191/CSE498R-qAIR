# qAIR CSE498R — research log

One entry per fix or run: date, item, commit, environment, what was run and
what came out. Commit hashes are filled in by the follow-up entry when the
fix commit cannot name itself.

---

## 2026-10-18 — E3: seeded, reproducible hypothesis generation

- **Branch / base:** `baseline`, on top of `d0a074a`.
- **Change:** per-batch seeded sampling (`batch_seed(gen_seed, raw_idx)`,
  sha256), autosave only at batch boundaries, `CACHE_VERSION` 46 with a
  provenance header, resume guard, `config.GEN_SEED = 0`,
  `scripts/build_cache.py --gen-seed`. Files: `models/generator.py`,
  `training/dataset.py`, `config.py`, `scripts/build_cache.py`,
  `tests/test_cache_determinism.py` (new).
- **Environment:** scratch venv outside the repo; Python 3.12.12, torch
  2.14.1 (CPU), transformers 4.52.4, datasets 3.6.0, huggingface_hub 0.33.4,
  pytest 9.1.1; Apple M4. The local HF token had expired; the real-model test
  ran with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (public model, cached weights).
- **Commands and results:**
  - `python -m pytest tests/test_cache_determinism.py -q` → 8 passed,
    1 skipped (slow) in 2.75 s.
  - Mutation check on scratch copies: old mid-batch autosave → 1 failed
    (resume test); unseeded batches → 2 failed (same-seed, resume).
  - `QAIR_SLOW=1 python -m pytest tests/test_cache_determinism.py::test_real_qwen_same_seed_same_hash -s`
    → 1 passed in 70 s; both 5-question ARC-Easy CPU builds:
    `hypotheses_sha256 = 5b4f840469306962c4f5d6379befd5827b7c91f70fb35dbe988f9d41cd23f51d`;
    Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`;
    `prompt_sha256 = 787016449994f6391a2356b0e13e490851ddd1f90bf6f186b59c5506b661191f`.
- **Not done:** no ARC-Easy cache built (smoke caches lived in pytest temp
  dirs); no training.
- **New issue logged:** A9 (any cache-load error moves the cache aside and
  rebuilds it), found when a `TorchVersion` header value failed
  `torch.load(weights_only=True)` in the first test run.
