# qAIR CSE498R — research log

One entry per fix or run: date, item, commit, environment, what was run and
what came out. Commit hashes are filled in by the follow-up entry when the
fix commit cannot name itself.

---

## 2026-10-18 — E3: seeded, reproducible hypothesis generation

- **Commit:** `b5c0ae8` (branch `baseline`, on top of `d0a074a`).
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

---

## 2026-10-18 — E7: reading a cache never builds one

- **Commit:** `6aa20a5`.
- **Change:** `QAIRDataset(..., allow_build=False)`; a read that the stored
  cache cannot satisfy raises with path, state, request and the build command,
  before any lock or model is touched. `scripts/build_cache.py` passes
  `allow_build=True`. Files: `training/dataset.py`, `scripts/build_cache.py`,
  `tests/test_cache_determinism.py` (helper builds with `allow_build=True`),
  `tests/test_cache_build_guard.py` (new).
- **Environment:** same scratch venv as E3.
- **Results:** `python -m pytest tests -q` → 13 passed, 1 skipped in 3.83 s.
  Mutation (`allow_build` default True) → 3 failed (the three refusal tests),
  2 passed.
- **Not done:** no ARC-Easy cache, no training.

---

## 2026-10-18 — A9: a cache that fails to load is never touched

- **Commit:** see the follow-up entry below.
- **Change:** `safe_load_cache` (`weights_only=True`, `TorchVersion`
  allow-listed via `torch.serialization.safe_globals`); load failures raise
  `CacheLoadError` with path, exception and the JSON sidecar header; `_save`
  writes `<cache>.header.json`. Files: `training/dataset.py`,
  `tests/test_cache_load_errors.py` (new).
- **Results:** `python -m pytest tests -q` → 18 passed, 1 skipped in 3.2 s.
  Mutations: rename handler restored → 3 failed; `weights_only=False` → the
  unsafe-pickle test failed (its marker file was created).
- **Scope note:** the checkpoint `torch.load` calls (checkpoint.py, runner.py,
  ablations.py, input_ablation.py, sample_inference.py) were added to A3's
  scope in issues.md.
- **Not done:** no ARC-Easy cache, no training.

