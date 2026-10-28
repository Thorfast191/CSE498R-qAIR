import contextlib
import hashlib
import json
import os
import time
import torch

from torch.torch_version import TorchVersion

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from config import (
    EMBEDDING_DIM as DIM,
    EMBEDDING_MODEL,
    GEN_BATCH_SIZE,
    GEN_MODES,
    GEN_SEED,
    CACHE_VERSION,
    ARC_CONFIG,
    cache_filename,
)

from benchmarks.arc import load_arc

from models.generator import HypothesisGenerator, LLM_NAME
from models.encoder import HypothesisEncoder


# Provenance fields that must match for a partial cache to be resumed:
# changing any of them changes the text that would be generated.
RESUME_INVARIANTS = (
    "gen_seed", "generator", "generator_revision", "gen_config", "prompt_sha256",
)


class CacheLoadError(RuntimeError):
    """A cache file exists but cannot be loaded. The file is left untouched."""


def safe_load_cache(cache_path):
    """
    torch.load with weights_only=True: tensors and plain containers only, so
    loading a cache can never execute code stored in it. TorchVersion (a str
    subclass) is allow-listed because a raw torch.__version__ in a header is
    harmless and would otherwise be refused by torch >= 2.6.
    """

    safe_globals = getattr(torch.serialization, "safe_globals", None)

    allow = safe_globals([TorchVersion]) if safe_globals else contextlib.nullcontext()

    with allow:
        return torch.load(cache_path, map_location="cpu", weights_only=True)


def header_path(cache_path):
    return cache_path + ".header.json"


def batch_seed(gen_seed, raw_idx):
    """
    Seed for the generation batch whose first question has raw index
    `raw_idx`. sha256, not hash(): Python's str hash is salted per process,
    so it would give a different seed after every restart.
    """

    digest = hashlib.sha256(f"{gen_seed}:{raw_idx}".encode("ascii")).digest()

    return int.from_bytes(digest[:8], "big") & (2 ** 63 - 1)


class QAIRDataset(Dataset):
    """
    Cached (question, hypotheses, options) embeddings for one ARC split.

    v45 sample schema
    -----------------
        Q            (D,)     question embedding
        H            (K, D)   hypothesis embeddings, K = len(modes) * N
        O            (N, D)   option embeddings
        polarity     (K,)     +1 support / -1 attack
        hyp_option   (K,)     which option each hypothesis is about
        is_fallback  (K,) bool
        llm_logprob  (N,)     length-normalized log P(option | question)
                              under the generator LLM
        y            int      index of the correct option
        source       str      "ARC-Challenge" / "ARC-Easy"

    Hypotheses are stored MODE-MAJOR -- all supports in option order,
    then all attacks in option order. collate_fn relies on that layout
    when it permutes options for shuffling.

    Two fields are new in v45 and neither can be back-filled by
    migration, because only the LLM can produce them: the attacking
    hypotheses and `llm_logprob`. A v33 cache is therefore refused rather
    than silently loaded with half the evidence channels missing.
    """

    def __init__(self, split="train", max_samples=None, cache_dir="./cache",
                 arc_config=None, modes=GEN_MODES, gen_seed=GEN_SEED,
                 autosave_every=50, allow_build=False):
        """
        allow_build: only scripts/build_cache.py passes True. Every other
        caller reads, and a cache that cannot satisfy the request (missing,
        capped below max_samples, or incomplete with max_samples=None)
        raises instead of silently starting hours of LLM generation.
        """

        self.samples = []

        self.gen_seed = gen_seed

        self.autosave_every = autosave_every

        # Set from the cache header on load, or from the generator on build.
        self.provenance = None

        self.arc_config = arc_config or ARC_CONFIG

        self.modes = tuple(modes)

        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(
            cache_dir, cache_filename(split, self.arc_config)
        )

        # Recorded in the run manifest (E6) with the file's sha256.
        self.cache_path = cache_path

        resume_raw_index = 0

        is_complete = False

        # ====================================================
        # LOAD CACHE (possibly partial)
        # ====================================================

        if os.path.exists(cache_path):

            print(f"[CACHE FOUND] {cache_path}")

            try:
                loaded = safe_load_cache(cache_path)

            except Exception as e:
                # Never rename, delete or rebuild here. The old handler
                # assumed every failure was a truncated write and moved the
                # file aside -- but a torch upgrade (weights_only=True by
                # default from 2.6) fails a perfectly good cache the same way,
                # and hours of generation would go to ".corrupt". The user
                # decides what happens to the file.
                raise CacheLoadError(self._load_failure_message(cache_path, e)) from e

            if isinstance(loaded, dict):
                self.samples = loaded.get("samples", [])
                is_complete = loaded.get("complete", False)
                resume_raw_index = loaded.get("raw_index", 0)
                version = loaded.get("version", 0)
                self.provenance = loaded.get("provenance")
            else:
                # Pre-v33 flat-list format. No version, no flags, no
                # question embedding -- nothing v45 can use.
                self.samples = loaded
                is_complete = True
                resume_raw_index = 0
                version = 0

            if self.samples:

                self._check_version(cache_path, version)

                # Refuse to silently mix embeddings built with a different
                # encoder/dim than the one config.py currently points at.
                # This is the exact "mat1 and mat2 shapes cannot be
                # multiplied" failure this project already hit once
                # (384 -> 768).
                cached_dim = self.samples[0]["H"].shape[-1]

                if cached_dim != DIM:

                    raise RuntimeError(
                        f"{cache_path} was built with {cached_dim}-dim "
                        f"embeddings, but config.EMBEDDING_DIM is now {DIM} "
                        f"(config.EMBEDDING_MODEL={EMBEDDING_MODEL}). Move, "
                        f"rename, or delete {cache_path} and rebuild before "
                        f"training -- refusing to silently mix incompatible "
                        f"cached embeddings."
                    )

            print(f"[CACHE LOADED] {len(self.samples)} samples "
                  f"(complete={is_complete}, version={version})")

            if self.samples and self.provenance is None:
                print(
                    "[PROVENANCE UNKNOWN] This cache predates seeded "
                    "generation (version < 46): its sampling seed, decoding "
                    "config and prompts were not recorded. Readable, but it "
                    "can never be resumed."
                )

            # Only the paths below may return early. A cache that failed to
            # load leaves samples empty, and returning here would hand the
            # caller a silently EMPTY dataset instead of rebuilding.
            # v45 caches have no per-sample score flag; nothing was flagged
            # then, so the honest default is "not recorded as failed".
            for sample in self.samples:
                sample.setdefault("score_failed", False)

            if self.samples:

                # A complete cache is final: return it (or its first
                # max_samples) and never enter the build path. Falling
                # through used to re-save it with complete=False.
                if is_complete:

                    if max_samples is not None and len(self.samples) > max_samples:
                        self.samples = self.samples[:max_samples]
                        print(f"[CACHE TRUNCATED TO max_samples] "
                              f"{len(self.samples)} samples")

                    elif max_samples is not None and len(self.samples) < max_samples:
                        print(f"[CACHE COMPLETE] the {split} split has only "
                              f"{len(self.samples)} samples; returning all "
                              f"{len(self.samples)} (max_samples={max_samples})")

                    self.report_quality(split)
                    return

                if max_samples is not None and len(self.samples) >= max_samples:
                    # Caller wants a capped subset and we already have enough.
                    #
                    # NOTE: a cache built with max_samples is stored with
                    # complete=False, because it does not cover the split.
                    # That is deliberate -- it lets a later full build
                    # resume and finish the rest. The consequence is that
                    # you must keep passing the same cap: calling this with
                    # max_samples=None against a capped cache will start
                    # generating the remainder of the split.
                    self.samples = self.samples[:max_samples]
                    print(f"[CACHE TRUNCATED TO max_samples] "
                          f"{len(self.samples)} samples")
                    self.report_quality(split)
                    return

            print(
                f"[CACHE INCOMPLETE] {len(self.samples)} samples, "
                f"raw_index {resume_raw_index}, complete={is_complete}"
            )

        else:

            print(f"[CACHE MISSING] {cache_path}")

        if not allow_build:
            self._refuse_build(
                cache_path, split, max_samples, is_complete, resume_raw_index
            )

        ds = load_arc(self.arc_config)

        if split not in ds:
            # Never substitute another split: a "test" cache silently built
            # from validation items would leak into model selection.
            raise ValueError(
                f"Split {split!r} does not exist in {self.arc_config}; "
                f"available splits: {sorted(ds.keys())}. Choose one of them, "
                f"or carve the split out explicitly in the dataset adapter."
            )

        if self.samples:
            print(f"[CACHE BUILD] Resuming {split} cache from raw index "
                  f"{resume_raw_index} ({len(self.samples)} samples so far)...")
        else:
            print(f"[CACHE BUILD] Building {split} cache "
                  f"({self.arc_config}) from scratch...")

        # ====================================================
        # BUILD (or RESUME) CACHE
        # ====================================================

        # Refuse to start if another process is already building this same
        # cache. Checked here rather than at load time so read-only users
        # of a COMPLETE cache are never blocked by a stray lock.
        self._check_lock(cache_path)

        self._touch_lock(cache_path, resume_raw_index)

        start_time = time.time()

        generator = HypothesisGenerator()

        try:
            self._start_provenance(generator, resume_raw_index)
        except RuntimeError:
            self._release_lock(cache_path)
            raise

        encoder = HypothesisEncoder()

        raw = ds[split]

        count = len(self.samples)

        last_raw_index = resume_raw_index

        # Examples that passed filtering but haven't been generated/encoded
        # yet: (raw_idx, stem, options, y, source).
        pending = []

        def flush_pending():

            nonlocal count, last_raw_index

            if not pending:
                return

            questions = [item[1] for item in pending]
            options_list = [item[2] for item in pending]

            count_before = count

            try:
                batch_out = generator.generate_batch_with_flags(
                    questions, options_list, modes=self.modes,
                    seed=batch_seed(self.gen_seed, pending[0][0]),
                )
            except Exception as e:
                print(f"[BATCH GENERATE FAILED] {e} -- falling back per-item")
                batch_out = [
                    {
                        "hypotheses": [
                            generator.fallback(q, o, m)
                            for m in self.modes for o in options
                        ],
                        "is_fallback": [True] * (len(self.modes) * len(options)),
                        "polarity": [
                            1.0 if m == "support" else -1.0
                            for m in self.modes for _ in options
                        ],
                        "hyp_option": [
                            oi for _ in self.modes for oi in range(len(options))
                        ],
                        "modes": [m for m in self.modes for _ in options],
                    }
                    for q, options in zip(questions, options_list)
                ]

            try:
                scores_list = generator.score_options(questions, options_list)
                score_failed = [False] * len(questions)
            except Exception as e:
                # Retry one question at a time, so one bad item does not
                # zero the LLM prior for the whole batch; flag what still fails.
                print(f"[OPTION SCORING FAILED] {e} -- retrying per question")
                scores_list, score_failed = [], []
                for q, opts in zip(questions, options_list):
                    try:
                        scores_list.append(generator.score_options([q], [opts])[0])
                        score_failed.append(False)
                    except Exception as item_error:
                        print(f"[OPTION SCORING FAILED] {item_error} -- storing "
                              f"zeros and score_failed=True for: {q[:60]!r}")
                        scores_list.append([0.0] * len(opts))
                        score_failed.append(True)

            all_hyp_texts = [h for item in batch_out for h in item["hypotheses"]]
            all_opt_texts = [o for options in options_list for o in options]

            H_all = encoder.encode(all_hyp_texts)
            O_all = encoder.encode(all_opt_texts)
            Q_all = encoder.encode(questions)

            h_off = 0
            o_off = 0

            for qi, ((raw_idx, stem, options, y, source), item) in enumerate(
                zip(pending, batch_out)
            ):

                h_len = len(item["hypotheses"])
                o_len = len(options)

                H = H_all[h_off:h_off + h_len]
                O = O_all[o_off:o_off + o_len]

                h_off += h_len
                o_off += o_len

                sample = {

                    "question": stem,

                    "options": options,

                    "source": source,

                    "hypotheses": item["hypotheses"],

                    "modes": item["modes"],

                    "is_fallback": torch.tensor(
                        item["is_fallback"], dtype=torch.bool
                    ),

                    "polarity": torch.tensor(
                        item["polarity"], dtype=torch.float
                    ),

                    "hyp_option": torch.tensor(
                        item["hyp_option"], dtype=torch.long
                    ),

                    "llm_logprob": torch.tensor(
                        scores_list[qi], dtype=torch.float
                    ),

                    "score_failed": score_failed[qi],

                    "Q": Q_all[qi].cpu(),

                    "H": H.cpu(),

                    "O": O.cpu(),

                    "y": y,
                }

                self.samples.append(sample)

                count += 1

                last_raw_index = raw_idx + 1

            pending.clear()

            # ========================================
            # PERIODIC SAVE (marked incomplete)
            #
            # Only at a batch boundary. A save in the middle of a batch
            # made a resumed build regroup the remaining questions into
            # different batches -- different padding and a different
            # seed per question -- so it could never reproduce an
            # uninterrupted build.
            # ========================================

            if count // self.autosave_every > count_before // self.autosave_every:

                self._save(cache_path, last_raw_index, complete=False)

                print(f"[AUTOSAVE] {count} samples "
                      f"(raw_index={last_raw_index})")

        for raw_idx, ex in enumerate(
            tqdm(raw, desc=f"Building {split} cache", initial=resume_raw_index)
        ):

            if raw_idx < resume_raw_index:
                continue

            if max_samples is not None and count >= max_samples:
                break

            try:

                question = ex["question"]

                # ============================================
                # ARC FORMAT
                # ============================================

                if isinstance(question, dict):

                    stem = question["stem"]

                    choices = question["choices"]

                    options = choices["text"]

                    labels = choices["label"]

                else:

                    stem = ex["question"]

                    choices = ex["choices"]

                    options = choices["text"]

                    labels = choices["label"]

                source = ex.get("source", self.arc_config)

                # ============================================
                # FILTERS
                # ============================================

                if len(options) < 2:
                    # Resolve anything already queued before this raw_idx
                    # advances past it, so last_raw_index never jumps ahead
                    # of samples that haven't actually been saved yet.
                    flush_pending()
                    last_raw_index = raw_idx + 1
                    continue

                answer = ex["answerKey"]

                if answer not in labels:
                    flush_pending()
                    last_raw_index = raw_idx + 1
                    continue

                y = labels.index(answer)

                # ============================================
                # QUEUE FOR BATCHED GENERATION + ENCODING
                # ============================================

                pending.append((raw_idx, stem, options, y, source))

                hit_chunk_size = len(pending) >= GEN_BATCH_SIZE

                hit_max_samples = (
                    max_samples is not None
                    and (count + len(pending)) >= max_samples
                )

                if hit_chunk_size or hit_max_samples:
                    flush_pending()

                if hit_max_samples:
                    break

            except Exception as e:

                print(f"[SKIP] {e}")

                flush_pending()

                last_raw_index = raw_idx + 1

                continue

        flush_pending()

        # ====================================================
        # FINAL SAVE (marked complete)
        # ====================================================

        finished_full_split = (max_samples is None)

        self._save(cache_path, last_raw_index, complete=finished_full_split)

        self._release_lock(cache_path)

        elapsed = (time.time() - start_time) / 60

        print(f"[CACHE SAVED] {cache_path} (complete={finished_full_split})")

        print(f"[TOTAL SAMPLES] {len(self.samples)}")

        print(f"[BUILD TIME] {elapsed:.2f} min")

        self.report_quality(split)

    # ========================================================
    # VERSION GATE
    # ========================================================

    def _check_version(self, cache_path, version):
        """
        v45 added attacking hypotheses and cached option log-likelihoods.
        Neither exists in a v33 cache and neither can be reconstructed
        without re-running the LLM, so there is no migration path -- only
        a rebuild. Failing loudly here is the whole point: a cache with
        support-only hypotheses is precisely the configuration the audit
        showed carries no discriminative signal (argmax diag(H.O) =
        0.2481 against a chance of 0.2500).
        """

        needs_rebuild = (
            version < 45
            or not self.samples
            or "llm_logprob" not in self.samples[0]
            or "polarity" not in self.samples[0]
        )

        if not needs_rebuild:
            return

        raise RuntimeError(
            f"\n{cache_path} is a version-{version} cache; v45 needs "
            f"version {CACHE_VERSION}.\n\n"
            f"  Missing: attacking hypotheses, per-hypothesis polarity, and "
            f"the cached\n  per-option LLM log-likelihood. None of these can "
            f"be migrated in -- only the\n  LLM can produce them.\n\n"
            f"  A support-only cache is the exact configuration the v44 audit "
            f"measured at\n  argmax diag(H.O) = 0.2481 (chance = 0.2500), i.e. "
            f"zero discriminative\n  signal. Training on it would reproduce "
            f"that result.\n\n"
            f"  Delete or rename {cache_path} and rebuild:\n"
            f"      python scripts/build_cache.py --splits train validation\n"
        )

    # ========================================================
    # LOAD FAILURE REPORT
    # ========================================================

    @staticmethod
    def _load_failure_message(cache_path, error):

        side = header_path(cache_path)

        if os.path.exists(side):
            try:
                with open(side) as f:
                    header = json.dumps(json.load(f), indent=2, sort_keys=True)
                header = (f"  sidecar header (written at the last save; may lag "
                          f"the cache by one save):\n{header}")
            except (OSError, ValueError) as side_error:
                header = f"  sidecar header {side} unreadable: {side_error}"
        else:
            header = (f"  no header sidecar ({side} missing): the cache predates "
                      f"version 46 or was never saved by this code")

        return (
            f"\nCannot load cache {cache_path}\n"
            f"  {type(error).__name__}: {error}\n"
            f"{header}\n"
            f"The file has NOT been moved, renamed or deleted.\n"
            f"  - weights_only / unsupported global: the cache contains a type "
            f"this torch refuses to unpickle safely; load it with the torch "
            f"version recorded above, never with weights_only=False on a file "
            f"you did not write.\n"
            f"  - truncated archive: inspect it, then move it yourself before "
            f"rebuilding with scripts/build_cache.py.\n"
        )

    # ========================================================
    # READ-ONLY GUARD
    # ========================================================

    def _refuse_build(self, cache_path, split, max_samples, is_complete,
                      raw_index):

        if os.path.exists(cache_path):
            seed = (self.provenance or {}).get("gen_seed", "unknown")
            state = (
                f"{len(self.samples)} samples stored, complete={is_complete}, "
                f"raw_index={raw_index}, gen_seed={seed}"
            )
        else:
            state = "missing"

        limit = "" if max_samples is None else f" --limit {max_samples}"

        raise RuntimeError(
            f"\nThe {split} cache cannot satisfy this request, and reading a "
            f"cache never builds one.\n"
            f"  cache       : {cache_path}\n"
            f"  state       : {state}\n"
            f"  requested   : max_samples={max_samples}\n"
            f"Read it with the cap it was built with, or build it explicitly:\n"
            f"  python scripts/build_cache.py --splits {split}{limit} "
            f"--arc-config {self.arc_config} "
            f"--cache-dir {os.path.dirname(cache_path)}\n"
        )

    # ========================================================
    # PROVENANCE
    # ========================================================

    def _start_provenance(self, generator, resume_raw_index):
        """
        Fresh build: record the generator's provenance. Resume: refuse if
        anything that determines the generated text differs from what the
        partial cache was built with, so one cache never mixes two
        sampling processes. Device and library versions may differ
        between segments (reproducibility is per device); each segment is
        recorded rather than refused.
        """

        current = dict(generator.provenance(), gen_seed=self.gen_seed)

        segment = {
            "start_raw_index": resume_raw_index,
            "device": current.pop("device"),
            "device_name": current.pop("device_name"),
            "torch_version": current.pop("torch_version"),
            "transformers_version": current.pop("transformers_version"),
        }

        if not self.samples:
            self.provenance = dict(current, segments=[segment])
            return

        if self.provenance is None:
            raise RuntimeError(
                "\nRefusing to resume a cache without provenance (version < 46).\n"
                "  Its seed and prompts are unknown, so new samples could not be\n"
                "  generated the same way. Build it again into a new cache dir.\n"
            )

        mismatched = {
            key: (self.provenance.get(key), current[key])
            for key in RESUME_INVARIANTS
            if self.provenance.get(key) != current[key]
        }

        if mismatched:
            lines = "\n".join(
                f"  {key}: cache={old!r}  now={new!r}"
                for key, (old, new) in mismatched.items()
            )
            raise RuntimeError(
                f"\nRefusing to resume: this partial cache was generated "
                f"differently.\n{lines}\n"
                f"Resume with the original settings, or build into a new cache dir.\n"
            )

        last = self.provenance["segments"][-1]

        if any(last[k] != segment[k] for k in segment if k != "start_raw_index"):
            print(
                f"[PROVENANCE] Resuming on a different device or library "
                f"version ({last['device_name']} -> {segment['device_name']}). "
                f"Text from the two segments is not reproducible on a single "
                f"device; both segments are recorded in the header."
            )

        self.provenance["segments"].append(segment)

    def hypotheses_sha256(self):
        """Hash of every question, option list and hypothesis, in order."""

        payload = json.dumps(
            [[s["question"], s["options"], s["hypotheses"]] for s in self.samples],
            ensure_ascii=False,
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ========================================================
    # SAVE / REPORT
    # ========================================================

    def _header(self, raw_index, complete):
        """Plain-JSON summary of the cache, readable even if torch.load is not."""

        provenance = self.provenance or {}
        segments = provenance.get("segments") or [{}]

        return {
            "version": CACHE_VERSION,
            "arc_config": self.arc_config,
            "raw_index": raw_index,
            "complete": complete,
            "n_samples": len(self.samples),
            "gen_seed": provenance.get("gen_seed"),
            "generator": provenance.get("generator"),
            "generator_revision": provenance.get("generator_revision"),
            "device": segments[-1].get("device"),
            "device_name": segments[-1].get("device_name"),
            "torch_version": segments[-1].get("torch_version"),
            "hypotheses_sha256": self.hypotheses_sha256(),
        }

    def _save(self, cache_path, raw_index, complete):
        """
        Atomic: write to a temp file in the same directory, then rename.

        The cache is autosaved every 50 samples during a build that takes
        hours, so being interrupted mid-write is a foreseeable state, not
        an exotic one. Writing in place meant a Ctrl-C (or a crash, or a
        reboot) landing inside torch.save left a TRUNCATED zip archive --
        which torch.load reports as an opaque
        "PytorchStreamReader failed reading zip archive: failed finding
        central directory" miniz error, destroying hours of generation
        with no indication of what went wrong. os.replace is atomic on
        both POSIX and Windows, so the real path only ever points at a
        complete file.
        """

        tmp_path = cache_path + ".tmp"

        with open(tmp_path, "wb") as f:

            torch.save(
                {
                    "samples": self.samples,
                    "raw_index": raw_index,
                    "complete": complete,
                    "encoder": EMBEDDING_MODEL,
                    "generator": LLM_NAME,
                    "arc_config": self.arc_config,
                    "modes": list(self.modes),
                    "version": CACHE_VERSION,
                    "provenance": self.provenance,
                    "hypotheses_sha256": self.hypotheses_sha256(),
                },
                f,
            )

            # os.replace makes the DIRECTORY ENTRY atomic, but on its own
            # says nothing about whether the file's CONTENTS reached the
            # platter. Without this fsync a hard power loss could leave a
            # perfectly valid directory entry pointing at data that was
            # still sitting in the OS write cache -- the same truncated-
            # archive symptom, just with a rarer trigger.
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, cache_path)

        side = header_path(cache_path)

        with open(side + ".tmp", "w") as f:
            json.dump(self._header(raw_index, complete), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(side + ".tmp", side)

        self._touch_lock(cache_path, raw_index)

    # --------------------------------------------------------
    # BUILD LOCK
    #
    # A cache build takes hours and autosaves as it goes, so the
    # realistic way to lose work is no longer a truncated write -- it's
    # TWO builds running against the same file. Both write atomically,
    # so neither corrupts anything, but they resume from each other's
    # raw_index and silently clobber each other's progress. That is easy
    # to trigger by accident: open the notebook while a background build
    # is running and the dataset cell will happily start a second one.
    #
    # Liveness is tracked with a heartbeat timestamp refreshed on every
    # autosave, NOT by probing the PID. os.kill(pid, 0) is the usual
    # trick and it is actively dangerous here: on Windows, os.kill with
    # any signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT calls
    # TerminateProcess, so a liveness "check" would kill the very build
    # it was asking about.
    # --------------------------------------------------------

    LOCK_STALE_SECONDS = 900  # 15 min; autosaves land every ~4 min

    @staticmethod
    def _lock_path(cache_path):
        return cache_path + ".lock"

    def _touch_lock(self, cache_path, raw_index):

        try:
            with open(self._lock_path(cache_path), "w") as f:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "heartbeat": time.time(),
                        "raw_index": raw_index,
                    },
                    f,
                )
        except OSError:
            # A lock we can't write is not worth failing a 5-hour build over.
            pass

    def _check_lock(self, cache_path):

        lock_path = self._lock_path(cache_path)

        if not os.path.exists(lock_path):
            return

        try:
            with open(lock_path) as f:
                info = json.load(f)
        except (OSError, ValueError):
            return

        age = time.time() - info.get("heartbeat", 0)

        if age > self.LOCK_STALE_SECONDS:
            print(
                f"[STALE LOCK] {lock_path} last updated {age / 60:.0f} min ago "
                f"(pid {info.get('pid')}). Assuming that build died; taking over."
            )
            return

        raise RuntimeError(
            f"\nAnother cache build is already running against {cache_path}.\n"
            f"  pid          : {info.get('pid')}\n"
            f"  last autosave: {age / 60:.1f} min ago "
            f"(raw_index={info.get('raw_index')})\n\n"
            f"Two builds on one cache overwrite each other's progress and "
            f"waste hours.\nWait for it to finish, or stop it and delete "
            f"{lock_path} to take over.\n"
        )

    def _release_lock(self, cache_path):

        try:
            os.remove(self._lock_path(cache_path))
        except OSError:
            pass

    # ========================================================
    # QUALITY REPORT -- the gate v44 did not have
    # ========================================================

    def report_quality(self, split):
        """
        v44 reported one number here: the template-fallback rate. It was
        0% on the final cache, and the cache was still useless, because
        a fluent hypothesis that justifies every option is uninformative
        in exactly the same way a template is -- it just doesn't look it.

        So this now measures DISCRIMINATIVE SIGNAL directly. Every line
        below is a zero-shot accuracy that needs no training, printed
        against chance:

          support agreement    argmax_n  cos(H_sup_n, O_n)
          attack agreement     argmin_n  cos(H_att_n, O_n)
          support - attack     argmax_n  [cos(H_sup_n,O_n) - cos(H_att_n,O_n)]
          LLM log-likelihood   argmax_n  llm_logprob_n

        If none of these beats chance, the cache carries no usable
        evidence and no architecture trained on it can do better than
        option priors. Rebuild with a different prompt; do not tune the
        model. That is the mistake this project spent a year making.
        """

        if not self.samples:
            return

        # The block decides whether a cache is usable; showing it for the
        # test split would invite selecting on test (CLAUDE.md rule 5).
        if split == "test":
            print(f"[CACHE QUALITY test / {self.arc_config}] not printed: "
                  f"the test split is never gated.")
            return

        n_total = len(self.samples)

        header = f"[CACHE QUALITY {split} / {self.arc_config}]"

        # ---------------- fallbacks ----------------

        flags = torch.cat([s["is_fallback"] for s in self.samples])

        n_fb = int(flags.sum())
        total_h = flags.numel()

        fully = sum(int(bool(s["is_fallback"].all())) for s in self.samples)

        print(
            f"{header} {n_total} samples   fallback hypotheses "
            f"{n_fb}/{total_h} = {100 * n_fb / max(total_h, 1):.1f}%   "
            f"fully-fallback samples {fully}/{n_total}"
        )

        if n_fb / max(total_h, 1) > 0.10:
            print(
                "  [WARNING] >10% fallbacks. These carry no information about "
                "which option is correct."
            )

        n_sf = sum(int(bool(s.get("score_failed", False))) for s in self.samples)

        print(f"  option scoring: score_failed {n_sf}/{n_total}")

        if n_sf:
            print(
                f"  [WARNING] {n_sf} samples have llm_logprob = 0 because "
                "scoring failed; the LLM prior is absent for exactly those items."
            )

        # ---------------- signal ----------------

        rows = [
            s for s in self.samples
            if "polarity" in s and s["H"].shape[0] == 2 * s["O"].shape[0]
        ]

        if not rows:
            print("  [WARNING] no support/attack pairs found -- cannot measure "
                  "discriminative signal.")
            return

        sup_hits, att_hits, margin_hits, llm_hits = 0, 0, 0, 0

        h_cos_sup, h_cos_att, o_cos, sup_att_cos = [], [], [], []

        chance = 0.0

        for s in rows:

            O = torch.nn.functional.normalize(s["O"].float(), dim=-1)
            H = torch.nn.functional.normalize(s["H"].float(), dim=-1)

            n = O.shape[0]

            chance += 1.0 / n

            sup, att = H[:n], H[n:2 * n]

            d_sup = (sup * O).sum(-1)
            d_att = (att * O).sum(-1)

            sup_hits += int(d_sup.argmax().item() == s["y"])
            att_hits += int(d_att.argmin().item() == s["y"])
            margin_hits += int((d_sup - d_att).argmax().item() == s["y"])

            lp = s.get("llm_logprob")
            if lp is not None and lp.numel() == n:
                llm_hits += int(lp.argmax().item() == s["y"])

            h_cos_sup.append(_offdiag_mean(sup @ sup.T))
            h_cos_att.append(_offdiag_mean(att @ att.T))
            o_cos.append(_offdiag_mean(O @ O.T))
            sup_att_cos.append((sup * att).sum(-1).mean().item())

        m = len(rows)
        chance /= m

        def line(label, hits):
            acc = hits / m
            verdict = "  <-- at chance" if abs(acc - chance) < 0.02 else ""
            print(f"    {label:<26s} {acc:.4f}{verdict}")
            return acc

        print(f"  zero-shot signal (chance = {chance:.4f}, n = {m}):")

        acc_sup = line("support agreement", sup_hits)
        acc_att = line("attack disagreement", att_hits)
        acc_margin = line("support - attack margin", margin_hits)
        acc_llm = line("LLM option likelihood", llm_hits)

        print(
            f"  diversity: cos(H_sup) {sum(h_cos_sup)/m:.4f}   "
            f"cos(H_att) {sum(h_cos_att)/m:.4f}   "
            f"cos(O) {sum(o_cos)/m:.4f}   "
            f"cos(sup_n, att_n) {sum(sup_att_cos)/m:.4f}"
        )

        best_h = max(acc_sup, acc_att, acc_margin)

        if best_h < chance + 0.02:
            print(
                "\n  [SIGNAL WARNING] No hypothesis-derived statistic beats "
                "chance by 2 points.\n"
                "  The generated evidence is symmetric: it justifies (or "
                "objects to) every\n  option equally well, so it carries no "
                "information about which is correct.\n"
                "  This is a GENERATION problem, not an architecture problem. "
                "Change the\n  prompts in models/generator.py and rebuild -- "
                "tuning the model cannot\n  recover a signal that was never "
                "generated.\n"
            )

        # 0.05 margin, not 0: on near-orthogonal embeddings both means sit
        # within noise of zero and a bare > comparison fires at random.
        if sum(h_cos_sup) / m > sum(o_cos) / m + 0.05:
            print(
                "  [DIVERSITY WARNING] supporting hypotheses are LESS diverse "
                "than the options\n  they were generated from -- generation is "
                "reducing discriminative spread."
            )

        if acc_llm > best_h + 0.05:
            print(
                f"  [NOTE] the raw LLM likelihood ({acc_llm:.4f}) beats every "
                f"hypothesis-derived\n  statistic ({best_h:.4f}). Any accuracy "
                f"gain the trained model shows must be\n  checked against the "
                f"no-LLM-prior ablation before it is attributed to reasoning."
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]


def _offdiag_mean(S):
    """Mean of the off-diagonal entries of a square similarity matrix."""

    k = S.shape[0]

    if k < 2:
        return 0.0

    return ((S.sum() - S.diagonal().sum()) / (k * (k - 1))).item()


def collate_fn(batch, shuffle_options=True):
    """
    shuffle_options: randomly permutes each sample's option order (and
    the matching hypotheses) so the model can't learn positional
    shortcuts from a small dataset. Pass shuffle_options=False for
    validation/eval so metrics are stable and reproducible.

    Ragged option/hypothesis counts (ARC has a handful of 3- and
    5-option questions) are zero-padded, and the masks say which entries
    are real. Those masks used to be computed here and then never passed
    to the model, so a zero-padded option participated in every reduction
    and could be returned as the prediction. Any new module that reduces
    over the K or N axis must accept and honor them.

    v45 emits three more tensors:

        polarity  (B, K)     +1 support / -1 attack
        align     (B, K, N)  1 where hypothesis k is ABOUT option n
        llm_lp    (B, N)     cached per-option LLM log-likelihood

    `align` is what lets the selector treat "the attack on option 2" as
    evidence about option 2 specifically. Without it the support/attack
    distinction is just a sign bit floating free of the option it refers
    to.
    """

    max_h = max(x["H"].shape[0] for x in batch)

    max_o = max(x["O"].shape[0] for x in batch)

    dim = batch[0]["H"].shape[-1]

    Hs, Os, Qs, ys = [], [], [], []
    H_masks, O_masks, fallbacks = [], [], []
    polarities, aligns, llm_lps = [], [], []

    for sample in batch:

        H = sample["H"]
        O = sample["O"]
        y = sample["y"]

        n = O.shape[0]
        k = H.shape[0]

        fb = sample.get("is_fallback", torch.zeros(k, dtype=torch.bool))

        pol = sample.get("polarity", torch.ones(k))

        hyp_opt = sample.get(
            "hyp_option", torch.arange(k) % max(n, 1)
        ).clone()

        llm_lp = sample.get("llm_logprob", torch.zeros(n)).float()

        if shuffle_options:

            # Mode-major layout: reorder each contiguous block of n
            # hypotheses by the SAME permutation, so block b row i still
            # refers to option i afterwards. Any other layout (a legacy or
            # hand-built sample) cannot be shuffled without misaligning
            # hypotheses and options, so it is refused (A12).
            mode_major = torch.arange(n).repeat(k // n)

            if k % n != 0 or not torch.equal(hyp_opt, mode_major):
                raise ValueError(
                    f"collate_fn cannot shuffle a sample whose hypotheses are "
                    f"not in mode-major order (K={k}, N={n}, hyp_option="
                    f"{hyp_opt.tolist()}): its hypotheses would point at the "
                    f"wrong options. Caches built by this code are always "
                    f"mode-major; pass shuffle_options=False for this sample."
                )

            perm = torch.randperm(n)

            O = O[perm]
            llm_lp = llm_lp[perm]

            idx = torch.cat([perm + block * n for block in range(k // n)])

            H = H[idx]
            fb = fb[idx]
            pol = pol[idx]
            hyp_opt = mode_major

            y = int((perm == y).nonzero(as_tuple=True)[0].item())

        h_len = H.shape[0]
        o_len = O.shape[0]

        H_mask = torch.zeros(max_h, dtype=torch.bool)
        O_mask = torch.zeros(max_o, dtype=torch.bool)

        H_mask[:h_len] = True
        O_mask[:o_len] = True

        fb_pad = torch.zeros(max_h, dtype=torch.bool)
        fb_pad[:h_len] = fb

        pol_pad = torch.zeros(max_h)
        pol_pad[:h_len] = pol

        align = torch.zeros(max_h, max_o)
        valid = hyp_opt[:h_len].clamp(min=0, max=o_len - 1)
        align[torch.arange(h_len), valid] = 1.0

        lp_pad = torch.zeros(max_o)
        lp_pad[:o_len] = llm_lp

        if h_len < max_h:
            H = torch.cat([H, H.new_zeros(max_h - h_len, dim)], dim=0)

        if o_len < max_o:
            O = torch.cat([O, O.new_zeros(max_o - o_len, dim)], dim=0)

        Hs.append(H)
        Os.append(O)
        Qs.append(sample["Q"] if "Q" in sample else torch.zeros(dim))

        ys.append(y)
        H_masks.append(H_mask)
        O_masks.append(O_mask)
        fallbacks.append(fb_pad)
        polarities.append(pol_pad)
        aligns.append(align)
        llm_lps.append(lp_pad)

    return {
        "H": torch.stack(Hs),
        "O": torch.stack(Os),
        "Q": torch.stack(Qs),
        "H_mask": torch.stack(H_masks),
        "O_mask": torch.stack(O_masks),
        "is_fallback": torch.stack(fallbacks),
        "polarity": torch.stack(polarities),
        "align": torch.stack(aligns),
        "llm_logprob": torch.stack(llm_lps),
        "y": torch.tensor(ys, dtype=torch.long),
    }
