# Dataset options — baseline now, medical later

Researched 2026-10-17. **Split sizes and licenses were fetched this session**
from the Hugging Face datasets-server `/size` and `/first-rows` endpoints and the
Hub API `cardData.license`. PIQA came from its original source files, because
the HF viewer rejects its loading script. Nothing below is from memory unless it
is marked [ASSUMPTION].

## How the current pipeline reads data (what "works unchanged" means)

- `benchmarks/arc.py:45` loads only `ai2_arc`. `--arc-config` accepts only
  `ARC-Challenge | ARC-Easy | merged` (`main.py:34-36`,
  `scripts/build_cache.py:59-61`).
- `training/dataset.py:351-400` expects `ex["question"]` (a string, or a dict
  with `stem`) plus `ex["choices"]["text"/"label"]` and `ex["answerKey"]`. Rows
  with `answerKey not in labels` are filtered out. Any exception skips the row
  silently (`:421-429`).
- K = 2 × N hypotheses per question (`config.py:170`), so generation cost scales
  with the number of options N.

## Cost model for "estimated T4 cost"  ([ASSUMPTION] — must be checked by a timed smoke run)

Measured inputs, from this repo only:
- Option scoring on a T4: 200 questions in 2:56, about **0.88 s/question**
  (`qAIR_v44.ipynb` cell 31, `Qwen/Qwen2.5-0.5B-Instruct direct: 200/200 [02:56]`).
- Training on a T4, v44: **1.24 s/step** at batch 16, K=4 (`qAIR_v44.ipynb`
  cell 21). No v45+ rate was saved (issue E8).

Assumed: generation of 8 × ≤64-token completions per 4-option question at
0.5–2.0 s. So **cache ≈ 1.4–2.9 s per 4-option question**, scaled by N/4.
Training is assumed at 1–3 s/step (K=8, MODEL_DIM 128). "Cache h" covers all
labelled splits used. "Train min/epoch" = ⌈train/16⌉ steps × 1–3 s.

## Table A — baseline dataset candidates

| | **ARC-Easy** | SciQ | OpenBookQA (main) | CommonsenseQA | PIQA |
|---|---|---|---|---|---|
| HF id | `allenai/ai2_arc` / `ARC-Easy` | `allenai/sciq` | `allenai/openbookqa` / `main` | `tau/commonsense_qa` | `ybisk/piqa` (script) |
| Format | science MCQ, `question` + `choices{text,label}` + `answerKey`; mostly 4 options; labels mix `A–D` and `1–4` (seen in test rows) | `question`, `correct_answer`, `distractor1-3`, `support` passage | `question_stem` + `choices{text,label}` + `answerKey`, 4 options | `question` + `choices` (5) + `answerKey` | `goal`, `sol1`, `sol2`, `label` (2 options) |
| Split sizes (verified) | train **2251**, val **570**, test **2376** | train **11679**, val **1000**, test **1000** | train **4957**, val **500**, test **500** | train **9741**, val **1221**, test **1140** | train **16113**, dev **1838**, test **3084** (line counts of source files) |
| Labelled test? | yes | yes (`correct_answer` present) | yes | **no**: test `answerKey` is `""` in all first 100 rows | **no**: no test labels file; HF script fills `-1` |
| Works unchanged? | **Yes**: `--arc-config ARC-Easy` (`config.py:54-58`). Sizes match `config.ARC_SPLIT_SIZES` | No: needs an adapter to build `choices` **with a seeded shuffle** (otherwise the correct answer sits at a fixed index, which is positional leakage), and must drop `support` to stay closed-book | No: `ex["question"]` raises KeyError on every row → all silently skipped → empty dataset. Needs a column rename plus plumbing | Row shape matches, but needs plumbing. Test is unlabelled, so official val becomes test and val is carved from train. K=10 | No: adapter (2 options, K=4); dev becomes test, val carved from train |
| License (HF card) | CC-BY-SA-4.0 | CC-BY-NC-3.0 (non-commercial; fine for coursework) | "unknown" on card | MIT | "unknown" on card |
| Expected difficulty for Qwen2.5-0.5B | [ASSUMPTION] easier than ARC-Challenge by construction (Challenge = questions retrieval and co-occurrence baselines got wrong; ARC paper, Clark et al. 2018, cited from memory). Measured reference: ARC-**Challenge** cloze 0.3375 (n=80) / 0.40 (n=20) in this repo; Qwen2.5 report Table 5: base 0.5B ARC-C **35.6** | [ASSUMPTION] easy closed-book science recall; trivial if `support` were shown | [ASSUMPTION] hard: needs an external "open book" fact not given at test | [ASSUMPTION] moderate–hard; commonsense, 5 options | [ASSUMPTION] binary, chance 50%; physical commonsense |
| Contamination risk | [ASSUMPTION] high: public since 2018, in many training mixes | [ASSUMPTION] high | [ASSUMPTION] high | [ASSUMPTION] high | [ASSUMPTION] high |
| Est. T4 cache | 5197 q → **≈ 2.0–4.2 h** | 13679 q → 5.3–11.0 h (train subsample of 2000 → 1.6–3.2 h) | 5957 q → 2.3–4.8 h | 10962 q × 5/4 → 5.3–11.0 h | 17951 q × 2/4 → 3.5–7.2 h |
| Est. train time/epoch | 141 steps → 2.4–7 min | 730 steps → 12–37 min (full) | 310 steps → 5–16 min | ~610 steps → 10–31 min | ~1008 steps → 17–50 min |

The contamination note applies to every row. All arms share the same generator,
so contamination shifts absolute numbers but largely cancels in paired
comparisons. The generator-only baselines measure it directly.

## Table B — medical step (later)

| | **MedMCQA** | MedQA (USMLE, 4-option) | PubMedQA (labelled) | MMLU medical subsets |
|---|---|---|---|---|
| HF id | `openlifescienceai/medmcqa` | `GBaker/MedQA-USMLE-4-options` (also `bigbio/med_qa` `med_qa_en_4options_source`) | `qiaojin/PubMedQA` / `pqa_labeled` | `cais/mmlu` |
| Format | `question`, `opa..opd`, `cop` (0–3), `choice_type`, `exp`, `subject_name` | long clinical vignette, `options` {A–D}, `answer_idx` letter | `question` + abstract `context` → `final_decision` yes/no/maybe | 4-option MCQ, `answer` index |
| Split sizes (verified) | train **182822**, val **4183**, test **6150** | GBaker: train **10178**, test **1273** (no val). bigbio 4-option: train 10178, val **1272**, test 1273 | **1000** labelled, single `train` split (also 61249 unlabelled, 211269 artificial) | test-only per subject: anatomy 135, clinical_knowledge 265, college_biology 144, college_medicine 173, medical_genetics 100, professional_medicine 272 (= 1089) |
| Labelled test? | **no**: test `cop = -1`. Use val (4183) as test and carve val from train | yes | no official split in this config | yes (test) |
| Works unchanged? | No: adapter (4 options to `choices`, `cop` to index). Filter `choice_type` to `single` [ASSUMPTION: multi rows exist; check counts] | No: adapter. Vignettes likely exceed `encoder.py:47` `max_length=128` (D4) | No: 3 fixed answers, and needs context, so a different task shape | adapter; evaluation-only (no train split of the same distribution) |
| License (HF card) | Apache-2.0 | GBaker: CC-BY-4.0 as declared by the uploader (verify against the original release); bigbio: "unknown" | MIT | card license not fetched; [ASSUMPTION] MIT (check) |
| Expected difficulty for 0.5B | [ASSUMPTION] hard; medical-entrance exam knowledge | [ASSUMPTION] very hard; near chance likely | [ASSUMPTION] depends on reading the abstract | [ASSUMPTION] hard |
| Contamination | [ASSUMPTION] high (public since 2022) | [ASSUMPTION] high | [ASSUMPTION] high | [ASSUMPTION] high |
| Est. T4 cache | val 4183 + train subsample 5000 → 3.6–7.4 h | 12723 q, assume ×1.5 for long prompts → 7.4–15.4 h | ~0.4–0.9 h (long contexts) | 1089 q → 0.4–0.9 h |

**Closest in format to ARC-Easy: MedMCQA.** It has a short exam-style stem,
4 options and a single answer index, so it needs only a column adapter. MedQA's
long vignettes and PubMedQA's context-plus-yes/no/maybe shape both break
assumptions in the encoder and loader.

## Decision (2026-10-17)

**ARC-Easy is the baseline dataset.** Its cache is **not** to be built until
issues E3 (unseeded generation) and E7 (capped cache regenerating) are fixed;
see `issues.md` §1. MedMCQA remains the proposed medical step and is not decided.

## Recommendation

**Baseline: ARC-Easy.**
1. It is the only candidate that runs **with zero code changes**
   (`--arc-config ARC-Easy`), so the baseline phase measures the capstone code
   and not a new loader. CLAUDE.md asks to keep changes minimal and to ask
   before changing datasets.
2. Same distribution family as the capstone (ARC science questions), so the
   ARC-Challenge history stays interpretable. [ASSUMPTION] It is easier, which
   gives the 0.5B generator its best chance of producing hypotheses with signal;
   the cache gate tests this on full splits (D1).
3. Larger evaluation sets than ARC-Challenge (val 570, test 2376 vs 299/1172), so
   tighter CIs (DoD §4.6).
4. CC-BY-SA-4.0; cache about 2–4 h on a T4 [ASSUMPTION], still above the 1-hour
   "ask me first" threshold.
Runner-up: OpenBookQA (same shape after one rename, but much harder) or SciQ
(adapter and shuffle needed; NC license).

**Medical (later): MedMCQA.** Closest to ARC-Easy in format, Apache-2.0, a
labelled validation of 4183 for use as test, and a large train pool to subsample.
Add MedQA-USMLE afterwards as the harder, clinically realistic target, once the
128-token encoder truncation (D4) is addressed.

## Cheap measurement before committing (not a training run)

Run the cloze scorer (`HypothesisGenerator.score_options`, the same code as
`evaluation/baselines.py --mode direct`) zero-shot on 300 validation questions of
each Table A candidate. At the measured 0.88 s/question that is about
4–5 min each on a T4. This replaces the [ASSUMPTION] difficulty cells with
measured numbers.

## Citations

Paper attributions (Clark et al. 2018 for ARC) are given from memory and were
**not fetched this session**. Check them against the HF dataset cards before
using them in the thesis. The Qwen2.5 figure is from the Qwen2.5 Technical
Report, Table 5 ("Performance of the smaller base models"), arXiv:2412.15115,
fetched this session. It is a **base**-model, few-shot number; this repo uses the
**Instruct** model zero-shot.
