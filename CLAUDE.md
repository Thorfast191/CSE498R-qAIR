# qAIR — Claude Code Project Instructions (CSE498R, baseline phase)

## Current phase
Supervisor direction (Oct 2026): first get the capstone baseline working
properly, possibly on an easier dataset. The medical domain comes later.
Drug-combination work is parked; it is not in this repo.

The baseline is the capstone code, **v46** (commit `9552498`, tag
`capstone-final`). Baseline work happens on branch `baseline`.

Repository: https://github.com/Thorfast191/CSE498R-qAIR.git (remote
`cse498r`). Local `baseline` tracks `cse498r/main` and is pushed there
(`git push`; this repo sets `push.default=upstream`). Its history starts
fresh at the v46 tree: commit `9552498` and tag `capstone-final` exist only
locally and in https://github.com/Thorfast191/qAIR-CSE499B (full history).
Never push `capstone-final`, `baseline-cse499b-history` or `main` to
`cse498r` — their history contains checkpoints. Local `main`
is older v40 code (it still has the input-ignoring circuit bug) — do not
train from it. The uncommitted v47 changes were not approved; they are kept
only as a backup patch (`~/qair_backup/v47_uncommitted.patch`) and must not
be applied without supervisor approval. v46's previous project notes are in
`docs/history/CLAUDE_v46.md`.

Baseline dataset: ARC-Easy, the default `ARC_CONFIG` since E9; `EPOCHS = 20`,
`PATIENCE = 5` (PD-7). Reproducing the capstone configuration needs
`--arc-config ARC-Challenge --epochs 8` (plus its 80/20 data caps). E3 and E7,
which blocked the ARC-Easy cache build, are fixed.

## What this project is
qAIR (Quantum-Augmented AI Reasoning) is a hybrid reasoning architecture for
multiple-choice question answering.

Pipeline (preserve this order):
Question → Hypothesis Generator (Qwen2.5-0.5B-Instruct, K support/attack pairs)
→ Sentence Encoder (frozen MiniLM, 384-d → 128-d projection)
→ Quantum Layer (6-qubit PennyLane StronglyEntanglingLayers, lightning.qubit)
→ Persistent Reasoner → Validator → Energy Scoring → Collapse → Prediction

The quantum layer is a CLASSICAL SIMULATION of a quantum circuit. Treat it as a
quantum-formalism inductive bias. Never claim quantum hardware speedup or
quantum advantage.

## Known issues (verify each against the code; full list in docs/baseline/issues.md)
- Capstone result not significant: McNemar p=0.6875, n=20, 95% CI [30%, 70%].
- Runs since v45 used 80 train / 20 validation questions, one seed, ~40 optimizer steps.
- Hypotheses from the 0.5B generator showed no signal beyond chance in the cache check.
- Readout is the squared norm of a vector linear in the evidence: with a reference
  term it ranks like additive scoring; without one it is sign-symmetric.
- Phase parameters barely moved (≤ 0.006 rad from π); one temperature never moved.
- K=8 hypotheses enter unpruned; no abstention mechanism.
- Early constant-circuit bug (Hadamard + AngleEmbedding rotation="X"): absent in v46
  (models/quantum_layer.py:179-181 uses rotation="Y"; output varies with input),
  present in v40 local main. Keep a regression test for it.
- v46 validator relevance target has no hypothesis axis (models/validator.py:142);
  a candidate fix exists in the v47 patch and needs supervisor approval.

## Goal of this phase
A baseline that is correct and honestly measured (see
docs/baseline/definition_of_done.md once agreed). Beating the baselines is the
research question, not a requirement for "working".

## Hard rules
1. Never fabricate numbers, results, dataset sizes, or citations. Every number
   must come from a command you ran or a file you read — name it.
2. Label claims: [VERIFIED: file:line or command], [HYPOTHESIS], [ASSUMPTION].
3. Read before changing: read the target file and all its callers first.
4. One issue per session, one variable per experiment. Record config, seed,
   dataset, split, and git commit hash for every run.
5. Never tune or select on the test split.
6. Do not remove the quantum layer or replace it with transformer layers.
7. Category C issues (design limitations) need supervisor approval before any
   change, because they alter the architecture. Propose, never implement unasked.
8. State every tensor shape change explicitly (before → after, and why).
9. Preserve checkpoint compatibility; if a change breaks it, version the
   checkpoint format and keep a loader for the old one.
10. Root cause before fix. If the root cause can't be identified from the code,
    say so and propose a diagnostic, not a guess.
11. Never commit data or checkpoints.

## Required baselines (same split, same seeds)
Mandatory:
- LLM option likelihood (generator-only, cloze scoring).
- LLM letter pick (generator-only, Qwen sees lettered options and picks one).
- Majority vote over the K generated hypotheses.
- P0: classical (decohered) collapse arm.
- P1: phase-ablated arm, phases pinned to 0.
Optional:
- Literal softmax arm (Born-rule collapse replaced by softmax).
- Parameter-matched classical aggregator (when time allows).
Run ≥3 seeds on the full evaluation split. Report mean ± std and paired tests
(McNemar or paired bootstrap).

## Metrics to log every run
Training loss, validation accuracy, test accuracy (final only), collapse entropy,
hypothesis diversity, spread, coherent_acc vs classical_acc, phase-parameter
drift, per-module gradient norms.

## Workflow
- Non-trivial tasks start in plan mode; wait for approval.
- Code changes: Problem / Root Cause / Proposed Solution / Expected Impact / Risks / Implementation.
- Experiment reviews: Observation / Interpretation / Recommendation / Next Experiment.
- Compute: Colab T4. Estimate time and memory first; smoke-test on ≤20 examples
  with a separate cache directory before any full run.
- Track issues in docs/baseline/issues.md; log every run in docs/research_log.md.
- Always commit before starting a training run; the code fingerprint ignores
  untracked .py files (checkpoint provenance, A3).

## Ask me before
changing pipeline order or adding/removing a stage; changing the dataset or
splits; breaking checkpoint compatibility; touching more than 3 files in one
change; any run over ~1 hour on T4; any category C change.

## Code style
Modular, descriptive names, one responsibility per function, no duplicate logic,
no placeholders or TODOs, comments only where they add understanding. Match the
existing style of the file being edited.
