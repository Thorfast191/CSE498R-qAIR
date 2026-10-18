Read CLAUDE.md first and follow it throughout.

# Task: write docs/baseline/issue_dossier.md — the defense document for every fix

Before fixing anything, I need one document that justifies every fix the way
I would defend it to my supervisor or a thesis examiner. For EVERY issue:
what we observed in the training results, why it happens, the mathematics or
theory behind it, the proposed fix, why the fix works, and how we will prove
it worked.

This session writes documents and one verification script. Do NOT fix any
issue, do NOT modify source code, do NOT start any training run.
Allowed new/changed files:

- docs/baseline/issue_dossier.md (new)
- docs/baseline/issues.md (only to add new issue IDs found here)
- scripts/verify_dossier_math.py (new)

Write the dossier section by section, saving after each section.

## Inputs

1. The v46 source code on this branch (the code is exactly capstone-final).
2. docs/baseline/issues.md and definition_of_done.md.
3. The v46 training run: notebooks/qAIR_v46.ipynb (its saved cell outputs).
   If the notebook in the repo has no outputs, use the evidence block below,
   which I extracted from my executed copy of that notebook.
4. The papers in the repo (Review Papers/ or docs/papers/) for theory.
   Cite only papers you actually read; never invent citations.

## Evidence from the executed v46 notebook (treat as [OBSERVED: v46 run])

Setup: ARC-Challenge, 80 train / 20 val (capped caches, complete=False),
8 epochs, batch 16 (5 steps/epoch), seed 42 with deterministic=False,
lightning.gpu, 1,056,382 parameters (54 circuit weights), 13,205 params/example.

Cache quality (train, n=80, chance 0.25): support agreement 0.1750,
attack disagreement 0.1875, support−attack margin 0.2000, LLM option
likelihood 0.3375. Diversity: cos(H_sup) 0.6644, cos(H_att) 0.6116,
cos(O) 0.4949, cos(sup_n, att_n) 0.7412.
Cache quality (val, n=20): 0.40 / 0.30 / 0.30 / LLM 0.40;
cos(H_sup) 0.6387, cos(O) 0.4106.
Example: for "dry palms" the attack hypothesis argues the correct answer is
wrong; for "Carbon dioxide" the attack hypothesis actually SUPPORTS the option.

Sanity checks all PASS, including destructive_fraction 1.0000 before training.

Training, epoch 1 → 8 (main run qair_v46):

- train loss 1.889 → 1.650 → 1.359 → 1.124 → 0.944 → 0.855 → 0.792 → …
- val acc 0.35, 0.35, 0.40, 0.45, 0.40, 0.40, 0.45, …
- ECE 0.161 → 0.162 → 0.206 → 0.296 → 0.307 → 0.292 → 0.335 (final 0.385)
- collapse entropy 2.0704 → 2.0763 (log 8 = 2.0794); collapse peak 0.149 → 0.141 (uniform = 0.125)
- diversity 0.0182 → 0.0067; energy-row pairwise cos 0.769 → 0.838; H cos ≈ 0.87–0.88
- phase effect 0.974 → 0.993; destructive fraction 1.0000 every epoch
- polarity_phase 3.1421, 3.1429, 3.1427, 3.1418, 3.1410, 3.1402 (init π = 3.1416)
- energy_alpha ≈ 0.499, validator_alpha ≈ 0.500, dt ≈ 0.100, anchor ≈ 0.499,
  llm_prior ≈ 0.3136 — essentially unchanged from init across 8 epochs
- coherent acc vs classical: +0.10, +0.05, 0.00, +0.10, +0.05, +0.05 …
- [INTERFERENCE SATURATED] warning fired at epoch 3.
  Best checkpoint evaluation (val, n=20): acc 0.50, classical_acc 0.30,
  coherent_acc 0.25, ECE 0.385, entropy 2.0763, diversity 0.0067.
  Input ablation (acc 0.50 intact): H:=zeros 0.30, H:=noise 0.30,
  H shuffled 0.35, H rolled 0.25, H:=O 0.35, support:=0 0.30, attack:=0 0.40,
  polarity:=0 0.45, align:=0 0.45, llm_logprob:=0 0.25, O:=0 0.40.
  Direct LLM baseline 0.40 (8/20). McNemar vs direct LLM: b=2, c=4, p=0.6875.
  95% CI [0.30, 0.70].
  Data-ceiling probes (val): O only 0.25; Q×O 0.25; Q×O+support 0.40;
  Q×O+support+attack 0.30; Q×O+evidence+llm 0.35.
  Ablation grid: A1_baseline SKIPPED ("already finished 8 epochs" via resume);
  A3_persistent resumed from epoch 7 → best 0.15 (pairwise cos 0.9706,
  diversity 0.0035); P0_classical_collapse resumed from epoch 6 → 0.20.
  "No test_records found" — the test split was never evaluated.
  Environment: pip resolver conflicts (huggingface-hub 0.33.4 vs diffusers/gradio),
  PennyLane warning about JAX 0.7.2 > 0.6.2.

## Required structure of docs/baseline/issue_dossier.md

### 0. One-page summary for my supervisor

Plain language: what is broken, what is merely noisy, what is a design
limitation, and the fix order. No equations here.

### 1. Training-run diagnosis (the whole v46 run read as one story)

Interpret the evidence above as a set of linked findings, each with the
numbers and the reasoning. At minimum:

- Overfitting: loss falls 58% while val acc is flat; 13,205 params/example.
- Calibration degrades: ECE 0.16 → 0.39 as training proceeds.
- Collapse barely selects: entropy within 0.004–0.009 nats of log K.
  Compute what the collapse distribution would need to look like to be
  "selective", and how far the observed one is from uniform (KL to uniform).
- Representation collapse: diversity falls and energy-row cosine rises
  (0.97 in A3).
- Interference metrics are saturated by initialization (polarity_phase at π).
- Most learned scalars (alphas, dt, anchor, llm_prior weight) never move.
- Accuracy rides on the LLM prior (llm_logprob:=0 → chance).
- Sub-chance arms are statistically consistent with chance at n=20:
  P(X ≤ 3 | n=20, p=0.25) ≈ 0.225.
- What n=20 can and cannot detect: half-width of a 95% CI ≈ 1.96·√(p(1−p)/n)
  (≈ ±0.22 at n=20; ±0.04 at n=570; ±0.02 at n=2376 for ARC-Easy test).
  Each finding links to the issue IDs it explains.

### 2. One section per issue

Cover every issue in issues.md (A, B, D, E, and C as "needs supervisor
discussion"), plus any NEW issue the training-run diagnosis reveals that is
not yet in issues.md (add it to issues.md with a new ID). Likely new ones:
collapse entropy ≈ log K; representation collapse; learned scalars never
moving; ECE rising; deterministic=False; unpinned/conflicting environment.
Verify each before adding it.

For EACH issue use exactly this template:

#### [ID] Title

- **Category / severity / blocks:** (e.g., blocks the ARC-Easy cache build)
- **Observation:** the symptom, with numbers from the v46 run or a command
  you ran. Label [OBSERVED: v46 run] or [VERIFIED: command].
- **Location:** file:line(s) in v46, with the relevant code quoted (short).
- **Root cause:** the mechanism that produces the symptom, in words.
- **Theory / mathematics:** a derivation showing WHY the root cause produces
  the observed symptom. Use LaTeX ($…$, $$…$$). Examples of the depth I want:
  - Sign symmetry (C1): show the readout depends on |s − a| (or (s − a)²),
    so options with equal-magnitude support or refutation tie; give a
    two-option counterexample with numbers.
  - Same validator target (A7): write the loss with target[…, :, y] = 1 for
    every hypothesis, take the gradient, and show the validator can only
    learn option relevance, not which hypothesis is sound.
  - Saturated destructive_fraction: show that with the polarity phase at π
    the attack term enters with sign −1, so the fraction is 1 by
    construction at initialization, independent of the data.
  - Phases not learning: derive ∂L/∂φ at φ = π (or show why it is small
    relative to other parameters' gradients) using the actual readout.
  - Near-uniform collapse: relate entropy to the logit spread; show what
    logit range gives H = 2.07 for K = 8, and why the temperature/energy
    scale keeps it there.
  - Overfitting: parameter-to-sample ratio; why falling loss with flat val
    acc and rising ECE is the overfitting signature.
  - Unseeded generation (E3): show that sampling without a fixed generator
    seed makes the cache a random variable, so results are not reproducible.
  - Resume across seeds (A3): show the checkpoint name collision and why a
    second seed silently continues the first seed's run (the grid log shows
    A1 "already finished" and A3/P0 resuming mid-run).
  - Selection bias (B3): show why max over epochs of a noisy val accuracy is
    biased upward, and the size of that bias at n=20, 8 epochs.
    For engineering issues, the "theory" can be a precise causal argument
    instead of equations, but it must still be exact.
- **Proposed fix:** the minimal change, with tensor shapes before → after
  if any, and checkpoint-compatibility impact.
- **Alternatives considered:** at least one, and why it was not chosen.
- **Why the fix works:** the argument or derivation showing the fix removes
  the root cause (not just the symptom).
- **Falsifiable prediction:** what metric should change, in which
  direction, and roughly by how much, if the fix is right. What result would
  show the fix was wrong.
- **Verification test:** the exact test (unit, numerical, or smoke run) that
  will prove it, runnable on CPU where possible.
- **Risks / side effects:** including effects on comparability with the
  capstone results.
- **Needs supervisor approval?** yes/no and why (any change to the training
  objective, architecture, or pipeline = yes).

### 3. Dependency graph and fix order

Which fixes must precede which (e.g., E3 before building the ARC-Easy cache;
A3 before multi-seed runs; B-items after the cache). Keep the order from
issues.md unless the analysis gives a reason to change it; justify changes.

### 4. Items for the supervisor meeting

A short list: A7, C1–C5, and any new issue that changes the objective or
architecture, each with a one-line recommendation and the decision needed.

### 5. Claims I must NOT make (honesty list)

From the evidence: e.g., no accuracy claim at n=20; no claim that
interference helps (coherent_acc 0.25 < classical 0.30 at the best
checkpoint); no claim from sub-chance arms. One line each, with the reason.

## Verification script: scripts/verify_dossier_math.py

CPU only, numpy/scipy. Numerically check every mathematical claim in the
dossier that can be checked, e.g.:

1. Entropy of the observed collapse vs log 8; KL to uniform.
2. Sign symmetry of the readout on a constructed example.
3. Validator gradient under the v46 target: same gradient direction for
   every hypothesis of the correct option.
4. destructive_fraction = 1 at polarity_phase = π on random inputs.
5. Binomial facts: P(X ≤ 3 | 20, 0.25); exact McNemar p for b=2, c=4;
   CI half-widths at n = 20, 570, 2376.
6. Expected upward bias of max-over-8-epochs accuracy at n=20 (simulation).
   Each check prints PASS/FAIL and also runs a deliberately wrong version that
   must FAIL. Paste the output into the dossier as [VERIFIED: scripts/verify_dossier_math.py].

## Labels

[OBSERVED: v46 run], [VERIFIED: file:line or command], [DERIVED] for math
shown in the document, [HYPOTHESIS], [ASSUMPTION]. Never present a
hypothesis as a finding.

## Finish

Commit the dossier, the script and any issues.md additions on `baseline`
and push to cse498r. Then give me: the list of issue IDs with one-line
root causes, any NEW issues you added, the script's pass/fail summary, and
the supervisor decisions needed.
