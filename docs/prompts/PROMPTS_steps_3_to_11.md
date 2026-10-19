# qAIR CSE498R — Claude Code prompts, Steps 3–11

Rules for every step:
- One Claude Code session on this repo at a time.
- Run in normal mode (not auto-accept), so edits wait for your approval.
- Paste each prompt only after the previous step is finished.
- Bring each final summary back to the claude.ai chat for checking.

---

## Step 3 — Phase 1 fixes (Tier 0, no supervisor approval needed)

```
Start Phase 1 of the fix order in docs/baseline/issues.md, in this order:
E3, E7, A6, A3 (also closes A8 going forward), A4, A5, E10, E11, A1/A2,
then the regression tests (circuit input dependence, gradient coverage).
Skip anything marked "needs supervisor approval".

For EACH issue, one at a time:
1. Re-read its section in docs/baseline/issue_dossier.md and confirm the
   root cause in the code. If the code disagrees with the dossier, stop
   and tell me.
2. Show me: Problem / Root Cause / Proposed Solution / Expected Impact /
   Risks / Implementation, including tensor shapes and checkpoint
   compatibility. For A1/A2, state whether removing or wiring the unused
   parameters breaks loading v46 checkpoints, and propose the option that
   keeps them loadable.
3. WAIT for my approval before editing anything.
4. After approval: implement the minimal change, run the verification
   test named in the dossier (CPU, tiny smoke cache only, never the real
   cache), and show the output.
5. Update issues.md (status: fixed, commit hash, test evidence) and the
   dossier's status line; append to docs/research_log.md; commit
   "fix <ID>: <title>"; push to cse498r.
Then move to the next issue and repeat from step 1.

Never touch two issues in one commit. If you notice a new problem, add it
to issues.md and keep going. Do not build the ARC-Easy cache or start any
training run; stop after the regression tests and summarize.
```

If a session ends partway, start the next one with:

```
Continue Phase 1 from docs/baseline/issues.md: list which Phase 1 issues
are fixed (with commit hashes) and which remain, then continue with the
next open one using the same one-issue-at-a-time procedure.
```

---

## Step 4 — Supervisor meeting

No Claude Code prompt. Bring:
- the one-page summary from docs/baseline/improvement_plan.md;
- docs/baseline/path_hamiltonian_design.md (its plain-language summary);
- the A8 disclosure (v46 ablation-grid numbers came from stale checkpoints).

Ask him to review the provisional decisions in
docs/baseline/supervisor_decisions.md (PD-1 to PD-6: A7, Tier 1, Tier 2,
Tier 3 including the H0/H1/H2 arms, C4/C5 deferred). Steps 8–10 run under
those provisional decisions before the meeting. Afterwards, record his
review with:

```
Record my supervisor's review from the meeting on <date> in
docs/baseline/supervisor_decisions.md: replace each "provisional —
pending supervisor review" with his decision and the date, and where it
differs record the action taken (arm removed, flag kept off, ...). Update
the status of each affected item in issues.md and improvement_plan.md.
Decisions:
- A7: <approved / rejected / modified: ...>
- Tier 1: <...>
- Tier 2: <...>
- Tier 3 (C1 signed readout, C2 rho=pi/2 arm, C3 abstention): <...>
- Path Hamiltonian arms H0/H1: <...>
- C4/C5: <...>
- Other notes: <...>
Commit and push to cse498r. Change no code.
```

---

## Step 5 — Phase 2 engineering (Tier 0)

```
Start Phase 2 of the fix order in docs/baseline/issues.md: E6 (run
provenance manifest), E5 (ablation results overwritten), E8 (tests/ and
a timed smoke-run script), E9 (stale docs and names).

Same procedure as Phase 1, one issue at a time:
confirm the root cause against the dossier; propose (Problem / Root
Cause / Proposed Solution / Expected Impact / Risks / Implementation);
WAIT for my approval; implement the minimal change; run its test (CPU,
smoke cache only); update issues.md, the dossier status and
docs/research_log.md; commit "fix <ID>: <title>"; push to cse498r.

For E6, every run must write a manifest with: git commit, dirty flag,
cache file hash and header (seed, generator model + revision, prompt
hash, generation config), dataset name/config/split sizes, seed,
deterministic flag, device (torch and quantum), package versions
(pip freeze hash), and the full model config.

For E8, write scripts/smoke_timing.py that builds a 20-question ARC-Easy
cache in a SEPARATE smoke directory and trains 1 epoch, reporting
seconds per stage (generation, scoring, embedding, train step, eval).
Do not run it on the real cache; I will run it on Colab.

Stop after E9 and summarize, including the exact Colab commands for the
smoke timing run.
```

---

## Step 6 — Build the ARC-Easy cache (first long GPU run)

First, prepare the Colab notebook:

```
Create notebooks/cse498r_baseline.ipynb for Colab T4 from the current
baseline branch. It must:
1. Mount Drive and use a NEW folder: /content/drive/MyDrive/qAIR_CSE498R/
   with subfolders cache/arc_easy/, ckpt/, runs/, exports/. Never read or
   write qAIR_V45 or any older folder.
2. Clone Thorfast191/CSE498R-qAIR at a pinned commit (print it) and
   install pinned requirements; print pip freeze hash.
3. Section A — smoke timing: run scripts/smoke_timing.py on 20 questions
   in a smoke subfolder; print per-stage times and the extrapolated time
   for the full ARC-Easy build (2251 train / 570 val / 2376 test).
4. Section B — full cache build, ONLY via scripts/build_cache.py with the
   fixed seed, resumable, autosaving; then run the cache-quality gate on
   the FULL splits and print it.
5. Write the run manifest (E6) to runs/.
Do not run anything; I will run it on Colab. Show me the cell list.
```

Then run Section A on Colab, paste the timings here, and only then run Section B.

---

## Step 7 — Phase 3: the true v46 baseline (end of Tier 0)

```
Start Phase 3 of the fix order in docs/baseline/issues.md: B4, B5, B6,
B3, B7, then the full-split runs that close B1, B2, B8, B10.

Same one-issue-at-a-time procedure (confirm, propose, WAIT, implement,
test on CPU smoke cache, update docs, commit, push).

Specifics:
- B4: LLM letter-pick baseline — Qwen sees the question and all lettered
  options and we read the probability of each letter token. Same model and
  revision as the generator.
- B5: evidence-weighted majority vote over hypotheses (a plain count is
  degenerate: one support and one attack per option gives net 0); define
  the weighting from the cached validator/LLM signals, pre-registered.
- B6: paired item-level statistics: McNemar per seed, paired bootstrap
  over items (not pooled item×seed cells), Holm correction across the
  comparison family.
- B3/B7: select checkpoints and probe epochs on validation only; report
  test exactly once per arm and seed.

Then prepare (do not run) the Colab section that trains and evaluates, on
full ARC-Easy, seeds 0/1/2, with deterministic=True and fresh checkpoint
folders: v46 (A3 arm), F0, P0, P1, plus the baselines: LLM option
likelihood, LLM letter pick, evidence-weighted majority vote. Output one
results table: test accuracy mean ± std, 95% CI, paired tests vs both LLM
baselines with Holm correction, and the diagnostics (collapse entropy,
diversity, phase drift, coherent vs classical accuracy). Estimate T4 time
from the Step 6 smoke timings and tell me before I run it.
```

After the Colab run, have it analysed:

```
Read the results in runs/<run folder>. Write docs/baseline/baseline_results.md:
Observation / Interpretation / Recommendation / Next Experiment. Report
test accuracy for every arm and baseline (mean ± std over 3 seeds, 95% CI),
paired tests with Holm correction, and which pre-registered predictions in
improvement_plan.md held or failed. This is the Tier 0 baseline every later
change is compared against. No claims beyond what the numbers support.
Commit and push.
```

---

## Step 8 — Tier 1: better evidence (provisional decision PD-3)

```
Implement Tier 1 from docs/baseline/improvement_plan.md under provisional
decision PD-3 in docs/baseline/supervisor_decisions.md (provisional —
pending supervisor review): each option behind a flag as a separate arm;
the v46 generation stays the default cache, and Tier 1 caches go to new
folders. Order:
(a) generator option for Qwen2.5-1.5B/3B fp16 or 7B 4-bit;
(b) contrastive prompts that show all options;
(c) attack verification by LLM scoring, filtering or regenerating
    non-compliant attacks (log the non-compliance rate η per split);
(d) 2–3 samples per hypothesis, averaged at embedding time.

One change per commit, each behind a config flag, v46 behaviour as the
default. Every result table reports the v46 baseline alongside. Same
procedure: propose, WAIT, implement, CPU smoke test, update docs, commit,
push.

Then prepare (do not run) a Colab section that:
1. audits 100 attacks from the CURRENT cache for η (before any change);
2. builds a Tier 1 cache on 500 ARC-Easy train items, new folder;
3. runs the pre-training gate: support–attack margin and each
   hypothesis-derived statistic vs chance, with 95% CIs, on those 500.
State the pass rule: proceed to a full Tier 1 cache only if the margin's
CI excludes chance. If it fails, report which of (a)–(d) to change next.
```

---

## Step 9 — Tier 2: learning signal (provisional decisions PD-2, PD-4)

```
Implement Tier 2 from docs/baseline/improvement_plan.md under provisional
decisions PD-2 and PD-4 in docs/baseline/supervisor_decisions.md
(provisional — pending supervisor review): every change behind a flag as
a separate arm; the v46 target, budget and optimizer groups stay the
defaults.
(1) A7 soundness-gated validator target, only as an extra arm behind a
    flag; the v46 target stays the default and the main baseline. Write it
    against the current code. Do not apply
    ~/qair_backup/v47_uncommitted.patch (CLAUDE.md); it may be read for
    reference. Reproduce its synthetic check;
(2) training budget config: full ARC-Easy train, 15–30 epochs, early
    stopping on validation;
(3) a separate optimizer group for phase and temperature scalars, with
    the learning rate set so rho can move 0.85 rad within the step budget
    (show the computation).
One change per commit, behind flags, same propose/WAIT/implement/test
procedure.

Then prepare (do not run) the Colab comparison: each change alone vs the
Step 7 baseline (the v46 baseline row in every table), on the best available cache, seeds 0/1/2, test once,
paired tests with Holm. Log C6/C7 diagnostics (collapse entropy vs log K,
within- vs across-channel energy-row cosine) before and after. Estimate
T4 time first.
```

---

## Step 10 — Tier 3: readout, abstention and the path Hamiltonian

Part A — readout and abstention:

```
Implement Tier 3 from docs/baseline/improvement_plan.md under provisional
decision PD-5 in docs/baseline/supervisor_decisions.md (provisional —
pending supervisor review): separate pre-registered arms; the v46 readout
stays the default. These are category C changes (CLAUDE.md rule 7): propose
each one and WAIT for my explicit confirmation.
(1) C1 signed-readout arm (document when it reduces to additive scoring);
(2) C2 rho = pi/2 initialization arm, keeping the rho = pi arm;
(3) C3 abstention: answer only when the conflict signal passes a
    threshold chosen on validation; report risk–coverage curves and AURC.
One change per commit, behind flags, same procedure. Prepare (do not run)
the Colab comparison against the best Tier 2 configuration and the v46
baseline, seeds 0/1/2, pre-registered predictions from the plan.
```

Part B — your multiverse idea:

```
Under provisional decision PD-5 in docs/baseline/supervisor_decisions.md
(provisional — pending supervisor review), as separate pre-registered arms
with the v46 readout as the default, implement the path Hamiltonian
exactly as specified in docs/baseline/path_hamiltonian_design.md: models/path_hamiltonian.py
(unary head, symmetric pairwise coupling head, Hamiltonian build, eigh
ground state, Born probabilities, answer marginals), a flag in
full_model.py, arms H0 (Gamma = 0), H1 (Gamma learned) and H2 (with its
H2s capacity control) in ablations.py, and the diagnostics in evaluate.py (Gamma, ||J||, gap,
entanglement entropy, ground-state/answer agreement). v46 path unchanged.

Propose the implementation first and WAIT. After approval: implement,
then run scripts/verify_path_hamiltonian.py and a CPU gradient check on
a 4-question smoke batch. Commit and push.

Then prepare (do not run) the Colab comparison: H0 vs H1 vs H2 vs the
best Tier 2/3 configuration vs the v46 baseline vs both LLM baselines, full ARC-Easy, seeds
0/1/2, test once, paired tests with Holm. Report the pre-registered
"quantum helps" prediction as held or falsified.
```

---

## Step 11 — Write-up

```
Read every results file in docs/baseline/ (baseline_results.md and all
later tier results), the issue dossier, the improvement plan and the path
Hamiltonian design. Draft docs/paper/draft.md:
title; abstract; introduction (the question: can reasoning paths in
superposition, with interaction between paths, improve multiple-choice
reasoning?); related work (only papers in the repo, verified); method
(v46 pipeline, the fixes, Tiers 1–3, the path Hamiltonian); experimental
setup (datasets, splits, seeds, baselines, statistics); results (tables
from the results files only); analysis (which predictions held or
failed); limitations (including what is and is not quantum); conclusion;
reproducibility statement (manifests, commits, caches).
Every number must come from a results file; label anything else. Then
write docs/paper/cse498r_report_outline.md mapping the same material to
the NSU report structure. Commit and push.
```
