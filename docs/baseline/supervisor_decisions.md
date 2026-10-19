# Supervisor decisions — PROVISIONAL

Recorded 2026-10-19. My supervisor is abroad and will review later. Until
then the decisions below are mine, made so the work can continue. **Every
decision here is provisional — pending supervisor review.** None of them is
a supervisor approval, and nothing in the repository may describe one as
"approved by the supervisor" until the review has happened.

Item IDs refer to `docs/baseline/issues.md` (A*, C*) and
`docs/baseline/improvement_plan.md` (Tiers, T*, D-*, H*).

## Rule for all decisions

**No change replaces v46 behaviour by default.** Every new behaviour sits
behind a flag or a separate arm whose default is off, and every result table
reports the v46 baseline (Tier 0) alongside.

- *Status:* provisional — pending supervisor review.
- *Reason:* if the supervisor rejects any item, the rejected arm is dropped
  and every other result still stands, because nothing was measured against a
  moved default.
- *How to revert:* nothing to revert in the code. Delete the rejected arm's
  flag, arm entry and rows from the result tables.

This rule refines improvement plan §0 and "Each tier's adopted change becomes
part of the reference". A change that wins on validation may serve as the
*comparison* reference for a later tier (Tier 0′, stacked arms). It never
becomes the code default, and the v46 baseline row stays in every table.

## Decisions

| # | Item | Provisional decision | Plan decisions covered |
|---|---|---|---|
| PD-1 | Tier 0: all fixes, the ARC-Easy cache build, the baseline runs | Proceed; no approval needed | D-1 |
| PD-2 | A7 soundness-gated validator target | Implement only as an extra arm behind a flag. The v46 target stays the default and the main baseline | D-5 (A7 part) |
| PD-3 | Tier 1: larger generator, contrastive prompts, attack verification, multiple samples | Proceed behind flags, each as a separate arm. v46 generation stays the default cache | D-4 |
| PD-4 | Tier 2: training budget, separate LR group for phase/temperature scalars | Proceed behind flags as separate arms | D-6 (T2b budget arms) |
| PD-5 | Tier 3: C1 signed readout, C2 ρ = π/2 arm, C3 abstention; path Hamiltonian arms H0/H1/H2 | Proceed as separate pre-registered arms. The v46 readout stays the default | D-7, D-8, D-9 (C3 part), D-12, D-13, D-15 |
| PD-6 | C4/C5 (Tier 4) | Deferred | D-9 (C4/C5 part) |
| PD-7 | Tier 0 epoch count | At most 20 epochs, early stopping patience 5 | D-2 |
| PD-8 | Which arm is "qAIR" | Primary F0 (full pipeline with LLM prior, the capstone's reported configuration); secondary A3 (no LLM prior) | D-3 |
| PD-9 | `amp_entropy` | Logged as a diagnostic only | D-5 (`amp_entropy` part) |
| PD-10 | Stopping rule for training | Early stopping on validation accuracy, patience 5, at most 20 epochs; pre-registered, never chosen after seeing results | D-10 (see note) |
| PD-11 | Primary success criterion | Primary qAIR arm beats both direct LLM baselines on full ARC-Easy test, 3 seeds, paired bootstrap with Holm correction, p < 0.05; v46 baseline reported alongside | D-11 |
| PD-12 | Path Hamiltonian pair head | Directed pair terms per `path_hamiltonian_design.md`, with a symmetric-only head as an ablation | D-14 |

Every D-item in `improvement_plan.md` §2 now has a provisional decision,
except the part of D-10 in the note under PD-10. PD-7 to PD-12 were recorded
2026-10-19, after PD-1 to PD-6.

### PD-1 — Tier 0: proceed

- *Status:* provisional — pending supervisor review.
- *Reason:* Tier 0 fixes bookkeeping, reproducibility and statistics. It
  does not change what v46 computes, so it is the measurement every other
  decision depends on.
- *How to revert:* the Tier 0 fixes are separate commits on `baseline`. Revert
  them one by one with `git revert <hash>` (hashes in `issues.md` §1). The
  cache and run outputs are untracked files and can be set aside.
- CLAUDE.md still applies: any T4 run over about 1 hour needs my go-ahead
  first.

### PD-2 — A7: extra arm behind a flag

- *Status:* provisional — pending supervisor review.
- *Reason:* A7 changes the training objective. As an extra arm it can be
  measured against the unchanged v46 target, which stays the default and the
  main baseline.
- *How to revert:* remove the flag and the arm. The default path is the v46
  target, so no other result changes.
- CLAUDE.md's rule "never apply the v47 patch without supervisor approval" is
  **not** changed by this record. The arm is written against the current code;
  the v47 patch may be read for reference but not applied.

### PD-3 — Tier 1: flags, separate arms, v46 cache stays the default

- *Status:* provisional — pending supervisor review.
- *Reason:* each option (T1a generator, T1b contrastive prompts, T1c attack
  verification, T1d M samples) changes the evidence the model sees. As
  separate flagged arms each is a single-variable comparison against the v46
  cache.
- *How to revert:* drop the flag and its cache folder. Tier 1 caches are
  written to new folders; the v46-generation cache is never overwritten.

### PD-4 — Tier 2: flags, separate arms

- *Status:* provisional — pending supervisor review.
- *Reason:* a longer budget (T2b) and a separate LR group (T2c) change
  training dynamics, not the architecture. As separate arms they are compared
  with the v46 recipe.
- *How to revert:* remove the flags. The defaults are the v46 epoch count and
  the v46 optimizer groups.
- Runs over 1 hour (e.g. the 30-epoch arms) still need my go-ahead
  (CLAUDE.md).

### PD-5 — Tier 3 and the path Hamiltonian: pre-registered arms

- *Status:* provisional — pending supervisor review.
- *Reason:* C1, C2, C3 and H0/H1/H2 change the readout, the central claimed
  mechanism. Pre-registered separate arms keep the v46 Born readout as the
  default and the reference, so each result is a falsifiable test, not a
  redefinition of qAIR.
- *How to revert:* remove the arms and the default-off flags (`readout="path"`
  for H0/H1/H2; the C1/C2/C3 flags). `models/path_hamiltonian.py` is a new file
  and can be deleted. C3 risk–coverage curves are measurement only; C3 pruning
  is not part of this decision.
- The prediction for each arm is written into the plan before the run (plan
  §0 rule 6); H2 requires its H2s capacity control (plan D-13).
- CLAUDE.md rule 7 ("Category C issues need supervisor approval before any
  change") is **not** changed by this record. Under it, the C1–C3 arms are
  proposed and implemented only after I explicitly confirm each one in the
  session, as with any other change.

### PD-6 — C4/C5: deferred

- *Status:* provisional — pending supervisor review.
- *Reason:* C4 (encoder) and C5 (cleanup) change the pipeline and parameter
  counts. They come after the Tier 1 to Tier 3 results.
- *How to revert:* nothing to revert; take them up when the supervisor
  approves.

### PD-7 — Tier 0 epoch count: at most 20, patience 5

- *Status:* provisional — pending supervisor review.
- *Reason:* the v46 run trained for 8 epochs on 80 questions (about 40
  optimizer steps, dossier B8); a fixed cap with early stopping on validation
  gives every arm the same, pre-stated budget on full data.
- *How to revert:* set the cap and patience back to the supervisor's choice
  in `config.py` (`EPOCHS`, `PATIENCE`; today 30 and 5) and rerun Tier 0.
  Checkpoints record both values in their provenance, so runs under either
  budget stay distinguishable.
- Consequences recorded in the plan: Tier 0 is no longer "v46 as trained" in
  epoch count (the v46 run used 8; `config.py` default is 30). T2b's budget
  arms were specified against 8 epochs and need re-specifying before Step 9.
  At the assumed 2.4–7 min per epoch [ASSUMPTION, plan §0.2] a 20-epoch run
  may exceed 1 hour on T4 and then needs my go-ahead (CLAUDE.md).

### PD-8 — qAIR arm: F0 primary, A3 secondary

- *Status:* provisional — pending supervisor review.
- *Reason:* F0 is the configuration the capstone reported. A3 is always
  reported next to it, because F0 contains the cloze score as a feature and
  A3 shows whether the reasoning adds anything without it.
- *How to revert:* swap the arm named in plan §3 item 1. Both arms are run
  in every comparison, so no rerun is needed.

### PD-9 — `amp_entropy`: diagnostic only

- *Status:* provisional — pending supervisor review.
- *Reason:* it is a measurement of the collapse, useful as a reference value
  for later tiers; it never enters the loss or model selection.
- *How to revert:* remove the metric from the logging. No result depends on
  it.
- It is written against the current code; the v47 patch is reference only
  (CLAUDE.md, PD-2).

### PD-10 — Training stopping rule: pre-registered early stopping

- *Status:* provisional — pending supervisor review.
- *Reason:* the stopping rule is fixed before any run so it cannot be chosen
  after seeing results (plan §0 rules 4 and 6): validation accuracy, patience
  5, at most 20 epochs, the same for every arm and seed.
- *How to revert:* change the rule in the plan and rerun; the history in
  every checkpoint records the epoch each run stopped at.
- **Note:** plan D-10 asked a different question, the *research* stopping
  rule: if no Tier 1 cache passes the pre-training gate, report D1 (no signal
  in the hypotheses) as the result instead of continuing architecture work.
  PD-10 records the training stopping rule. The research stopping rule is
  still open.

### PD-11 — Primary success criterion

- *Status:* provisional — pending supervisor review.
- *Reason:* fixed before any test-set evaluation (plan §0 rule 4), so the
  test is evaluated once and the claim cannot move after the result.
- *How to revert:* edit plan §3 before any test evaluation. After a test
  evaluation, a changed criterion is reported as a post-hoc change.
- Plan §3 item 4 also requires qAIR to beat each baseline in all 3 seeds and
  reports per-seed McNemar p-values. PD-11 does not mention these; until
  decided otherwise they stay in §3 as stated.

### PD-12 — Directed pair head, symmetric-only ablation

- *Status:* provisional — pending supervisor review.
- *Reason:* the directed (a, b, r) pair terms are the design in
  `path_hamiltonian_design.md`; the symmetric-only head (arm H1-sym) tests
  whether the direction matters.
- *How to revert:* make the symmetric head the default and keep the directed
  one as the ablation; both are built behind the default-off `readout="path"`
  flag (PD-5).

## After the supervisor's review

For each row: change "provisional — pending supervisor review" to the
supervisor's decision and its date. If the decision differs, record the
difference and the action taken (arm removed, flag default kept, etc.). Then
update `improvement_plan.md` §2 and the Step 8–10 prompts.
