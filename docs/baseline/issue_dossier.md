# qAIR v46 — issue dossier (the defense document for every fix)

Branch `baseline` (code byte-identical to v46 `9552498`, tag `capstone-final`).
Written 2026-10-17. **No issue is fixed by this document.** It states, for each
issue, what we observed, why it happens, the mathematics, the proposed fix, why
the fix works, and how we will show it worked.

**Labels.** [OBSERVED: v46 run] = saved output of the executed notebook
`notebooks/qAIR_v46.ipynb` (cell numbers 0-based). [VERIFIED: …] = a file:line
or a command run in this session. [DERIVED] = mathematics shown here; every
checkable [DERIVED] claim is checked by `scripts/verify_dossier_math.py`
(output in Appendix A). [HYPOTHESIS] = plausible, not yet tested.
[ASSUMPTION] = taken as given.

**The run is the baseline code.** The notebook cloned `130ff00` (cell 6 prints
`130ff00 (HEAD -> main…) v46: add notebooks/qAIR_v46.ipynb`), and
`git diff 130ff00 9552498` changes only `.DS_Store`, `.gitignore`, `gitlog.txt`
and checkpoints [VERIFIED: git diff --stat]. So every observation below is an
observation of the code on this branch.

**Literature.** No papers are stored in the repo. In the parent folder I read
pp. 1–2 of two PDFs. I cite only one, for experimental practice:
Flores-Garrigós et al., "Quantum combinatorial reasoning for large language
models", *Phys. Rev. Research* 8, 033070 (2026). Its abstract compares against
majority vote and self-consistency at a matched sampling budget, and calls its
own small quantum-over-classical gain "not conclusive" and possibly
"statistical fluctuations". The other PDF (CONIT 2025, LLM-driven engineering
optimization) is not relevant to any derivation here and is not cited. All
mathematics below is derived from the code itself.

---

## 0. One-page summary for my supervisor

**What the v46 run actually showed.** On 80 training and 20 validation
questions, the full model reached 50% validation accuracy (10 of 20). The
language model on its own got 40% (8 of 20). That difference is 2 questions
one way and 4 the other: statistically nothing (p = 0.69), and 20 questions
cannot detect anything smaller than about ±22 points. Nothing in this run is a
result. The notebook says so itself ("It is a plumbing test").

**What is broken (bugs — fix first, no approval needed except where noted):**
- **A1/A2:** 7 of the model's 118 parameter tensors never learn, because they
  never reach the loss. One temperature is completely dead.
- **A3/A8:** Runs silently resume old checkpoints with the same name. In the
  v46 ablation grid, one arm was skipped as "already finished" and three others
  resumed from checkpoints made in earlier sessions with unrecorded code. The
  grid numbers do not describe v46.
- **A4, A5, A6:** Training history lost on resume. The input-ablation test can
  rebuild the wrong model. The cache can silently change state.
- **A7 (needs approval):** The validator is trained to give every hypothesis
  the same target, so it cannot learn which hypothesis is sound. A candidate fix
  exists but changes the training objective.
- **E3, E7:** Generating the hypothesis cache is not reproducible, and a capped
  cache can silently restart hours of generation. Both must be fixed before the
  ARC-Easy cache is built.

**What is merely noisy or under-powered (evaluation, not bugs):**
- n=20, one seed, about 40 optimizer steps (B1, B2, B8).
- The "best" checkpoint is picked on the same 20 questions it is scored on (B3).
- The test split was never used.
- Two of the required baselines do not exist yet: the LLM picking a letter (B4)
  and majority vote (B5).
- The statistics script pools seeds incorrectly (B6).
- The phase parameters barely moved, but this run could have moved them by at
  most about 0.011 radians whatever the gradient said (B8). So "the phases don't
  learn" is **not** established.

**What is a design limitation (needs supervisor discussion):**
- **C1:** The "quantum" readout scores an option by the *size* of the
  support-minus-attack difference, not its sign. A strongly refuted option can
  tie a strongly supported one.
- **C2:** The interference metric reads 100% at initialization by construction,
  so it says nothing about learning.
- **C6:** The collapse over hypotheses is almost exactly uniform (0.003 nats
  from uniform), so the model does not select hypotheses.
- **C7:** Hypotheses produce increasingly similar option scores.
- **C3–C5:** No pruning or abstention. Everything passes through frozen MiniLM
  embeddings. Some redundant computation.

**What is a data limitation:**
- **D1/D2:** The 0.5B generator's hypotheses carry no usable signal on the 80
  training questions: all three zero-shot tests score at or below chance, and none is distinguishable from it.
- **D5:** Attacks sometimes *support* their option.
- Most accuracy rides on the LLM likelihood prior: removing it drops accuracy to
  chance.

**Fix order.** (1) Bugs and reproducibility: E3, E7, A6, A3/A8, A4, A5, A1/A2,
regression tests; A7 after approval. (2) Remaining engineering. (3) Build the
ARC-Easy cache (with your go-ahead). (4) Evaluation fixes and full-split,
3-seed runs. (5) Data. Design items wait for the supervisor.

---

## 1. Training-run diagnosis — the v46 run read as one story

Setup [OBSERVED: v46 run, cells 10, 12, 14, 22]: ARC-Challenge, 80 train /
20 val from capped caches (`complete=False`), 8 epochs × 5 steps (batch 16),
`set_seed(42)` with `deterministic=False`, circuit on `lightning.gpu`,
1,056,382 parameters (54 circuit weights), 13,205 parameters per training
example. Optimizer [VERIFIED: `ckpt/qair_v46_best.pt` param_groups and
scheduler]: AdamW, weight decay 0.05, base LRs 5e-4 / 2e-4, cosine to 1e-6 over
`T_max = 8` epochs.

### F1 — Epoch 1 is an untrained model; training then fits the 80 items without moving validation accuracy → B1, B8, B11, (overfitting)
- Train loss 1.889 → 0.7924 by epoch 7 (−58.1%) → 0.7425 by epoch 8 (−60.7%).
  Validation accuracy across the 8 epochs: 0.35, 0.35, 0.40, 0.45, 0.40, 0.40,
  0.45, 0.50 [OBSERVED].
- The loss is `CE_smoothed + 0.4·ranking + 0.1·BCE − 0.02·tanh(diversity)`
  (`training/losses.py:150-155`). At a uniform posterior over 4 options the
  terms are $\ln 4 + 0.4\cdot1 + 0.1\ln 2 \approx 1.855$, against the
  observed epoch-1 mean of 1.889. Epoch 1 is therefore essentially the
  untrained model [DERIVED; check 8]. The floor is the label-smoothed CE
  entropy, 0.201.
- Every validation value lies within one 95% half-width (±0.22, F8) of their
  mean 0.41. The curve is consistent with **no change**.
- [HYPOTHESIS] The model memorizes: 13,205 parameters per example, falling loss,
  flat validation accuracy. (ECE is not usable evidence at n=20; see F2.) Train accuracy was not logged, so
  the classic train/val gap cannot be shown directly. That is part of E6/B11.

### F2 — ECE rises, but at n=20 ECE is mostly estimator noise → B11
- ECE 0.161, 0.162, 0.206, 0.296, 0.307, 0.292, 0.335, final 0.385 [OBSERVED].
  Over the same epochs accuracy did not change measurably (F1).
- **Caveat that dominates [DERIVED; check 10]:** a model that is calibrated by
  construction has an *expected* 15-bin ECE of **0.241 at n=20** (0.047 at
  n=570, 0.023 at n=2376), assuming confidences spread U(0.3, 0.9)
  [ASSUMPTION]. The epoch-1 value 0.161 is *below* that, and the final 0.385
  is about 1.6× it. Neither the level nor a 7-point trend is reliable evidence
  of miscalibration at n=20.
- Two mechanisms could push confidence up without evidence. They are reasons
  to measure calibration properly, not findings [DERIVED for the training set;
  the transfer to validation is HYPOTHESIS]:
  1. **Margin loss.** The margin term requires
     $\log p_y - \log p_j \ge 1$ for every distractor on *training* items,
     i.e. $p_y \ge e/(e+3) \approx 0.475$. This is satisfied only by growing
     logit gaps.
  2. **Born normalization is scale-free** (check 9). Near-cancelled residuals
     (0.010, 0.020, 0.015, 0.012) give P = (0.12, 0.46, 0.26, 0.17). Tiny
     absolute differences become large probability differences.

### F3 — The collapse barely selects: it is within 0.003 nats of uniform → C6, A7, B8
- Collapse entropy 2.0704 → 2.0763 against $\log 8 = 2.0794$ [OBSERVED]. Since
  $\mathrm{KL}(p\,\|\,U) = \log K - H(p)$ [DERIVED; check 1], the distance to
  uniform is **0.0090 nats at epoch 1 and 0.0031 at the end**. The model
  became *less* selective as it trained.
- Collapse peak 0.149 → 0.141, against 1/8 = 0.125 for uniform.
- What "selective" would require [DERIVED; check 1]:

  | target | entropy | KL to uniform |
  |---|---|---|
  | observed | 2.0763 | 0.0031 |
  | 0.5 mass on one hypothesis, rest uniform | 1.666 | 0.413 |
  | uniform over the 4 sound hypotheses | 1.386 | 0.693 |

  A 0.5 peak needs a logit gap of $\ln 7 = 1.95$. Observed final KL is 132×
  smaller than that reference.
- The logged `diversity` is the variance of the collapse energy across
  hypotheses. With logits $-E_k/T$,
  $\mathrm{KL} \approx \mathrm{Var}(\ell)/2 = \text{diversity}/(2T^2)$. This
  gives $T = 1.003$ (epoch 1) and $1.033$ (epoch 7), against an initial
  $T = 0.3 + \mathrm{softplus}(0) = 0.993$. The entropy is fully explained by
  energy differences of standard deviation $\sqrt{0.0067} = 0.082$, with the
  temperature still at its initial value. Gaussian logits of that variance
  predict a peak of 0.1401 against the observed 0.1406 [DERIVED; check 1].

### F4 — Hypotheses produce more and more similar option scores → C7, D1
- Energy-row pairwise cosine 0.769 → 0.838, and **0.97** in the A3 arm.
  Diversity 0.0182 → 0.0067. Post-reasoning hypothesis cosine flat at
  0.87–0.88 [OBSERVED].
- The hypothesis vectors stay distinct while their option-energy profiles
  converge. This is the pattern the code's own diagnostic attributes to "the
  SELECTOR ignoring them … what a low-signal cache looks like"
  (`training/train.py:303-318`) [VERIFIED: code]. It is consistent with D1.

### F5 — The interference metrics are saturated by initialization, not by learning → C2
- `destructive_fraction` = 1.0000 in the pre-training sanity check (cell 24) and
  in every epoch [OBSERVED].
- With the polarity phase at π and near-uniform amplitudes, it is 1 by
  construction for every sample and every option. The threshold is
  $\rho^\* = \arccos(-(N-1)/N) = 2.419$ rad for N=4. The simulation of the exact
  readout reproduces `config.py`'s measured step (0.479 vs 0.494 at 2.4 rad;
  0.996 vs 0.994 at 2.8 rad) [DERIVED; check 4].
- `phase_effect` ≈ 0.99 says only that the coherent sum differs from the
  in-phase sum, which near-total cancellation guarantees. The sanity check
  "destructive interference occurs" (`> 0`) passes by construction.

### F6 — Learned scalars barely move, and could not have moved much → B8 (not a separate defect)
- `polarity_phase` 3.1421, 3.1429, 3.1427, 3.1418, 3.1410, 3.1402, 3.1398, 3.1396
  (init π = 3.1416). `energy_alpha` 0.4999 → 0.4988; `validator_alpha`
  0.5001 → 0.5002; `dt` 0.1000 → 0.1003; `anchor` 0.4998 → 0.4987;
  `llm_prior` 0.3135 → 0.3136 [OBSERVED].
- Adam moves a parameter by about LR per step. Under this schedule the
  40-step budget is $\sum_t \mathrm{lr}_t = 0.0113$ rad.
- `polarity_phase` travelled 0.0046 (41% of the budget), net −0.0020. Weight
  decay alone predicts −0.0018 [DERIVED; check 5].
- Moreover, π is an exact stationary point of the readout in ρ when the
  channels are phase-aligned (C2) [DERIVED; check 5].
- **Conclusion:** "parameters never move" is what *any* gradient would produce
  with this budget. It is subsumed by B8 and not added as a new issue. The one
  true exception, a parameter with *zero* gradient, is A1.

### F7 — Accuracy rides on the LLM prior → C4, B4
- Input ablation from 0.50 intact [OBSERVED cell 34]:

  | corruption | accuracy |
  |---|---|
  | `llm_logprob := 0` | **0.25** (chance) |
  | H := zeros | 0.30 |
  | H := noise | 0.30 |
  | H shuffled | 0.35 |
  | H rolled | 0.25 |
  | support := 0 | 0.30 |
  | attack := 0 | 0.40 |
  | polarity := 0 | 0.45 |
  | align := 0 | 0.45 |
  | O := 0 | 0.40 |

- The cached LLM likelihood alone scores 0.40. Data-ceiling probes on 80 train
  items are at chance for "O only" and "Q×O" (0.25), so the probes cannot learn
  from 80 items either [OBSERVED cell 37].
- At n=20 none of these deltas is significant. Even 5 net flips with no reverse
  flips give exact McNemar p = 0.0625 [DERIVED; check 6]. "Rides on the prior"
  is consistent with the data, not demonstrated.

### F8 — What n=20 can and cannot detect → B1, B3, honesty list
- 95% half-width $1.96\sqrt{p(1-p)/n}$ at p=0.5 [DERIVED; check 6]:

  | n | 95% half-width |
  |---|---|
  | 20 | ±0.219 |
  | 570 (ARC-Easy val) | ±0.041 |
  | 1172 (ARC-Challenge test) | ±0.029 |
  | 2376 (ARC-Easy test) | ±0.020 |

- Sub-chance arms are consistent with chance: A3 at 3/20 has
  $P(X\le3\mid 20, 0.25) = 0.225$; P0 at 4/20 has $P(X\le4) = 0.415$.
- Main result vs direct LLM: b=2, c=4, exact McNemar p = 0.6875. Bootstrap
  95% CI for 10/20 is [0.30, 0.70] [DERIVED; check 6].
- Best-of-8-epochs selection inflates the reported number. With a true
  accuracy of 0.41, the expected maximum over 8 epochs is +0.10 (correlated
  epochs) to +0.16 (independent) above it, and under independent epochs
  $P(\text{best} \ge 0.50) = 0.93$ [DERIVED; check 7].

### F9 — The ablation grid did not measure v46 → A8, A3, E6
- Grid log [OBSERVED cell 39]:
  - A1: `[RESUME A1_baseline_s42] Resuming from epoch 8 … [SKIP] … already
    finished 8 epochs` — no v46 training at all.
  - A3: resumed from epoch 7; P0 from epoch 6; F0 also resumed.
  - All from `/content/drive/MyDrive/qAIR_V45/ckpt/`.
  - Cell 41 then reports `No test_records found`.
- The resumed arms ran 1–2 epochs of v46 code on weights from unrecorded
  earlier sessions.

### F10 — The run is not reproducible as recorded → E3, E10, E11, E6
- The cache was loaded, not built, in this session (`[CACHE FOUND]`, cell 14).
  How and with what RNG state it was generated is unrecorded.
- Training ran with `deterministic=False` on GPU.
- The environment had resolver conflicts (`huggingface-hub 0.33.4` vs
  `diffusers 0.39.0` / `gradio 6.20.0`) and a PennyLane warning about
  JAX 0.7.2 > 0.6.2 (cells 8, 10) [OBSERVED].

---

## 2. One section per issue

Order: A (bugs), B (evaluation), E (engineering), D (data), C (design — all
"needs supervisor discussion"). New IDs found in this dossier are marked
**(new)** and have been added to `issues.md`.

Shared notation (v46, `models/interference.py`, `models/full_model.py:378-406`):
$K = 2N$ hypotheses, support set $\mathcal S$ (polarity +1), attack set
$\mathcal A$ (polarity −1). Hypothesis weight
$p_k = r_k^2 = \mathrm{softmax}_k(-E_k/T_h)$, compatibility
$t_{kn} \propto e^{-E_{kn}/2T_o}$, $w_{kn} = r_k t_{kn} \ge 0$, phase
$\phi_k = \varphi_k + \rho\,\mathbb 1[k\in\mathcal A]$ ($\varphi_k$ from the
circuit, $\rho$ = `polarity_phase`), compatibility phase $\theta_{kn}$.
Option amplitude $A_n = \sum_k w_{kn} e^{i(\phi_k+\theta_{kn})}$, readout
$P(n) = |A_n|^2/\sum_m|A_m|^2$, decohered mixture
$C_n = \sum_k w_{kn}^2$. Channel sums:
$S_n = \sum_{k\in\mathcal S} w_{kn}e^{i(\varphi_k+\theta_{kn})}$,
$R_n = \sum_{k\in\mathcal A} w_{kn}e^{i(\varphi_k+\theta_{kn})}$, so
$A_n = S_n + e^{i\rho}R_n$.

### Part A — Bugs

#### A0 Input-ignoring circuit (Hadamard + default `rotation="X"`) — verified ABSENT in v46
- **Category / severity / blocks:** A / none in v46 / nothing; regression test pending.
- **Observation:** v46 per-observable output std over 64 random inputs is
  0.036–0.30, and outputs at two inputs differ by up to 0.74. v40 (local `main`)
  gives std ≈ 1e-16 [VERIFIED: scratch PennyLane 0.42.3 test, issues.md A0].
  The v46 run's own sanity check: `[PASS] circuit state responds to input`,
  `phase spread 0.1126` [OBSERVED cell 24].
- **Location:** `models/quantum_layer.py:178-181`:
  `qml.Hadamard(wires=i)` … `qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")`.
- **Root cause (of the v40 bug):** RX applied to |+⟩, the +1 eigenstate of X, is
  a global phase.
- **Theory:** $R_X(t)|+\rangle = e^{-itX/2}|+\rangle = e^{-it/2}|+\rangle$, so
  every expectation $\langle\psi|O|\psi\rangle$ is independent of $t$.
  $R_Y(t)|+\rangle$ rotates the Bloch vector in the x–z plane, so
  $\langle X\rangle = \cos t$ depends on the input [DERIVED].
- **Proposed fix:** none needed. Commit the scratch test as
  `tests/test_circuit_input_dependence.py` (Phase 1, item 9).
- **Alternatives considered:** relying on the notebook sanity check alone.
  Rejected: it uses inputs of scale 0.05 and only one pair.
- **Why the fix works:** the test fails on v40 and passes on v46, so it
  discriminates.
- **Falsifiable prediction:** any future edit that reintroduces a constant
  circuit makes the test report std < 1e-6.
- **Verification test:** the test itself on CPU (`default.qubit`), 64 inputs,
  3 weight draws.
- **Risks:** none. **Needs supervisor approval?** No.

#### A1 `CoherentCollapse.hypothesis_temp` is a dead parameter
- **Category / severity / blocks:** A / Medium / blocks DoD §2.2 (gradient coverage).
- **Observation:** drift exactly 0.000000 in all five v46-run checkpoints
  [VERIFIED: checkpoint read]. The v46 optimizer state lacks exactly these 7
  tensors: `collapse.confidence.{0,2}.{weight,bias}`,
  `interference.hypothesis_temp`, `phase_head.{weight,bias}`. That is 111 of
  118 tensors with state [VERIFIED: `ckpt/qair_v46_best.pt`], matching the
  notebook's `111 tensors with grad` [OBSERVED cell 24].
- **Location:** `models/full_model.py:405`,
  `hyp_temperature=collapse_out["temperature"],`; and `models/interference.py:138-141`:
  ```python
  if hyp_temperature is not None:
      t_hyp = hyp_temperature
  else:
      t_hyp = 0.3 + F.softplus(self.hypothesis_temp)
  ```
- **Root cause:** the caller always supplies the per-sample temperature, so the
  `else` branch, the only use of `hypothesis_temp`, is unreachable.
- **Theory:** $\partial L/\partial\tau = 0$ identically, because $\tau$ is not
  in the computational graph. PyTorch leaves `grad = None`, and AdamW skips
  parameters whose grad is None, *including weight decay*. That is why the
  drift is exactly 0 and not merely small, while all other scalars at least
  show the decay-scale drift of F6 [DERIVED from the AdamW update rule].
- **Proposed fix:** keep one temperature source, either (a) delete the
  parameter, or (b) use it as a learned base,
  $T = (0.3 + \mathrm{softplus}(\tau)) \cdot \mathrm{softplus}(z)/\ln 2$.
  Shapes unchanged (scalar; temperature (B,1)). (a) removes a state_dict key
  and needs `strict=False` or a key-dropping loader for old checkpoints
  (rule 9). (b) keeps the key, but the forward pass changes.
- **Alternatives considered:** excluding it from the optimizer. That silences
  the symptom but leaves dead code that misleads readers.
- **Why the fix works:** after (a) no unreachable parameter remains. After (b)
  $\tau$ is on the forward path, so $\partial L/\partial\tau \ne 0$ in general.
- **Falsifiable prediction:** the gradient-coverage test lists 0 parameters
  with `grad is None` for this arm (currently 7 including A2). If it still
  lists `hypothesis_temp`, the fix is wrong.
- **Verification test:** CPU unit test. Build the model, run one forward/backward
  on a synthetic batch, and assert every optimizer parameter has a finite
  non-None grad.
- **Risks:** (b) changes the collapse temperature, so it is not comparable to
  capstone numbers. **Needs supervisor approval?** (a) no; (b) yes (forward pass).

#### A2 Other parameters that never receive gradient
- **Category / severity / blocks:** A / Low–Medium / DoD §2.2.
- **Observation:** see A1. Missing optimizer state for `collapse.confidence.*`
  (4 tensors) and `phase_head.*` (2 tensors) in the default config [VERIFIED].
- **Location:** `models/collapse.py:85`,
  `confidence = self.confidence(z_global).squeeze(-1)`, returned only as
  `collapse_confidence` (`models/full_model.py:470`) and never used in
  `training/losses.py`. `models/full_model.py:384-387`:
  `if self.phase_source == "circuit" and quantum_phase is not None: phase_h = quantum_phase`
  `else: phase_h = bounded_phase(self.phase_head(H)…)`.
- **Root cause:** modules built unconditionally but used only by some
  configurations (`phase_head`) or by none (`confidence`).
- **Theory:** same as A1. Off the loss path means zero gradient and no update.
  It also biases the reported parameter count: 1,056,382 includes
  545 (`confidence`: 32·16+16 + 16·1+1) + 129 (`phase_head`: 128+1)
  + 1 (`hypothesis_temp`) = **675** parameters that cannot learn [DERIVED from
  the layer shapes in `models/collapse.py:55-60`, `models/full_model.py:203`].
- **Proposed fix:** build `phase_head` only when `phase_source == "learned"` or
  `backend == "none"`. Remove the `confidence` head or use it. Report trainable
  (grad-receiving) parameter counts per arm. Removing keys affects checkpoint
  loading (rule 9), so add a loader that drops unknown keys and logs them.
- **Alternatives considered:** leave them and document. Rejected: the
  parameter-matched comparisons (A1b vs A1c) count them.
- **Why the fix works:** removes the parameters, not the symptom.
- **Falsifiable prediction:** grad-coverage reports 118 → 111 tensors, all with
  grad; per-arm counts differ exactly by the removed heads.
- **Verification test:** the A1 unit test, run for every arm in
  `training/ablations.py`.
- **Risks:** parameter counts change, which affects matched-parameter claims.
  **Needs supervisor approval?** No (no forward-path change).

#### A3 Run name lacks the seed; resume ignores config
- **Status:** **fixed 2026-10-19** in `f605c55` (evidence in
  issues.md A3). **Change from the proposal below:** resume compares a code
  fingerprint (sha256 of tracked `*.py` + `requirements.txt` contents, plus
  the dirty-diff hash) instead of the git commit, which is recorded for
  information only, so a docs-only commit does not block a resume.
- **Category / severity / blocks:** A / High / blocks multi-seed runs (DoD §4.1).
- **Observation:** a single-run name has no seed (`name="qair_v45.1"`,
  `training/runner.py:125`). The same name-keyed resume mechanism produced the
  grid behaviour in A8 [OBSERVED cell 39].
- **Location:** `training/checkpoint.py:15`,
  `latest_ckpt = os.path.join(ckpt_dir, f"{name}_latest.pt")`; `:45-46`,
  `start_epoch = ckpt["epoch"] + 1`; `training/runner.py:175` passes the
  seed-free `name`.
- **Root cause:** the checkpoint key is a function of the name only, and no
  config, seed, code or cache check guards the restore.
- **Theory (exact causal argument):** let $\pi(\cdot)$ map a run to its path.
  For `--mode train`, $\pi(\text{seed}=42) = \pi(\text{seed}=43)$ because
  `name` does not contain the seed. `load_or_resume` restores model, optimizer,
  scheduler and $e_0 = e_{\text{saved}}+1$. If $e_{\text{saved}} = E-1$, the
  loop `range(e_0, E)` is empty. Zero steps run, and the test evaluation that
  follows reports seed 42's weights under seed 43. If
  $e_{\text{saved}} < E-1$, the run continues seed 42's weights on seed 43's
  data order, a hybrid that is neither run.
- **Proposed fix:** make the run key
  `f"{arm}_{dataset}_s{seed}"`. Store `{config, seed, git commit, cache hash}`
  in every checkpoint, and refuse to resume on any mismatch (raise with both
  values). No tensor shapes change. Checkpoints gain metadata keys; old ones
  load with a warning and are never auto-resumed.
- **Alternatives considered:** a fresh `ckpt_dir` per run. Rejected as the only
  defense, because it fails silently when someone reuses a directory, as A8
  shows happened.
- **Why the fix works:** distinct seeds map to distinct paths (injective key),
  and identical paths with different provenance raise instead of resuming.
- **Falsifiable prediction:** two 1-epoch smoke runs with seeds 42 and 43 in one
  directory give two checkpoint files and different epoch-1 losses. Resuming
  with a changed config raises.
- **Verification test:** CPU smoke test, ≤20 examples, separate cache dir, fake
  or tiny cache.
- **Risks:** old checkpoint names no longer resume. That is intended.
  **Needs supervisor approval?** No.

#### A4 Resumed training history is discarded
- **Status:** **fixed 2026-10-19** in `4083ecf` (evidence in
  issues.md A4). History and early-stopping state are stored in the
  checkpoint itself rather than merged from `_history.pt`. The prediction
  "stops at the same epoch as an uninterrupted run" holds for the
  bookkeeping (tested with scripted validation accuracy); the training
  computation after a resume is not identical (A11).
- **Category / severity / blocks:** A / Medium / blocks DoD §1.6.
- **Observation:** `A3_persistent_s42_latest.pt` is at epoch 7 but its history
  file holds 1 epoch [VERIFIED: checkpoint read]. Consistently, the grid printed
  only the post-resume epochs [OBSERVED cell 39].
- **Location:** `training/runner.py:175`, `start_epoch, best_acc, _ = load_or_resume(…)`;
  `training/train.py:132` builds a fresh `history = {…}`; `:203`
  `self.save_history(history)` overwrites the file every epoch; `:147`
  `epochs_no_improve = 0`.
- **Root cause:** the resumed history is returned but dropped, and early-stopping
  state is not checkpointed.
- **Theory:** after resume at $e_0$, the saved history is
  $h[e_0..e]$, not $h[0..e]$. Every per-epoch curve (drift, ECE, entropy) used
  for diagnosis is truncated. With patience $P$, early stopping fires at
  $e_0 + P$ rather than $e_{\text{last best}} + P$ [DERIVED].
- **Proposed fix:** initialize `history` from the resumed history, append, and
  persist `epochs_no_improve` in the checkpoint. No shape changes; one new
  checkpoint key, defaulting to 0 when absent.
- **Alternatives considered:** writing one history file per epoch. Rejected as
  more files for the same information.
- **Why the fix works:** the history becomes the concatenation, and the stopping
  state is restored exactly.
- **Falsifiable prediction:** an interrupted-then-resumed smoke run has
  `len(history["acc"]) == epochs`, and stops at the same epoch as an
  uninterrupted run with the same seed (on CPU, deterministic).
- **Verification test:** CPU smoke test (interrupt after epoch 2, resume).
- **Risks:** none. **Needs supervisor approval?** No.

#### A5 `input_ablation` rebuilds the wrong architecture without a saved config
- **Status:** **fixed 2026-10-19** in `b00595a` (evidence in
  issues.md A5). The falsifiable prediction held on CPU smoke checkpoints:
  "intact" accuracy equals the logged validation accuracy exactly for three
  arms; altering the stored config breaks the equality.
- **Category / severity / blocks:** A / High / DoD §2.4.
- **Observation:** ablation runs write no `_config.pt`; only `qair_v45` and
  `qair_v46` have one [VERIFIED: `ls ckpt`].
- **Location:** `evaluation/input_ablation.py:252-258`:
  `cfg = {"use_quantum": True, "use_validator": True, "persistent_steps": 3}`
  as fallback; `:232` `use_llm_prior=cfg.get("use_llm_prior", True)`.
- **Root cause:** a silent default config, plus identical parameter sets across
  arms, so `load_state_dict(strict=True)` cannot detect the mismatch.
- **Theory:** every arm instantiates the same 118 tensors (A1/A2: heads exist
  whether used or not). So for an arm $a$ trained with
  $(\text{prior}=\text{off})$, loading into a model built with
  $(\text{prior}=\text{on})$ succeeds, and the forward computes
  $\log P_a(n) + \lambda\,z(\ell_n)$ with $\lambda = \mathrm{softplus}(-1) =
  0.313$, a model never trained. Every corruption delta is then measured on the
  wrong function.
- **Proposed fix:** store the resolved config in every checkpoint (shared with
  A3), and make `input_ablation` raise when it is missing.
- **Alternatives considered:** infer the config from parameter values. Rejected:
  it is impossible here, because the parameter sets are identical.
- **Why the fix works:** the model is rebuilt from the config it was trained
  under.
- **Falsifiable prediction:** "intact" accuracy in `input_ablation` equals the
  training log's validation accuracy for that checkpoint, exactly (same items,
  eval mode).
- **Verification test:** run on an A3 smoke checkpoint; assert equality and
  assert a missing config raises.
- **Risks:** none. **Needs supervisor approval?** No.

#### A6 Cache edge cases silently change state or data
- **Status:** **fixed 2026-10-18** in `0a4f464` (evidence in
  issues.md A6). **Versioning decision:** the per-sample `score_failed` flag
  was added **without** bumping `CACHE_VERSION` past 46, because 46 was
  introduced by E3 the same day and no real v46 cache had been built; the
  loader defaults the flag to `False` when absent (v45 caches). This replaces
  the "bump `CACHE_VERSION`" step proposed below.
- **Category / severity / blocks:** A / Medium / should precede the ARC-Easy cache build.
- **Observation:** found by reading the code. No run has hit it yet
  [VERIFIED: code].
- **Location:** `training/dataset.py:154` (capped early return) and `:170-172`
  (`if is_complete and max_samples is None: … return`). The fall-through re-saves
  via `:437-439`, `finished_full_split = (max_samples is None)` →
  `complete=False`. `:203-209`: a missing split silently becomes
  `"validation"`. `:256-260`: a scoring failure stores zeros.
- **Root cause:** state-machine gaps, i.e. conditions that fall through to the
  build path, and silent fallbacks.
- **Theory (causal):**
  1. complete cache + `max_samples > len(cache)` → neither return fires → the
     build loop runs from `raw_index` (end of split), generates nothing → the
     save writes `complete=False`. A complete cache is downgraded, and the next
     uncapped load starts generation (E7).
  2. For datasets without a labelled test split (CommonsenseQA, MedMCQA, PIQA),
     a test cache would silently contain validation items, i.e. leakage into
     model selection.
  3. Zeroed `llm_logprob` z-scores to 0 for that item
     (`models/full_model.py:29-40`). The prior silently disappears for exactly
     those items, and nothing records which.
- **Proposed fix:** (1) treat `complete=True` as final; (2) raise on a missing
  split; (3) add a per-sample `score_failed` flag counted in `report_quality`.
  The cache schema gains one bool per sample; bump `CACHE_VERSION` and keep a
  loader that defaults the flag to False (rule 9).
- **Alternatives considered:** documenting the behaviour. Rejected, since these
  are silent.
- **Why the fix works:** each fall-through becomes an explicit error or a
  recorded flag.
- **Falsifiable prediction:** unit tests with a fake generator: (1) the complete
  flag stays True; (2) `split="test"` on a dataset without test raises; (3) a
  failed scoring batch yields `score_failed=True` and is counted.
- **Verification test:** CPU unit tests with stub generator/encoder (no LLM).
- **Risks:** a cache version bump. **Needs supervisor approval?** No.

#### A7 Validator relevance target has no hypothesis axis
- **Category / severity / blocks:** A / High / blocks any claim about the validator
  or the collapse; **fix needs supervisor approval**.
- **Observation:** collapse entropy is within 0.003 nats of uniform (F3). This is
  consistent with, but not proof of, this cause.
- **Location:** `models/validator.py:140-142`:
  ```python
  if y is not None:
      target = torch.zeros_like(relevance)
      target[torch.arange(B, device=H.device), :, y] = 1.0
  ```
  It enters the energy only through the option-mean (`:163-170`,
  `+ weights[..., 3] * mean(relevance)`, then `energy = -energy`).
- **Root cause:** the target is identical for all K hypotheses, so the loss
  cannot reward telling a sound hypothesis from an unsound one.
- **Theory / mathematics [DERIVED; check 3]:** with logits $\ell_{kn}$ and
  target $\tau_{kn} = \mathbb 1[n=y]$ for every $k$,
  $$\mathcal L_{\text{val}} = \frac{1}{|M|}\sum_{k,n}
  -\big[\tau_{kn}\log\sigma(\ell_{kn}) + (1-\tau_{kn})\log(1-\sigma(\ell_{kn}))\big],
  \qquad \frac{\partial\mathcal L_{\text{val}}}{\partial\ell_{kn}} =
  \frac{\sigma(\ell_{kn}) - \mathbb 1[n=y]}{|M|}.$$
  1. At $n=y$ the gradient is negative for **every** $k$: the support for the
     right answer, the attack on the right answer, the support for a wrong
     answer and the attack on a wrong answer are all pushed to score the true
     option higher. Check 3 shows all 8 gradients negative at $n=y$ for random
     logits, including the 4 unsound hypotheses.
  2. The population minimizer is
     $\ell^\*_{kn} = \mathrm{logit}\,\Pr(n = y \mid x_{kn})$. This is *option
     relevance given the features*: the hypothesis identity matters only
     insofar as it predicts which option is correct.
  3. The channel that reaches the collapse is the option-mean. Its target
     $\frac1N\sum_n\tau_{kn} = 1/N$ is the same for all $k$, so the supervision
     carries **no information distinguishing hypotheses** even at zero loss.
- **Proposed fix (candidate, v47 patch, not applied):** gate by soundness,
  $\tau_{kn} = s_k\,\mathbb 1[n=y]$, with $s_k = 1$ iff (support and
  $m_k = y$) or (attack and $m_k \ne y$), where $m_k$ is the option hypothesis
  $k$ argues about. Shapes unchanged: target (B,K,N), computed from `align`
  (B,K,N) and `polarity` (B,K). No new parameters, so checkpoints are
  compatible. The training objective changes.
- **Alternatives considered:** (i) remove the relevance head. That loses the
  only supervised per-hypothesis signal. (ii) Supervise the option-mean
  directly with $s_k$, which is equivalent in effect but changes the head's
  meaning.
- **Why the fix works:** the gradient at $n=y$ becomes
  $\sigma(\ell_{ky}) - s_k$. It is positive for unsound $k$ (pushed down) and
  negative for sound $k$ (check 3), and the option-mean target $s_k/N$ now
  varies with $k$.
- **Falsifiable prediction:** on a synthetic task where soundness is recoverable,
  mean collapse weight on sound hypotheses exceeds that on unsound ones.
  Full-run collapse KL to uniform rises well above 0.003 nats. If the KL stays
  at ≈0.003, A7 was not the bottleneck (then C6/D1 are).
- **Verification test:** (1) the CPU synthetic test above; (2) a one-variable
  full run on ARC-Easy, with and without the change (rule 4).
- **Risks:** uses labels in a new way during training only (still no leakage at
  eval, `training/evaluate.py:122` passes no `y`). Changes all
  collapse-related metrics, so not comparable with the capstone.
  **Needs supervisor approval?** **Yes**, it changes the training objective.

#### A8 (new) The ablation grid resumes stale checkpoints from earlier sessions
- **Status:** **fixed going forward 2026-10-19** with A3 in `f605c55`. The per-version
  `ckpt/{dataset}/{commit_short}/` directory was not added (decision
  2026-10-19); stale checkpoints fail the provenance check instead.
- **Category / severity / blocks:** A / High / invalidates the v46 grid; blocks the ablation runs.
- **Observation [OBSERVED cell 39]:** `[RESUME A1_baseline_s42] Resuming from
  epoch 8 … [SKIP] A1_baseline_s42 already finished 8 epochs.`; A3
  "Resuming from epoch 7"; P0 "Resuming from epoch 6"; F0 resumed; all from
  `/content/drive/MyDrive/qAIR_V45/ckpt/`. The A3 arm scored 0.15 after one
  v46 epoch on top of earlier weights.
- **Location:** `training/ablations.py:227` (`run_name = f"{name}_s{seed}"`) and
  `:265-272` (`load_or_resume` → `[SKIP]` when `start_epoch >= epochs`);
  `config.py:289` `CKPT_DIR = "./ckpt"`, mapped by the notebook to a Drive
  folder shared across versions.
- **Root cause:** same mechanism as A3 (name-keyed resume, no provenance check),
  across sessions and code versions rather than seeds.
- **Theory (causal):** the grid's reported numbers are
  $f(\theta_{\text{old}}, \text{1–2 epochs of v46})$ for A3/P0/F0, and
  $f(\theta_{\text{old}})$ for A1. $\theta_{\text{old}}$ comes from a session
  whose code commit and cache are unrecorded (E6). Comparisons between arms
  therefore mix code versions and training budgets.
- **Proposed fix:** the A3 fix (provenance in checkpoints, refuse mismatches),
  plus a per-version, per-dataset `CKPT_DIR`
  (`ckpt/{dataset}/{commit_short}/`). No shape changes.
- **Alternatives considered:** deleting old checkpoints by hand before each
  grid. Rejected as error-prone; it is what failed here.
- **Why the fix works:** stale checkpoints either live in another directory or
  fail the provenance check.
- **Falsifiable prediction:** re-running the grid in a directory holding old
  checkpoints raises instead of printing `[SKIP]`/`Resuming`.
- **Verification test:** CPU smoke test. Write a checkpoint with a fake commit
  hash, run one arm, and assert it raises.
- **Risks:** none. **Needs supervisor approval?** No.

#### A9 (new, 2026-10-18) Any cache-load error moved the cache aside and rebuilt it
- **Status:** **fixed 2026-10-18** in `0313a15` (details and
  test evidence in issues.md A9). Found while fixing E3: under torch ≥ 2.6,
  `torch.load` defaults to `weights_only=True`, and the catch-all handler
  renamed a valid cache to `.corrupt` and rebuilt it. Loads now use
  `weights_only=True` with only `TorchVersion` allow-listed, failures raise
  with the path, exception and a JSON sidecar header, and the file is never
  touched.

#### A10 (new, 2026-10-19) An early-stopped run trains again when re-run
- **Status:** **fixed 2026-10-19** with A4 in `4083ecf` (details in issues.md A10).
- **Observation:** found while proposing A4, by reading `training/train.py`:
  the latest checkpoint is written at the stop epoch, then the loop breaks.
- **Root cause:** `stopped_early` was not saved, and resuming reset
  `epochs_no_improve`.
- **Theory:** with patience $P$ and stop epoch $e_s < E-1$, a re-run resumes
  at $e_s + 1$ with count 0, so it trains at least $\min(P, E-1-e_s)$ more
  epochs (more if validation accuracy improves meanwhile). The reported run
  is then longer than the one whose stopping rule fired.
- **Fix:** checkpoint `stopped_early`; a run is finished when
  `stopped_early or start_epoch >= epochs`; finished runs are skipped.
- **Verification test:** re-run an early-stopped CPU smoke run; assert 0
  validation calls and an unchanged checkpoint (also for one grid arm).
- **Risks:** none. **Needs supervisor approval?** No.

#### A11 (new, 2026-10-19) A resumed run is not RNG-continuous
- **Status:** **fixed 2026-10-19** with E10 in `bbea237` ( details in issues.md A11). The falsifiable prediction held on CPU: identical histories and weights after resume.
- **Observation:** CPU probe, losses uninterrupted
  `[1.8829, 1.3686, 1.0157, 0.8861]` vs resumed after epoch 2
  `[1.8829, 1.3686, 0.9016, 0.8123]`.
- **Root cause:** neither the DataLoader generator nor the global torch RNG
  is checkpointed; the loader's generator is re-seeded on every start.
- **Proposed fix:** save and restore both RNG states per epoch.
- **Falsifiable prediction:** identical losses on CPU with `deterministic=True`.
- **Needs supervisor approval?** No.

### Part B — Evaluation problems

#### B1 Evaluation on n=20 validation / 80 train
- **Category / severity / blocks:** B / Critical / every accuracy claim.
- **Observation:** `train_samples 80`, `val_samples 20` (`ckpt/qair_v46_config.pt`)
  [VERIFIED]. Accuracy 0.50, 95% CI [0.30, 0.70]; vs direct LLM b=2, c=4,
  p=0.6875 [OBSERVED cell 41].
- **Location:** notebook cell 12, `TRAIN_SAMPLES = 80`, `VAL_SAMPLES = 20`. The
  code supports full splits (`config.py:56-60`).
- **Root cause:** a smoke-test configuration was reported as a result.
- **Theory [DERIVED; check 6]:** $\widehat{\mathrm{acc}} \sim
  \mathrm{Bin}(n,p)/n$, so the half-width is $1.96\sqrt{p(1-p)/n}$: ±0.219 at
  n=20, ±0.041 at n=570, ±0.020 at n=2376. A paired difference needs
  discordant pairs. With b+c=6, even c=6, b=0 gives $p = 2\cdot2^{-6} =
  0.031$, so at n=20 a significant difference needs an almost one-sided split.
- **Proposed fix:** full ARC-Easy splits (2251 / 570 / 2376) after E3/E7.
- **Alternatives considered:** larger capped subsets. Rejected: there is no
  reason to cap once the cache exists, and caps invite E7.
- **Why the fix works:** the CI shrinks by $\sqrt{2376/20} \approx 10.9\times$.
- **Falsifiable prediction:** the test-split CI half-width is ≤0.021 for every
  arm. If an arm's CI still spans >0.1, the run did not use the full split.
- **Verification test:** the manifest records split sizes, and `stats.py`
  prints n per arm.
- **Risks:** cost (dataset_options.md). **Needs supervisor approval?** No
  (dataset already decided).

#### B2 One seed
- **Category / severity / blocks:** B / High / DoD §4.1.
- **Observation:** all saved runs are seed 42 (`*_s42_*`, `qair_v46_config.pt`)
  [VERIFIED].
- **Location:** `config.py:286` `SEEDS = (42, 43, 44, 45, 46)`, unused by the
  notebook run.
- **Root cause:** run budget.
- **Theory:** observed accuracy is $\mathrm{acc}(s, D) = \mu + u_s + e_D$, with
  seed effect $u_s$ (init, data order, dropout) and finite-sample error $e_D$.
  One seed cannot separate $u_s$ from an arm effect: an arm difference
  $\Delta$ is detectable only if $|\Delta| \gg \sqrt{2}\,\sigma_u$. Without at
  least 2 seeds, $\sigma_u$ is unestimated [DERIVED].
- **Proposed fix:** 3 seeds (42, 43, 44) per mandatory arm.
- **Alternatives considered:** 5 seeds, the code default. Kept optional because
  of cost.
- **Why the fix works:** gives an estimate of $\sigma_u$ and paired per-seed tests.
- **Falsifiable prediction:** the per-arm std across seeds is reported. Any
  claimed arm difference smaller than ~2× that std is withdrawn.
- **Verification test:** `stats.py` output includes the per-seed table.
- **Risks:** 3× compute. **Needs supervisor approval?** No.

#### B3 Validation selects the checkpoint and supplies the number; test never evaluated
- **Category / severity / blocks:** B / High / every reported accuracy.
- **Observation:** best = max of 0.35, 0.35, 0.40, 0.45, 0.40, 0.40, 0.45, 0.50
  (`qair_v46_history.pt`) [VERIFIED]. `No test_records found` [OBSERVED cell 41].
- **Location:** `training/train.py:325-327`, `if metrics["acc"] > best_acc: …
  save_checkpoint(epoch, best_acc, best=True)`; `training/runner.py:87-95` skips
  test without a cache.
- **Root cause:** the same 20 items both choose and score.
- **Theory [DERIVED; check 7]:** max is convex, so by Jensen
  $\mathbb E[\max_e \hat a_e] \ge \max_e \mathbb E[\hat a_e]$, with equality only
  if there is no noise. With true accuracy 0.41, n=20 and 8 epochs, the bias is
  +0.16 (independent epochs) or +0.10 (correlated, item outcome kept with
  probability 0.8 [ASSUMPTION]), and $P(\text{best} \ge 0.50) = 0.93$ (iid).
  The reported 0.50 is what a model with flat accuracy ≈0.41 is expected to
  produce. (In this run the best epoch was also the last, so the bias here
  comes from the noise, not the selection rule.)
- **Proposed fix:** select on validation, evaluate the selected checkpoint once
  on test, and report test. Build the test cache.
- **Alternatives considered:** report the last epoch. That removes selection
  bias but wastes the validation split. It can be reported as a secondary number.
- **Why the fix works:** test items take no part in selection, so the estimate is
  unbiased for the selected model.
- **Falsifiable prediction:** on full splits, mean (best-val − test) across arms
  and seeds is ≥ 0.
- **Verification test:** DoD §4.5; the manifest shows test was evaluated once.
- **Risks:** none. **Needs supervisor approval?** No.

#### B4 No "LLM letter pick" baseline (mandatory)
- **Category / severity / blocks:** B / High / DoD §3.
- **Observation:** the only generator baseline is length-normalized cloze
  likelihood, 0.40 at n=20 [OBSERVED cell 36].
- **Location:** `models/generator.py:357-451` (`score_options`, prompt
  `f"Question: {question}\nAnswer:"` + option); `evaluation/baselines.py` has no
  MCQ mode.
- **Root cause:** missing implementation.
- **Theory:** cloze scoring ranks options by
  $\frac1{|o|}\sum_t\log P(o_t\mid q, o_{<t})$. It never shows the model the
  alternatives, so it cannot compare them, and length normalization shifts
  scores between short and long options. A letter-pick prompt scores
  $\log P(\text{letter}\mid q, o_{1..N})$, i.e. the model sees and compares all
  options. They are different estimands of "what the generator knows"
  [DERIVED]. Which is higher for Qwen2.5-0.5B is unknown [HYPOTHESIS: either].
- **Proposed fix:** a deterministic letter-pick baseline. Chat template, lettered
  options; take the logit of each letter token at the answer position; argmax;
  per-item records.
- **Alternatives considered:** generate free text and parse it. Rejected: it is
  stochastic and parsing is error-prone.
- **Why the fix works:** it is the standard way an instruct model is asked an MCQ
  [ASSUMPTION], so it is the baseline a reviewer will expect.
- **Falsifiable prediction:** two runs give identical predictions (determinism).
  Its accuracy is reported next to the cloze one, without a predicted direction.
- **Verification test:** a determinism test on 20 items, plus a unit test that
  each letter maps to a single token for the tokenizer.
- **Risks:** prompt sensitivity; fix the prompt before seeing test.
  **Needs supervisor approval?** No (decided mandatory).

#### B5 No majority-vote-over-hypotheses baseline (mandatory)
- **Category / severity / blocks:** B / High / DoD §3.
- **Observation:** absent [VERIFIED: `evaluation/baselines.py` modes `llm`,
  `asymmetry`, `probe`, `direct`].
- **Location:** `evaluation/baselines.py`.
- **Root cause:** missing implementation.
- **Theory:** the vote $v_n = \sum_k \mathrm{pol}_k\,\mathbb 1[m_k = n]$ uses
  the same hypotheses as qAIR with zero trained parameters. A pipeline that does
  not beat it adds nothing over its own inputs. The QCR-LLM paper uses majority
  vote at a matched sampling budget as its reference for exactly this reason
  (Flores-Garrigós et al. 2026, abstract). With one support and one attack per
  option, $v_n = 0$ for every option, so a pure count vote is degenerate
  [DERIVED]. The vote must therefore weight by evidence:
  $v_n = \sum_{k\in\mathcal S(n)}\cos(H_k,O_n) - \sum_{k\in\mathcal A(n)}\cos(H_k,O_n)$,
  which is the cache gate's "support − attack margin" statistic
  (`training/dataset.py:711`).
- **Proposed fix:** implement the margin vote with a fixed tie-break (lowest
  index) and per-item records. Document the degenerate count vote.
- **Alternatives considered:** sample several hypotheses per option and count.
  Rejected: it changes the cache (D4) and the generation budget.
- **Why the fix works:** a parameter-free aggregate of exactly qAIR's evidence.
- **Falsifiable prediction:** [HYPOTHESIS] if D1 holds on ARC-Easy, the vote is
  within ±0.04 of chance (val n=570). If it is well above chance, D1 does not
  transfer to ARC-Easy.
- **Verification test:** a unit test on a hand-built cache sample with a known
  answer.
- **Risks:** none. **Needs supervisor approval?** No.

#### B6 Pooling records across seeds makes McNemar and bootstrap anti-conservative
- **Category / severity / blocks:** B / High / DoD §4.2–4.3.
- **Observation:** the code pools [VERIFIED: `evaluation/stats.py:95-100`]. It
  has not been exercised on multi-seed results yet.
- **Location:** `evaluation/stats.py:97-100`:
  `pooled[name] = [v for run in recs for v in run]`; McNemar on pooled at `:127`;
  bootstrap at `:120`.
- **Root cause:** treats (item, seed) cells as independent pairs.
- **Theory [DERIVED]:** let $d_{is}\in\{-1,0,1\}$ be the paired outcome
  difference for item $i$, seed $s$, with variance $\sigma^2$ and inter-seed
  correlation $\rho$ on the same item. Then
  $$\mathrm{Var}\Big(\frac1{nS}\sum_{i,s}d_{is}\Big) = \frac{\sigma^2}{nS}\big(1+(S-1)\rho\big).$$
  The pooled test assumes $\rho = 0$, so it underestimates the variance by the
  factor $1+(S-1)\rho$. With S=5 and strongly item-driven outcomes ($\rho\to1$),
  test statistics are inflated by up to $\sqrt5 = 2.24\times$. Simulated under
  H0 with S=5 [DERIVED; check 11]: pooled McNemar rejects 4.3% / 15.5% / 27.7%
  when seeds keep an item's outcome with probability κ = 0 / 0.5 / 0.8, while a
  single-seed test stays at 3.3–4.8%.
- **Proposed fix:** per-seed McNemar; an item-level paired bootstrap (resample
  items, average over seeds inside each resample); Holm across the
  pre-registered contrasts.
- **Alternatives considered:** a mixed-effects logistic regression. It is more
  powerful but heavier to explain; it is a possible later addition.
- **Why the fix works:** resampling units = items, the actual independent units.
- **Falsifiable prediction:** on simulated identical arms with seed noise, the
  new test rejects at ≈5%, the pooled one at >5%.
- **Verification test:** that simulation, as a CPU unit test.
- **Risks:** wider CIs; this is correct. **Needs supervisor approval?** No.

#### B7 Probe "data ceiling" selects its best epoch on validation
- **Category / severity / blocks:** B / Medium / interpretation of probes.
- **Observation:** probes at n_val=20: 0.25 / 0.25 / 0.40 / 0.30 / 0.35
  [OBSERVED cell 37].
- **Location:** `evaluation/baselines.py:346`,
  `best = max(best, (model(Xva).argmax(1) == yva).float().mean().item())`.
- **Root cause / theory:** same as B3, with 60 epochs instead of 8, so a larger
  upward bias. A "ceiling" estimated this way is optimistic.
- **Proposed fix:** fixed epoch count chosen on a train-split holdout; report
  validation once.
- **Alternatives considered:** keep it and label it optimistic. Rejected: the
  ceiling is used to judge qAIR.
- **Why the fix works:** removes selection on the evaluated items.
- **Falsifiable prediction:** fixed-epoch probe accuracy ≤ best-epoch accuracy,
  on average.
- **Verification test:** run both on the ARC-Easy cache.
- **Risks:** none. **Needs supervisor approval?** No.

#### B8 Optimization budget too small to test claims about learned scalars
- **Category / severity / blocks:** B / High / C2 (phase-learning claims).
- **Observation:** see F6 [OBSERVED cell 27]. Per-scalar drifts are all < 0.01.
- **Location:** 80 items, `BATCH_SIZE = 16` (`config.py:268`), `EPOCHS = 8`
  (cell 12), cosine schedule (`training/train.py:127-130`).
- **Root cause:** 40 optimizer steps.
- **Theory [DERIVED; check 5]:** Adam's update is
  $\Delta\theta_t = -\mathrm{lr}_t\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ with
  $|\hat m_t/\sqrt{\hat v_t}| \lesssim 1$ for consistent gradients (bounded by
  $(1-\beta_1)/\sqrt{1-\beta_2} \approx 3.2$ in the worst case for
  $\beta = (0.9, 0.999)$). The total movement is therefore at most
  $\sum_t \mathrm{lr}_t = 5\sum_{e=0}^{7}\big[10^{-6} + (5\cdot10^{-4}-10^{-6})\tfrac{1+\cos(\pi e/8)}{2}\big]
  = 0.0113$ rad (≤0.036 in the worst case). `polarity_phase` moved a path
  length of 0.0046 and net −0.0020; decoupled weight decay alone gives
  $-\sum_t \mathrm{lr}_t\cdot0.05\cdot\pi = -0.0018$. The data cannot
  distinguish "no gradient signal" from "no step budget".
- **Proposed fix:** full splits: 2251/16 ≈ 141 steps per epoch, i.e. about 18×
  more steps per epoch. Log per-parameter drift and gradient norms every epoch
  (DoD §2.7).
- **Alternatives considered:** raising the LR for scalar parameters. Rejected for
  now: it is a confound (one variable at a time).
- **Why the fix works:** a budget of ~0.2 rad over 141×8 steps would allow
  meaningful movement if the gradient is consistent.
- **Falsifiable prediction:** on full ARC-Easy runs, either `polarity_phase` moves
  ≫ 0.01 rad, or its logged gradient stays ≈0 with alternating sign. The latter
  confirms the stationary-point argument of C2.
- **Verification test:** the drift and grad-norm logs.
- **Risks:** none. **Needs supervisor approval?** No.

#### B9 *Withdrawn*
- Earlier wording claimed the capstone number measured pre-v47 code. The baseline
  is v46, and the run is v46 (`130ff00`, cell 6). B1–B3 cover its real weaknesses.

#### B10 Cache-quality gate computed on capped caches
- **Category / severity / blocks:** B / Medium / D1 conclusions.
- **Observation:** gate on train n=80 (0.175 / 0.1875 / 0.200; LLM 0.3375) and
  val n=20 (0.40 / 0.30 / 0.30; LLM 0.40) [OBSERVED cell 14].
- **Location:** `training/dataset.py:627-775` runs on whatever was loaded.
- **Root cause:** capped caches.
- **Theory:** half-widths at p≈0.25 are $1.96\sqrt{0.1875/n}$: ±0.095 at n=80
  and ±0.19 at n=20 [DERIVED]. The validation "support agreement 0.40" and the
  train "0.175" are compatible with each other and with chance.
- **Proposed fix:** gate on full splits, with CIs printed.
- **Alternatives considered:** none needed.
- **Why the fix works:** ±0.04 at n=570.
- **Falsifiable prediction:** on full ARC-Easy, each gate statistic's CI either
  excludes 0.25 (signal) or includes it (no signal). Either answer is
  informative.
- **Verification test:** gate output on the full cache.
- **Risks:** none. **Needs supervisor approval?** No.

#### B11 (new) Calibration degrades with training and is neither controlled nor properly measured
- **Category / severity / blocks:** B / Low–Medium / DoD §5 (reported diagnostics).
- **Observation:** ECE 0.161 → 0.385 while accuracy stays within noise
  [OBSERVED cell 27]. Mean confidence was not logged, so the direction
  (over- or under-confidence) is not observed. A calibrated model would show
  expected ECE ≈ 0.24 at n=20 (check 10), so this observation is weak; the
  issue is mainly that calibration is not measured in a way that could show a
  real effect.
- **Location:** `training/evaluate.py:212-242` (15-bin ECE over the predicted
  probability). Contributing terms: margin loss, `training/losses.py:90-105`;
  the normalized Born readout, `models/interference.py:228-237`.
- **Root cause:** [HYPOTHESIS] confidence grows without accuracy, through two
  mechanisms (F2): the margin term demands $p_y \ge 0.475$ on training items,
  and the Born normalization is scale-invariant, so near-cancelled residuals
  produce sharp distributions. Separately, ECE at n=20 is a high-variance and
  upward-biased estimator.
- **Theory [DERIVED; checks 9, 10]:**
  $P(n) = |A_n|^2/\sum_m|A_m|^2$ is homogeneous of degree 0 in $w$:
  $P(cw) = P(w)$. Shrinking evidence does not soften the distribution; only
  ratios matter, and residual ratios under cancellation are noisy. For the
  estimator, check 10 gives expected ECE 0.241 / 0.047 / 0.023 for a calibrated
  model at n = 20 / 570 / 2376. At n=20 both the level (0.385) and the trend
  are within what estimator noise can produce, so neither is evidence.
- **Proposed fix (measurement only):** log mean confidence and accuracy per
  epoch; report ECE with a bootstrap CI and a reliability diagram on full
  splits. Optionally report post-hoc temperature scaling fitted on validation,
  as an evaluation-only number.
- **Alternatives considered:** changing the loss (dropping the margin term).
  That is an objective change and needs approval; not proposed now.
- **Why the fix works:** it separates estimator noise from real miscalibration
  and gives the direction.
- **Falsifiable prediction:** on full validation (n=570), if the mechanisms are
  real, ECE rises clearly above the calibrated-noise floor (≈0.05) and
  (confidence − accuracy) > 0 grows with epochs. If ECE stays near 0.05, F2 was
  noise.
- **Verification test:** a unit test that a calibrated synthetic model gives
  bootstrap-CI ECE overlapping its expected finite-sample value.
- **Risks:** none. **Needs supervisor approval?** No (measurement). A loss
  change would need it.

### Part E — Engineering

#### E1 Restore incomplete — *resolved by approved git step*
- **Category / severity / blocks:** E / was Critical / —.
- **Observation:** the working tree had been v40 [VERIFIED earlier: `git status`,
  `git log`]. Now branch `baseline` = v46 tree; `capstone-final` = `9552498`
  (local; full history in `Thorfast191/qAIR-CSE499B`).
- **Location / root cause:** a manual folder copy restored an old clone.
- **Theory:** n/a (state, not mechanism).
- **Proposed fix / why it works:** done; the code equals `9552498`
  [VERIFIED: `git diff capstone-final` before the fresh-history commit].
- **Alternatives considered:** recovering 7530559. It never existed.
- **Falsifiable prediction / verification:** `git diff` of the code against
  `9552498` in the CSE499B clone is empty.
- **Risks:** none. **Needs supervisor approval?** No (done on instruction).

#### E2 Capstone history scattered; v47 edits uncommitted — *resolved for the baseline*
- **Category / severity / blocks:** E / was Critical / —.
- **Observation:** the v47 diff is backed up at `~/qair_backup/v47_uncommitted.patch`
  (sha256 `59a0d595…6010437`), not applied, and still uncommitted in the
  CSE499B clone.
- **Location / root cause:** work spread over 4 clones and remotes.
- **Theory:** n/a.
- **Proposed fix / why it works:** the baseline lives only in `CSE498R-qAIR`;
  the v47 patch is applied only if A7 is approved.
- **Alternatives considered:** committing the v47 diff on a side branch. Not
  requested.
- **Falsifiable prediction / verification:** `shasum -a 256` of the patch matches.
- **Risks:** the patch exists in one place only; copy it to Drive.
  **Needs supervisor approval?** Applying it: yes (A7).

#### E3 Hypothesis generation is unseeded → caches are not reproducible
- **Status:** **fixed 2026-10-18** in `b5c0ae8` (details and test
  evidence in issues.md E3). A second cause was found: mid-batch autosaves.
- **Category / severity / blocks:** E / High / **blocks the ARC-Easy cache build**.
- **Observation:** no seeding anywhere in the build path [VERIFIED: grep for
  `set_seed|manual_seed` in `scripts/build_cache.py`, `training/dataset.py`,
  `models/generator.py` → none]. The v46 run loaded a cache built in an
  unrecorded session (`[CACHE FOUND]`, cell 14).
- **Location:** `models/generator.py:306-314`:
  `outputs = self.model.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE, top_p=0.95, do_sample=True, …)`.
- **Root cause:** sampled decoding draws from an RNG whose state is neither set
  nor recorded.
- **Theory (exact):** the cache is $C = g(D, \text{prompts}, \theta_{\text{LLM}}, \omega)$.
  $\omega$ is the stream of uniform draws consumed by top-p sampling at
  $T = 0.7$. $\omega$ is determined by the process RNG state when generation
  starts (whatever earlier code consumed), and by where a resumed build
  restarts (`training/dataset.py:341-346` restarts at `raw_index` in a new
  process with a new RNG state). Neither is recorded in the cache header
  (`training/dataset.py:516-526`). From the recorded configuration, $C$ is
  therefore a random variable.
  Any reported accuracy $a = f(C, s)$ has
  $\mathrm{Var}(a) = \mathrm{Var}_\omega\,\mathbb E_s[f] + \mathbb E_\omega\,\mathrm{Var}_s[f]$.
  A multi-seed protocol on one cache estimates only the second term; the
  generation term is invisible and cannot be re-measured [DERIVED].
  [ASSUMPTION] GPU sampling kernels may add non-determinism even with a seed.
- **Proposed fix:** seed the build (`set_seed(gen_seed)` before generation, and
  re-seed deterministically per batch from `(gen_seed, raw_idx)` so resumed
  builds reproduce). Record `gen_seed`, the decoding parameters, the model
  revision hash and a prompt hash in the cache header. Hash the finished cache.
  No tensor shape changes. Bump `CACHE_VERSION` for the new header fields; old
  caches load with "provenance unknown".
- **Alternatives considered:** greedy decoding (deterministic). Rejected: it
  changes the hypothesis distribution, which is a design change to D-level
  behaviour.
- **Why the fix works:** with per-batch seeding, $\omega$ is a function of
  `(gen_seed, raw_idx)`, so $C$ is a deterministic function of recorded inputs,
  including across resumes.
- **Falsifiable prediction:** two 20-question builds into separate cache dirs
  with the same seed give byte-identical hypothesis strings on the same device.
  An interrupted-and-resumed build equals an uninterrupted one. Different seeds
  give different strings.
- **Verification test:** a 20-question smoke build on CPU (slow but feasible)
  or a T4; compare text hashes.
- **Risks:** CPU vs GPU outputs may still differ (document per device).
  **Needs supervisor approval?** No.

#### E4 Checkpoints committed to git — *resolved by approved git step (no history rewrite in CSE499B)*
- **Category / severity / blocks:** E / was Medium / —.
- **Observation:** `CSE498R-qAIR` history has no `*.pt`, no checkpoint or cache
  contents and no `.DS_Store`; total 832 KiB [VERIFIED: pre-push audit].
  `.gitignore` ignores `ckpt/`, `cache/`, `.DS_Store`.
- **Root cause:** `!ckpt/*.pt` in the old `.gitignore`. Removed.
- **Theory / alternatives / prediction:** n/a. Verification: `git ls-files '*.pt' | wc -l` = 0.
- **Risks:** none. **Needs supervisor approval?** No.

#### E5 `ablation_results.pt` is overwritten per invocation
- **Category / severity / blocks:** E / Medium / chunked grid runs on Colab.
- **Observation:** found by reading the code [VERIFIED: code].
- **Location:** `training/ablations.py:212` (`results = {}`) and `:318`
  (`torch.save(results, os.path.join(ckpt_dir, "ablation_results.pt"))`).
- **Root cause:** one file holds only the current call's configs.
- **Theory:** running arms in two calls $X$ and $Y$ leaves the file holding $Y$
  only, so `stats.py` silently compares a subset.
- **Proposed fix:** one record per (arm, seed), with its manifest, as
  `results/{arm}_s{seed}.pt`. `stats.py` aggregates the directory.
- **Alternatives considered:** read-modify-write the single file. Rejected:
  concurrent Colab sessions race on it.
- **Why the fix works:** writes never overwrite another run's record.
- **Falsifiable prediction / verification:** run two arms in separate calls; the
  aggregate shows both.
- **Risks:** none. **Needs supervisor approval?** No.

#### E6 Run provenance not recorded
- **Category / severity / blocks:** E / High / makes A8-type errors undetectable.
- **Observation:** the v46 run's commit is known only because the clone cell
  printed it (cell 6). Nothing in the checkpoint or history records it, and
  nothing records the cache or its build [VERIFIED: `ckpt/qair_v46_config.pt`
  keys: model config, seed, sample caps only].
- **Location:** `training/runner.py:189-205` (config only); the ablations save
  none.
- **Root cause:** no manifest.
- **Theory:** a result is reproducible only if
  $(\text{code}, \text{cache}, \text{config}, \text{seed}, \text{env}, \text{device})$
  are all recorded. Missing any one makes two runs with the same "config"
  incomparable (A8 is the realized case).
- **Proposed fix:** a manifest per run: git commit + dirty flag, cache hash,
  dataset revision, `pip freeze` hash, device names (torch and circuit), seeds,
  split sizes, wall time. Store it inside the checkpoint and as JSON. Log a line
  in `docs/research_log.md`.
- **Alternatives considered:** an experiment tracker (W&B). It is an extra
  dependency and account; possible later.
- **Why the fix works:** every number traces to recorded inputs.
- **Falsifiable prediction / verification:** a unit test that the manifest has
  all keys and that the commit equals `git rev-parse HEAD`.
- **Risks:** none. **Needs supervisor approval?** No.

#### E7 A capped cache can silently restart hours of generation
- **Status:** **fixed 2026-10-18** in `6aa20a5` (evidence in
  issues.md E7). A missing cache was also built silently; now it raises too.
- **Category / severity / blocks:** E / High / **blocks the ARC-Easy cache build**.
- **Observation:** documented in the code itself [VERIFIED: `training/dataset.py:157-163`
  comment; `evaluation/baselines.py:304-309` docstring].
- **Location:** `training/dataset.py:174-197`: any load of an incomplete cache
  with `max_samples=None` falls into the build path and constructs
  `HypothesisGenerator()` (`:197`).
- **Root cause:** load and build share one constructor with no "read-only" mode.
- **Theory:** let the stored state be `(complete=False, n_stored)`. Any caller
  with `max_samples ∈ {None} ∪ (n_stored, ∞)` triggers generation. That
  includes training (`training/runner.py:52-59`), baselines, input ablation and
  notebooks. Combined with A6(1), even a *complete* cache can be downgraded and
  then regenerated.
- **Proposed fix:** an `allow_build=False` default. Only `scripts/build_cache.py`
  passes True. Otherwise raise with the stored cap and `complete` flag. No
  schema change.
- **Alternatives considered:** a warning plus a confirmation prompt. Rejected:
  non-interactive runs.
- **Why the fix works:** the build path becomes unreachable from consumers.
- **Falsifiable prediction / verification:** a unit test with a capped stub cache:
  `QAIRDataset(max_samples=None)` raises, and the generator constructor is never
  called (monkeypatched to fail).
- **Risks:** callers relying on implicit builds break loudly; intended.
  **Needs supervisor approval?** No.

#### E8 No v45+ timing data; no tests
- **Category / severity / blocks:** E / Medium / cost planning; regression safety.
- **Observation:** v46 tqdm bars were saved at 0% (cell 27). The only rate on
  record is v44, `1.24s/it` [VERIFIED: `qAIR_v44.ipynb` cell 21 in the CSE499B
  clone]. No `tests/` directory.
- **Location:** repo layout; `models/full_model.py:336-358` runs the reasoner and
  the circuit twice when validator feedback is on.
- **Root cause / theory:** without measured per-step time, a full run (≈141
  steps/epoch on ARC-Easy) cannot be budgeted against the 1-hour approval rule.
- **Proposed fix:** a timed 20-question smoke build and 1-epoch train on T4,
  recorded in the manifest. Add `tests/` with A0, A1/A2 (grad coverage), A3,
  A4, A5, A6, E3, E7.
- **Alternatives considered:** extrapolating from v44. Done in
  dataset_options.md, explicitly as an assumption.
- **Why the fix works:** measured numbers replace assumptions; tests make
  regressions fail loudly.
- **Falsifiable prediction / verification:** `pytest` passes on CPU; the smoke
  manifest contains s/step and s/question.
- **Risks:** none. **Needs supervisor approval?** No.

#### E9 Stale docs and names
- **Category / severity / blocks:** E / Low / readability.
- **Observation / location:** `evaluation/stats.py:4-7` (869 = merged-ARC
  validation); `training/runner.py:125,136` ("v45.1");
  `scripts/build_cache.py:149` (`qair_v45_best.pt`);
  `docs/history/CLAUDE_v46.md` ("Current version: v45") [VERIFIED: code].
- **Root cause / theory:** text not updated with versions.
- **Proposed fix:** correct when the files are touched for other fixes.
- **Alternatives / prediction / verification:** n/a; grep for `v45` after the edits.
- **Risks:** none. **Needs supervisor approval?** No.

#### E10 (new) Training is non-deterministic by default
- **Status:** **fixed on CPU 2026-10-19** in `bbea237` (
  evidence in issues.md E10). **Change from the proposal below:** strict by
  default (`warn_only=False`); `--deterministic-warn-only` opts into
  warnings and is recorded and compared. The T4 prediction is still
  untested: the Step 6 notebook runs it and checks `lightning.gpu`.
- **Category / severity / blocks:** E / Medium / DoD §1.5 (same seed ⇒ same predictions).
- **Observation:** `[SEED] Set to 42 (deterministic=False)` (cells 10, 39); the
  circuit ran on `lightning.gpu` (cell 22) [OBSERVED].
- **Location:** `training/seed.py:8`, `def set_seed(seed=42, deterministic=False)`.
  No caller passes `deterministic=True` [VERIFIED: grep `set_seed(` in v46].
- **Root cause:** cuDNN autotuning and non-deterministic CUDA kernels stay
  enabled. Floating-point reduction order varies between runs.
- **Theory:** with non-associative floating-point sums, identical seeds give
  bitwise-different gradients after the first non-deterministic reduction.
  Training is chaotic in the weights, so trajectories diverge, and "same seed"
  does not mean "same run". The seed-to-seed std then includes this kernel noise
  [DERIVED, standard numerical argument].
- **Proposed fix:** `deterministic=True` for reported runs (the code already
  supports it: `training/seed.py:29-33`). Record whether it was on in the
  manifest. CPU smoke tests are deterministic anyway.
- **Alternatives considered:** accept the non-determinism and rely on multiple
  seeds. Rejected for DoD §1.5, which asks for exact replay.
- **Why the fix works:** forces deterministic algorithms (with `warn_only=True`,
  so ops lacking a deterministic version still warn; check the log).
- **Falsifiable prediction:** two GPU runs with the same seed give identical
  validation predictions. If they differ, some op is non-deterministic, and the
  warning log names it.
- **Verification test:** a 1-epoch smoke run twice on T4; diff the prediction files.
- **Risks:** slower; `lightning.gpu` determinism is unknown [ASSUMPTION], so test it.
  **Needs supervisor approval?** No.

#### E11 (new) Environment not pinned; resolver conflicts in the v46 run
- **Status:** **fixed 2026-10-19 except the torch pin** in `4c4a766`
  (evidence in issues.md E11). Startup pin check for
  runtime packages (raises; recorded override), environment in provenance
  (torch, CUDA, pennylane, lightning compared on resume; the rest
  recorded), package list per run. The torch pin comes from the Step 6 T4
  run; until then torch is recorded only.
- **Category / severity / blocks:** E / Medium / reproducibility on a fresh Colab.
- **Observation [OBSERVED cells 8, 10]:**
  `diffusers 0.39.0 requires huggingface-hub<2.0,>=0.34.0, but you have huggingface-hub 0.33.4`;
  `gradio 6.20.0 requires huggingface-hub<2.0,>=1.2.0 …`;
  `PennyLane is not yet compatible with JAX versions > 0.6.2. You have version 0.7.2`.
- **Location:** `requirements.txt:1` (`torch` unpinned); pins installed on top of
  a moving Colab base image.
- **Root cause:** partial pinning on a mutable base environment.
- **Theory:** the effective environment is (Colab image at run date) ⊕ (pins). The
  image changes over time, so the same notebook resolves different versions on
  different dates. The conflicts here involve packages qAIR does not import
  (diffusers, gradio, JAX). That is harmless today [HYPOTHESIS], but it shows
  the environment is not controlled.
- **Proposed fix:** pin `torch` and record `pip freeze` in the manifest (E6). For
  reported runs, install into a clean venv on Colab, or uninstall the conflicting
  unused packages first.
- **Alternatives considered:** Docker. Not available on Colab.
- **Why the fix works:** the environment becomes a recorded, reproducible input.
- **Falsifiable prediction / verification:** two installs on different days give
  identical `pip freeze` hashes, and the run log contains no resolver errors.
- **Risks:** pinned torch may lag the Colab CUDA version; choose a compatible pin.
  **Needs supervisor approval?** No.

### Part D — Data / input problems

#### D1 The 0.5B generator's hypotheses carry no detectable signal
- **Category / severity / blocks:** D / Critical for the research question; does
  not block "working".
- **Observation [OBSERVED cell 14]:** train cache, n=80: support agreement
  0.1750, attack disagreement 0.1875, support−attack margin 0.2000 (chance 0.25);
  LLM option likelihood 0.3375; `[SIGNAL WARNING]`.
- **Location:** generation `models/generator.py:136-161` (prompts), `:306-314`
  (sampling); measurement `training/dataset.py:689-736`.
- **Root cause:** [HYPOTHESIS] a 0.5B instruct model, asked separately per option,
  writes fluent but non-discriminative support and attack (see D2, D3, D5).
- **Theory [DERIVED; check 12]:**
  1. **Significance.** One-sided $P(X\le k\mid 80, 0.25)$ = 0.074, 0.121, 0.184
     for the three statistics. None is significantly *below* chance; all are
     consistent with **no signal**. The LLM likelihood (27/80) has
     $P(X\ge27) = 0.050$, borderline.
  2. **Why no architecture can recover it** (data-processing inequality). If
     $I(y; H\mid Q, O) = 0$, then for any learned $f$,
     $\mathrm{err}(f(H,Q,O)) \ge \mathrm{err}^\*(Q,O)$: the Bayes error with
     hypotheses equals the Bayes error without them. Architecture work on qAIR
     cannot fix a cache with no conditional information.
- **Proposed fix:** first *measure* on the full ARC-Easy cache (B10), with CIs. If
  the signal is still absent, generation changes (D3/D5, or a larger generator)
  go to the supervisor.
- **Alternatives considered:** tuning prompts now on 80 items. Rejected:
  under-powered, and it risks selection on noise.
- **Why the fix works:** it decides with power whether D1 is real on the
  baseline dataset.
- **Falsifiable prediction:** [HYPOTHESIS] ARC-Easy is easier, so the margin
  statistic exceeds chance with a 95% CI excluding 0.25 at n=570. If not, D1
  holds on ARC-Easy too, and any qAIR gain must come from the LLM prior (F7).
- **Verification test:** the full-split cache gate (B10) and the majority-vote
  baseline (B5).
- **Risks:** none. **Needs supervisor approval?** Measuring: no. Changing the
  generator or prompts: yes.

#### D2 Hypotheses are less diverse than the options; support ≈ attack for the same option
- **Category / severity / blocks:** D / High / limits D1.
- **Observation [OBSERVED cell 14]:** cos(H_sup) 0.6644, cos(H_att) 0.6116 vs
  cos(O) 0.4949; cos(sup_n, att_n) 0.7412 (train). Val: 0.6387 / 0.5887 vs 0.4106;
  0.7286.
- **Location:** `models/encoder.py:44-64` (mean-pooled MiniLM, L2-normalized);
  prompts `models/generator.py:140-161`.
- **Root cause:** [HYPOTHESIS] hypotheses restate the question's content (shared
  tokens), and mean pooling makes shared content dominate the embedding.
- **Theory [DERIVED, a simple model]:** write $h_k = \alpha q + \beta o_{m_k} +
  \epsilon_k$ with unit-ish, roughly orthogonal components. Then
  $\cos(h_k, h_j) \approx (\alpha^2 + \beta^2\cos(o_{m_k},o_{m_j}))/(\alpha^2+\beta^2+\sigma^2)$.
  This *exceeds* $\cos(o, o')$ whenever the shared question weight $\alpha^2$ is
  large enough, which is what we observe. For the same option,
  $\cos(\text{sup}_n, \text{att}_n)$ is high because $q$ and $o_n$ are shared,
  and the polarity lives in a few tokens ("fails", "wrong") that mean pooling
  dilutes.
- **Proposed fix (measurement first):** report $\cos(H, Q)$ and $\cos(H, O)$ by
  polarity on the full cache. Changing the encoder or pooling is C4 territory.
- **Alternatives considered:** subtract the question embedding from $H$.
  Rejected for now: it changes the model input, a pipeline change.
- **Why the fix works:** it tests the model above before acting on it.
- **Falsifiable prediction:** [HYPOTHESIS] cos(H, Q) is high (>0.6) for both
  polarities. If it is low, the shared-content explanation is wrong.
- **Verification test:** a cache-level script on the full ARC-Easy cache (CPU,
  no training).
- **Risks:** none. **Needs supervisor approval?** Measuring: no.

#### D3 Each generation prompt sees one option only
- **Category / severity / blocks:** D / Medium / limits the support/attack asymmetry.
- **Observation:** prompts contain the question and one proposed answer, and say
  "do not mention other options" [VERIFIED: `models/generator.py:140-161`].
- **Location:** `models/generator.py:136-161`.
- **Root cause:** the per-option design chosen in v44 to guarantee alignment
  (docstring `:45-50`).
- **Theory:** the attack on option $m$ is sampled from $p(\cdot\mid q, o_m)$.
  The only information distinguishing a correct from a wrong $o_m$ is the
  model's internal belief about $o_m$ in isolation, which is the same
  information the cloze score $\log p(o_m\mid q)$ measures. Without the
  alternatives, the attack cannot be *contrastive* ("B is wrong because C…"), so
  the attack channel is bounded by the generator's per-option judgement
  (0.3375 on train) [DERIVED as an information argument; magnitude HYPOTHESIS].
- **Proposed fix (for discussion):** a contrastive attack prompt showing all
  options, with alignment kept by asking about one target option at a time.
- **Alternatives considered:** one completion per question listing all options.
  Rejected: the pre-v44 alignment drift returns.
- **Why the fix works:** gives the generator the information needed for
  contrastive refutation.
- **Falsifiable prediction:** the attack-disagreement statistic increases on the
  same items. If unchanged, the 0.5B model's knowledge, not the prompt, is the limit.
- **Verification test:** a paired comparison of two caches on the same items.
- **Risks:** a new cache version, not comparable with the capstone.
  **Needs supervisor approval?** Yes (changes the model's input evidence).

#### D4 Encoder truncates at 128 tokens; one sampled hypothesis per (option, mode)
- **Category / severity / blocks:** D / Medium (High for MedQA later) / —.
- **Observation:** `max_length=128` [VERIFIED: `models/encoder.py:47`]; one sample
  at T=0.7 per (option, mode) [VERIFIED: `models/generator.py:292-314`]; first
  line only [VERIFIED: `models/generator.py:198`].
- **Root cause:** cost choices.
- **Theory:** (i) truncation drops all text after 128 word-pieces. That is
  irrelevant for short ARC hypotheses [ASSUMPTION; length distribution not
  measured], but MedQA vignettes are long. (ii) With one sample, the hypothesis
  is one draw $h\sim p(\cdot\mid q,o,\text{mode})$. Its embedding has sampling
  variance $\sigma^2_{\text{gen}}$, which averaging $M$ samples would reduce by
  $1/M$. With $M=1$, the evidence for each (option, mode) is as noisy as
  generation allows [DERIVED].
- **Proposed fix:** measure the token-length distribution (cheap). Consider
  $M>1$ samples per (option, mode) later (cost ×M).
- **Alternatives considered:** greedy decoding (removes variance but also
  diversity).
- **Why the fix works / prediction:** if the length 95th percentile is < 128 on
  ARC-Easy, (i) is moot for the baseline.
- **Verification test:** a tokenizer length histogram over the full cache texts.
- **Risks:** cost. **Needs supervisor approval?** Changing $M$ or the encoder: yes.

#### D5 (new) Attack hypotheses do not reliably attack
- **Category / severity / blocks:** D / High / corrupts the polarity channel.
- **Observation [OBSERVED cells 16, 43]:**
  - Correct option "Carbon dioxide", **attack**: "Plants absorb carbon dioxide
    (CO2) through their stomata … This absorption process is crucial…". This
    supports the option.
  - Seatbelts, attack: "Seat belts reduce injuries by increasing speeds safely,
    not slowing them down."
  - Supports that are factually wrong also appear ("increasing a passenger's
    speed through the inertia effect").
  - For the correct option "dry palms", the attack argues it is wrong. That is
    the intended behaviour, an attack on the correct option, and not a defect.
  - The *rate* of non-compliance has not been measured.
- **Location:** `models/generator.py:150-161` (attack prompt); polarity is
  assigned by *mode*, not content (`models/generator.py:347`,
  `r["polarity"].append(POLARITY[mode])`).
- **Root cause:** a 0.5B model does not reliably follow the "state the reason
  this answer FAILS" instruction, and the label is attached regardless.
- **Theory [DERIVED]:** let an attack's content actually support its option with
  probability $\eta$, while its label stays −1. The expected signed evidence of
  the attack channel for option $n$ is
  $\mathbb E[\text{signed}] = (1-\eta)\,a_n - \eta\,a_n = (1-2\eta)\,a_n$.
  At $\eta = 0.5$ the attack channel carries **zero** expected signal. For
  $\eta > 0.5$ it is anti-informative. In the readout these attacks enter with
  phase π and subtract genuine support.
- **Proposed fix:** measure first. Audit compliance on a random sample of 100
  attacks with a fixed rubric (manual, or an NLI model), and report $\eta$ with a
  CI. Only then consider filtering by an entailment check or regenerating.
- **Alternatives considered:** dropping attacks (the E0 arm already tests
  support-only).
- **Why the fix works:** $\eta$ determines whether the attack channel can help at
  all, by the formula above.
- **Falsifiable prediction:** [HYPOTHESIS] $\hat\eta$ is non-negligible (>0.2).
  If $\hat\eta < 0.05$, D5 is not a material cause of D1.
- **Verification test:** the audit, CPU only, on cached texts.
- **Risks:** none for measurement. **Needs supervisor approval?** Filtering or
  regeneration: yes.

### Part C — Design limitations (all **needs supervisor discussion**)

#### C1 The Born readout is sign-symmetric; with a reference term it becomes additive scoring
- **Category / severity / blocks:** C / High / any "interference helps" claim.
- **Observation:** at the best checkpoint, coherent_acc 0.25 < classical_acc 0.30
  [OBSERVED cells 27, 32]. The coherent ranking was no better than the decohered
  mixture of the same amplitudes.
- **Location:** `models/interference.py:217-220`:
  ```python
  real = (w * torch.cos(phase)).sum(dim=1)
  imag = (w * torch.sin(phase)).sum(dim=1)
  coherent = real ** 2 + imag ** 2
  ```
  `models/full_model.py:390`: `phase_h = phase_h + 0.5 * (1.0 - polarity) * self.polarity_phase`.
- **Root cause:** a squared modulus of a sum in which attacks enter with phase
  ≈π. The sign of the support-minus-attack difference is discarded.
- **Theory / mathematics [DERIVED; check 2]:**
  $$|A_n|^2 = |S_n|^2 + |R_n|^2 + 2\,\mathrm{Re}\big(\bar S_n R_n e^{i\rho}\big).$$
  With phase-aligned channels (all $\varphi_k+\theta_{kn}$ equal, so $S_n, R_n$
  are real and ≥0) and $\rho = \pi$: $|A_n|^2 = (S_n - R_n)^2$. Then
  $P(n) \propto (S_n-R_n)^2$ depends on $|S_n - R_n|$ only.
  **Two-option counterexample:** option 1 has support 0.6 and attack 0.1;
  option 2 has support 0.1 and attack 0.6. Then $(0.6-0.1)^2 = (0.1-0.6)^2 =
  0.25$, so $P = (0.5, 0.5)$. A strongly refuted option ties a strongly
  supported one. Check 2 reproduces this through the exact readout code path
  ($P_0 = P_1$ to $10^{-9}$).
  **With a reference amplitude** $c > \max_n(R_n - S_n)$:
  $|c + S_n - R_n|^2$ is strictly increasing in $S_n - R_n$. The ranking then
  equals that of the additive score $S_n - R_n$; the squared link only reshapes
  probabilities (check 2: $P_{\text{ref}} = (0.9, 0.1)$). So the readout is
  either sign-symmetric (no reference) or ranks like additive scoring
  (reference). Neither uses interference to rank.
  Consequence at near-uniform amplitudes (C6): $S_n \approx R_n$, and the
  ranking is driven by the small residual $|S_n - R_n|$.
- **Proposed fix (options for discussion):**
  1. Keep the readout and report the limitation; rely on the P0/P1 controls.
  2. A signed readout,
     $\mathrm{score}_n = \mathrm{Re}(A_n e^{-i\psi})$ with a learned reference
     phase $\psi$. It is monotone in the projected evidence, but no longer a Born
     probability.
  3. Add an explicit reference amplitude, e.g. from the LLM prior:
     $|c\,e^{i\psi} + A_n|^2$. This makes the ranking additive, as shown above.

  Shapes are unchanged in all options ((B,N) scores). Options 2 and 3 add 1–2
  scalar parameters (checkpoint key added; old checkpoints load with the
  default).
- **Alternatives considered:** fixing ρ to π (removing the parameter). That does
  not address the symmetry.
- **Why the fix works:** option 2 makes the score's sign follow the evidence's
  sign by construction; option 3 is proven monotone above.
- **Falsifiable prediction:** on the constructed example, options 2 and 3 rank
  the supported option first. On full data, if C1 matters, coherent_acc rises
  relative to classical_acc. If coherent ≈ classical after the change, the
  symmetry was not the binding constraint (D1 is).
- **Verification test:** a CPU unit test with the counterexample, plus the
  coherent-vs-classical paired comparison on ARC-Easy.
- **Risks:** changes the central claimed mechanism; capstone not comparable.
  **Needs supervisor approval?** **Yes** (architecture/readout).

#### C2 Phase channel: `destructive_fraction` saturated by initialization; π is a stationary point
- **Category / severity / blocks:** C / High / any phase or interference claim.
- **Observation:** `destructive_fraction` 1.0000 before training and at every
  epoch; `polarity_phase` 3.1421 → 3.1396; the sanity check
  `destructive interference occurs` passed with 1.0000 [OBSERVED cells 24, 27].
- **Location:** `models/full_model.py:226`,
  `self.polarity_phase = nn.Parameter(torch.tensor(math.pi))`;
  `models/interference.py:287`,
  `destructive_fraction = masked_mean((interference < 0).to(classical.dtype))`.
- **Root cause:** the metric is a step function of ρ, and its init sits past the
  step. Separately, the readout's gradient in ρ vanishes at ρ = π for aligned
  channels.
- **Theory / mathematics [DERIVED; checks 4, 5]:**
  1. **Saturation.** For aligned channels, interference
     $I_n = |A_n|^2 - C_n = [S_n^2 - \textstyle\sum_{\mathcal S} w^2] + [R_n^2 -
     \sum_{\mathcal A} w^2] + 2S_nR_n\cos\rho$. With uniform $w = c$
     ($S_n = R_n = Nc$):
     $$I_n = 2N(N-1)c^2 + 2N^2c^2\cos\rho < 0 \iff \cos\rho < -\tfrac{N-1}{N}
     \iff \rho > \arccos(-\tfrac{N-1}{N}) = 2.419\ \text{rad}\ (N=4).$$
     This condition is independent of $c$ and of the data. At $\rho = \pi$,
     $I_n = -2Nc^2 < 0$ for every option of every sample, so the fraction is
     exactly 1 at initialization. Check 4 reproduces the step exactly for
     uniform amplitudes. At the init scale (energy spread from the standardized
     selector, circuit-phase spread 0.11 as measured in cell 24) it gives
     0.000 / 0.479 / 0.996 / 1.000 at ρ = 1.6 / 2.4 / 2.8 / π, against
     `config.py:223-226`'s measurements on the real model of
     0.000 / 0.494 / 0.994 / 1.000. The saturation weakens only when phases are
     spread widely (fraction at π: 1.00, 0.90, 0.68, 0.62 for phase sd 0.1, 0.5,
     1.0, 2.0).
  2. **Stationary point.**
     $\partial|A_n|^2/\partial\rho = -2\,\mathrm{Im}(\bar S_n R_n e^{i\rho})$.
     At $\rho=\pi$ this is $2|S_n||R_n|\sin(\arg R_n - \arg S_n)$. That is 0 for
     aligned channels (check 5: |grad| = 0 at π vs 1.75 at π/2) and small for
     the measured phase spread. By the chain rule
     $\partial\mathcal L/\partial\rho = \sum_n \frac{\partial\mathcal L}{\partial|A_n|^2}\frac{\partial|A_n|^2}{\partial\rho}$
     is ≈0 as well, and its sign depends on per-option misalignments of random
     sign. Together with the Adam budget (B8), this explains ρ's sub-0.006
     movement without any claim about what the data "wants".
- **Proposed fix (options for discussion):**
  1. **Measurement-only (no approval):** replace the sanity check `> 0` with
     "differs from an untrained model's value"; report coherent_acc −
     classical_acc with a paired CI as the decisive phase metric (it exists in
     `training/evaluate.py`).
  2. **Initialization change (approval):** init ρ = π/2. There
     $\partial|A_n|^2/\partial\rho = -2S_nR_n$ is maximal in magnitude, and the
     fraction starts at 0, so any learned cancellation becomes visible as ρ
     crossing 2.42.
- **Alternatives considered:** a random ρ init per seed. That adds variance
  without a clear test.
- **Why the fix works:** (1) removes a metric that is true by construction;
  (2) starts ρ where the gradient is informative and the metric is not
  pre-saturated.
- **Falsifiable prediction:** with init π/2 on full ARC-Easy, if the data favours
  cancellation, ρ rises past 2.42 and destructive_fraction climbs from ≈0. If ρ
  stays near π/2 or falls, cancellation is not learned. This is a clean,
  pre-registrable test.
- **Verification test:** a CPU unit test of the threshold (already in check 4);
  the full-run drift log.
- **Risks:** changes the init; not comparable with the capstone.
  **Needs supervisor approval?** Option 1 no; option 2 **yes**.

#### C3 K = 2N hypotheses enter unpruned; no abstention
- **Category / severity / blocks:** C / Medium / —.
- **Observation:** no pruning [VERIFIED: `models/full_model.py`]; always argmax
  [VERIFIED: `training/evaluate.py:124-128`]; collapse near-uniform (F3).
- **Root cause:** design.
- **Theory [DERIVED]:** if $m$ of the $K$ hypotheses carry signal $\mu$ and all
  carry independent noise $\sigma$, a uniform sum has
  $\mathrm{SNR} = m\mu/(\sigma\sqrt K)$. Adding uninformative hypotheses lowers
  SNR as $1/\sqrt K$, so with a non-selective collapse (C6), unpruned
  hypotheses dilute the informative ones. Abstention trades coverage for
  accuracy and requires calibrated confidence (B11).
- **Proposed fix (options):** a top-k or threshold prune on collapse weight
  (once C6 is resolved); selective prediction reported as an accuracy–coverage
  curve (evaluation-only).
- **Alternatives considered:** pruning by the validator alone. It depends on A7.
- **Why the fix works / prediction:** if the collapse becomes selective, pruning
  to the top-m raises SNR. Prediction: accuracy at 80% coverage > full-coverage
  accuracy if confidence is informative.
- **Verification test:** an accuracy–coverage curve on full ARC-Easy.
- **Risks:** pruning before selectivity exists removes signal at random.
  **Needs supervisor approval?** Pruning yes; the coverage curve (measurement) no.

#### C4 All LLM knowledge reaches the model through frozen 384-d MiniLM embeddings
- **Category / severity / blocks:** C / High / ceiling on what H can contribute.
- **Observation:** F0 (with the prior) best validation 0.35 vs cached LLM alone
  0.40 on the same 20 items; `llm_logprob := 0` → 0.25 [OBSERVED cells 39, 36, 34].
  None is significant at n=20.
- **Location:** `config.py:88-96` (the comment states the bottleneck);
  `models/full_model.py:416-424` (the prior is the only bypass).
- **Root cause:** a frozen sentence encoder sits between generator and model.
- **Theory [DERIVED]:** for the frozen map $e$,
  $I(y; e(H)) \le I(y; H)$ (data-processing inequality). Whatever discriminative
  content the hypothesis text has, mean-pooled MiniLM can only lose some of it.
  The generator's own judgement enters only through the scalar `llm_logprob`.
- **Proposed fix (options):** keep and report (recommended for the baseline);
  later, use the generator's hidden states, or fine-tune the encoder (cost; a
  pipeline change).
- **Alternatives considered:** a larger frozen encoder. Same bound, slightly looser.
- **Why the fix works:** removes or loosens the bottleneck.
- **Falsifiable prediction:** a probe on generator hidden states beats a probe on
  MiniLM embeddings on the same items.
- **Verification test:** probes (B7 fixed) on both representations.
- **Risks:** large pipeline change. **Needs supervisor approval?** **Yes.**

#### C5 Redundant and doubled computation
- **Category / severity / blocks:** C / Low / cost, interpretability of fusion weights.
- **Observation:** the notebook notes that validator feedback runs the reasoner
  and circuit twice ("turning this off roughly halves per-epoch time", cell 12)
  [OBSERVED].
- **Location:** `models/answer_selector.py:185-187`, hamiltonian and cosine
  energies; `models/full_model.py:336-358`, the second pass.
- **Root cause:** design.
- **Theory [DERIVED]:** with $H$ normalized, $-\cos(H, O') = -H\cdot O'/\|O'\|$
  while the hamiltonian energy is $-H\cdot O'$. After per-sample
  standardization over (K,N), they differ only by the per-option factor
  $1/\|O'_n\|$. They are nearly collinear, so the softmax fusion weights between
  them are weakly identified.
- **Proposed fix (options):** drop one energy; make feedback an ablation arm
  rather than the default.
- **Alternatives considered:** keep both (status quo).
- **Why it works / prediction:** removing the cosine term changes validation
  accuracy by less than seed noise.
- **Verification test:** a one-variable ablation on full data.
- **Risks:** parameter count changes. **Needs supervisor approval?** **Yes.**

#### C6 (new) The collapse over hypotheses is near-uniform
- **Category / severity / blocks:** C / High / any "selects hypotheses" claim.
- **Observation:** KL to uniform 0.0090 → 0.0031 nats; peak 0.149 → 0.141
  (uniform 0.125) [OBSERVED; DERIVED check 1].
- **Location:** fused energy `models/energy_fusion.py:50-70`; temperature
  `models/collapse.py:83`; amplitudes `models/interference.py:143-162`.
- **Root cause:** [HYPOTHESIS, multi-causal] the per-hypothesis energy differences
  are tiny, std ≈ 0.08. Contributors: (i) fixed-at-init blend weights halve and
  quarter each k-specific signal; (ii) the relevance target is k-invariant (A7);
  (iii) hypotheses carry no signal (D1); (iv) the budget is too small to
  sharpen anything (B8); (v) no loss term rewards selectivity beyond the weak
  diversity reward ($0.02\cdot\tanh$).
- **Theory [DERIVED; check 1]:** $p = \mathrm{softmax}(-E/T)$ and
  $\mathrm{KL}(p\|U) = \log K - H(p) \approx \mathrm{Var}_k(E)/(2T^2)$. With
  logged diversity $\mathrm{Var}_k(E)$ = 0.0182 and 0.0067, this gives
  $T = 1.003$ and $1.033$. That is the temperature controller's initial value
  (0.993), so the entropy is fully explained by small energy differences at an
  unmoved temperature. The fused energy is
  $E_k = (1-v)[\alpha\,\tilde e_k + (1-\alpha)q_k] + vV_k$ with
  $\alpha = \sigma(\cdot) \approx 0.499$ and $v \approx 0.500$, as printed every
  epoch. So $E_k \approx 0.25\,\tilde e_k + 0.25\,q_k + 0.5\,V_k$, and the
  variance of each k-specific component is scaled by $1/16$, $1/16$, $1/4$. A
  selective collapse (0.5 peak) needs a logit gap of $\ln 7 = 1.95$, about 24×
  the observed energy std.
- **Proposed fix:** none yet. Re-measure after A7 (if approved) and on full data
  (B8). If it is still near-uniform, discuss (a) a temperature floor or scale,
  or (b) an explicit selectivity term. Both are objective or architecture changes.
- **Alternatives considered:** an entropy penalty now. Rejected: it would force
  selectivity even without signal (D1), the v44-style mistake of rewarding a
  metric.
- **Why the fix works:** it addresses causes in order of how cheap they are to
  test.
- **Falsifiable prediction:** if A7 is the main cause, KL rises well above 0.003
  after the A7 change on full data. If it does not, D1/C7 dominate.
- **Verification test:** KL and diversity per epoch on full ARC-Easy, A7 on/off
  (one variable).
- **Risks:** none for measurement. **Needs supervisor approval?** Any change: **yes.**

#### C7 (new) Hypotheses' option-energy profiles converge (energy-row cosine rises)
- **Category / severity / blocks:** C / High / multi-hypothesis claim.
- **Observation:** energy-row pairwise cosine 0.769 → 0.838 (main run), 0.9706 (A3
  arm); hypothesis-vector cosine stable 0.87–0.88 [OBSERVED cells 27, 39].
- **Location:** `models/answer_selector.py:142-199` (energy from features
  `[H, O, H−O, H·O, Q, Q·O, Q−O, align, pol, align·pol]`, of which the Q/O
  terms are identical across k); metric `training/evaluate.py:331-351`.
- **Root cause:** [DERIVED in a limit, HYPOTHESIS in general] at near-uniform
  amplitudes the loss sends the *same* gradient to every hypothesis's energy row
  within a channel. The updates are common-mode, so the rows move together.
- **Theory [DERIVED; check 13]:** with aligned phases at ρ = π,
  $|A_n|^2 = (S_n - R_n)^2$ and $w_{kn} = r_k t_{kn}$,
  $t_{kn} \propto e^{-E_{kn}/2T_o}$. Then
  $$\frac{\partial\log P(y)}{\partial E_{kn}} =
  \frac{\partial\log P(y)}{\partial|A_n|^2}\cdot 2(S_n - R_n)\,s_k\cdot
  \Big(-\frac{w_{kn}}{2T_o}\Big),\qquad s_k = \pm1.$$
  This is the direct path through $t_{kn}$. The second path, through the
  fused energy's soft-min (`models/energy_fusion.py:56`), has the same
  common-mode form when $r$ is near-uniform [HYPOTHESIS]; check 13 holds the
  collapse energy fixed and tests the direct path only.
  When $w_{kn}$ is nearly equal across $k$, the gradient row is the same vector
  for every support hypothesis, and its negative for every attack. Check 13:
  mean pairwise cosine of centered gradient rows = 1.0000 at energy spread 0.05,
  0.62 at spread 2.0, with support/attack row sums of opposite sign. Adding the
  same vector to every row raises the centered-row cosine. This matches the
  observed rise while $H$ itself stays distinct (cos 0.87).
  Side effect [HYPOTHESIS]: near cancellation,
  $\partial\log|A_n|^2/\partial(S_n-R_n) = 2/(S_n-R_n)$ is large and
  sign-unstable (check 13's gradients are of order 100). The resulting
  high-variance updates are consistent with sign-inconsistent scalar drift (F6).
- **Proposed fix:** none yet. It follows C6 and D1: once amplitudes are selective
  and hypotheses informative, the gradient rows differ. Diagnostic now: log the
  energy-row cosine separately within and across channels.
- **Alternatives considered:** a decorrelation penalty on energy rows. Rejected:
  it rewards a metric, not evidence.
- **Why the fix works:** removes the common-mode condition rather than masking it.
- **Falsifiable prediction:** within-channel energy-row cosine rises faster than
  across-channel cosine during training. If not, the common-mode mechanism is
  not the driver.
- **Verification test:** the split-by-channel metric on full ARC-Easy (measurement).
- **Risks:** none for measurement. **Needs supervisor approval?** Any change: **yes.**

---

## 3. Dependency graph and fix order

Edges read "must precede" (X → Y: fix X before Y can be done or measured).

```
E3 (seeded generation) ─┐
E7 (no implicit build) ─┼─→ [build ARC-Easy cache] ─→ B10, B5, B4, D1, D2, D4, D5 (measure)
A6 (cache edge cases) ──┘            │
                                     └─→ B1 (full splits) ─→ B3 (select val, report test)
A3 + A8 (provenance, seed in key) ─→ B2 (3 seeds) ─→ B6 (item-level paired stats)
A4 (history) ──────────────────────→ B8/F6 drift logs, C2/C6/C7 per-epoch diagnostics
A5 (config in ckpt) ───────────────→ input-ablation results (DoD §2.4)
A1, A2 (dead params) ──────────────→ DoD §2.2 gradient coverage
E6 (manifest) ─→ every reported run;  E10, E11 ─→ DoD §1.5 replay
A7 [approval] ─→ C6 re-measurement ─→ C3 pruning discussion
B8 (budget) ─→ C2 conclusions about ρ;  C6 ─→ C7
```

**Order (unchanged from issues.md §1, with three placements made explicit):**

1. **Phase 1:** E3, E7, A6 → A3 **+ A8** → A4 → A5 → A1/A2 → A7 (blocked:
   approval) → regression tests (A0 circuit, gradient coverage, E3/E7 tests).
   *Change:* A8 joins A3 because they share one fix (provenance-checked resume)
   and A8 is the case that actually happened.
2. **Phase 2:** E6 → E10, E11 → E5 → E8 (timed smoke run) → E9 → build the
   ARC-Easy cache (needs go-ahead).
   *Change:* E10 and E11 are placed with E6 because the manifest must record the
   determinism flag and the environment hash.
3. **Phase 3:** B4, B5 → B6 → B3, B7 → B11 (measurement) → full-split 3-seed
   runs (closes B1, B2, B8, B10).
   *Change:* B11 is measurement-only; it can ride along with the Phase-3 runs.
4. **Phase 4:** D1, D2, D5 measurements on the full cache → D3, D4 (approval for
   any generation change).
5. **Supervisor:** C1, C2 (option 2), C3–C7, A7.

---

## 4. Items for the supervisor meeting

| Item | Recommendation | Decision needed |
|---|---|---|
| **A7** validator target | Approve the soundness-gated target (v47 patch) as a one-variable change, after Phase 1 | Apply it? (objective change) |
| **C1** sign symmetry | Keep the readout for the baseline; report the limitation; evaluate a signed readout later | Is the readout part of the claim to defend, or open to change? |
| **C2** ρ init / saturated metric | Drop `destructive_fraction` as evidence now (no approval); consider init ρ = π/2 as a pre-registered test | Allow the init change? |
| **C3** pruning / abstention | Report accuracy–coverage only; no pruning until C6 is resolved | Agree to defer? |
| **C4** MiniLM bottleneck | Keep for the baseline; record as a known ceiling | Agree? |
| **C5** redundancy | Defer | Agree? |
| **C6** near-uniform collapse (new) | Re-measure after A7 and on full data before any change | Agree to measure first? |
| **C7** energy-row convergence (new) | Measure within/across-channel cosine; no change | Agree? |
| **D3/D5** generation | Audit attack compliance (η) first; change prompts only with evidence | Allow a cache rebuild later if η is high? |

---

## 5. Claims I must NOT make (honesty list)

1. **No accuracy claim from v46.** n=20 gives a ±0.22 half-width (check 6).
2. **No claim that qAIR beats the LLM.** 0.50 vs 0.40, b=2, c=4, p=0.6875 (check 6).
3. **No claim that interference helps.** coherent_acc 0.25 < classical_acc 0.30
   at the best checkpoint (cell 32), and the readout cannot rank by sign (C1).
4. **No claim from `destructive_fraction` = 1.0** or `phase_effect` ≈ 0.99. Both
   are guaranteed by initialization (C2, check 4).
5. **No claim that "phases don't learn".** The run's step budget caps movement
   at ≈0.011 rad, and π is a stationary point (B8, C2, check 5).
6. **No claim from the sub-chance arms** (A3 0.15, P0 0.20). They are
   consistent with chance, $P(X\le3) = 0.225$, and were resumed from stale
   checkpoints (A8).
7. **No claim about any ablation contrast in the v46 grid.** A1 was not trained
   in v46; A3/P0/F0 mixed sessions (A8); one seed.
8. **No claim that hypotheses are "below chance"** or that they "hurt". The
   statistics are consistent with chance (check 12).
9. **No claim that the model "uses its hypotheses".** The input-ablation PASS
   rests on 4–5 items of 20, and even a one-sided 5-item flip gives p = 0.0625
   (check 6).
10. **No claim that the model is miscalibrated.** ECE at n=20 has an expected
    value of 0.24 even for a calibrated model (check 10).
11. **No claim of overfitting as a finding.** Train accuracy was not logged; it
    is a hypothesis (F1).
12. **No "quantum advantage" or hardware claim of any kind.** The circuit is a
    classical simulation (CLAUDE.md), and the only cited quantum-reasoning paper
    itself calls its small gain inconclusive (Flores-Garrigós et al. 2026).
13. **No comparison of v46 numbers with v47-patched runs as if the code were
    the same.**

---

## Appendix A — `scripts/verify_dossier_math.py` output

Run with Python 3.12, numpy (CPU), 2026-10-17. Each check evaluates the
dossier's claim (must be True) and a deliberately wrong control claim (must
be False). [VERIFIED: scripts/verify_dossier_math.py]

```

=== 1. Collapse entropy vs log 8 ===
[PASS] KL(p||U) = log K - H(p) (identity)
        claim=True   control(wrong)=False  KL=0.341320
        log 8 = 2.07944; KL epoch 1 = 0.00904, KL final = 0.00314 nats
        selective references: peak 0.5 on one hypothesis -> H=1.6661, KL=0.4133; uniform on 4 of 8 -> KL=0.6931
        logit gap needed for a 0.5 peak: ln 7 = 1.9459
[PASS] observed collapse is near-uniform (KL < 0.01 nats every epoch)
        claim=True   control(wrong)=False  final KL is 132x below the 0.5-peak reference
[PASS] KL ~ Var(logits)/2 for small spread (within 5%)
        claim=True   control(wrong)=False  KL/(Var/2) = 0.999, 0.999, 0.997
        implied collapse temperature: epoch 1 T=1.003, epoch 7 T=1.033; init 0.3+softplus(0)=0.9931
[PASS] entropy is explained by diversity with T ~ 1 (both epochs within 10%)
        claim=True   control(wrong)=False  
[PASS] Gaussian logits with Var=final diversity reproduce the collapse peak (+/-0.005)
        claim=True   control(wrong)=False  predicted 0.1401, observed 0.1406

=== 2. Sign symmetry of the coherent readout ===
        S-A per option = (0.5, -0.5); coherent = [0.17361111 0.17361111]; P = [0.5 0.5]
[PASS] strongly supported and strongly refuted options tie: P0 == P1
        claim=True   control(wrong)=False  control claims the readout prefers the supported option
[PASS] with a reference term the ranking follows S - A (additive scoring)
        claim=True   control(wrong)=False  P_ref = [0.9 0.1]
[PASS] (S-A)^2 is invariant under swapping S and A for every random draw
        claim=True   control(wrong)=False  

=== 3. Validator BCE gradient under the v46 target ===
        sound hypotheses (support on y, attacks on wrong options): [0 1 0 0 1 0 1 1]
[PASS] v46: d loss/d logit[k, y] < 0 for EVERY hypothesis k (sound or not)
        claim=True   control(wrong)=False  grad[:, y] = [-0.628 -0.242 -0.635 -0.688 -0.647 -0.453 -0.314 -0.818]
[PASS] v46: option-mean of the target is 1/N for every k (no k signal)
        claim=True   control(wrong)=False  mean_n target[k] = [0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25]
[PASS] gated target (v47 candidate): unsound hypotheses are pushed DOWN at y
        claim=True   control(wrong)=False  option-means now [0.   0.25 0.   0.   0.25 0.   0.25 0.25]

=== 4. destructive_fraction at polarity_phase = pi ===
        uniform-amplitude threshold rho* = arccos(-(N-1)/N) = 2.4189 rad
[PASS] uniform amplitudes: fraction jumps 0 -> 1 at rho* (exact)
        claim=True   control(wrong)=False  
        rho -> fraction: 0.00:0.000, 1.60:0.000, 2.00:0.001, 2.40:0.479, 2.80:0.996, 3.00:1.000, 3.14:1.000
        (config.py:223-226 measured on the real model: 0.000 up to 1.6, 0.494 at 2.4, 0.994 at 2.8, 1.000 at >=3.0)
[PASS] init-scale random inputs: fraction >= 0.99 at pi, < 0.05 at 1.6
        claim=True   control(wrong)=False  
        fraction at pi vs phase spread (circuit=compat sd): 0.1:1.000, 0.5:0.903, 1.0:0.680, 2.0:0.616

=== 5. d coherent / d rho and the Adam step budget ===
[PASS] aligned channels: d coherent_n / d rho = 0 at rho = pi (all options)
        claim=True   control(wrong)=False  max|grad| at pi = 0.00e+00, at pi/2 = 1.752
[PASS] misaligned by delta: gradient at pi = 2|S||A| sin(delta)
        claim=True   control(wrong)=False  fd [0.34815 0.32447 0.26385 0.20851] vs analytic [0.34815 0.32447 0.26385 0.20851]
        Adam budget sum(lr_t) over 40 steps = 0.01127 rad; observed path length 0.00461, net drift -0.00199; weight-decay-only drift -0.00177
[PASS] observed polarity_phase movement is within the Adam step budget
        claim=True   control(wrong)=False  

=== 6. Small-sample statistics ===
[PASS] P(X <= 3 | n=20, p=0.25) = 0.225 (A3 at 3/20 is consistent with chance)
        claim=True   control(wrong)=False  P(X<=3)=0.2252, P(X<=4)=0.4148 (P0 at 4/20)
[PASS] exact McNemar b=2, c=4 -> p = 0.6875
        claim=True   control(wrong)=False  
[PASS] 5 net flips out of 20 cannot reach p < 0.05 even with no reverse flips
        claim=True   control(wrong)=False  p(b=0, c=5) = 0.0625
        95% CI half-width at p=0.5: n=20: +/-0.219, n=570: +/-0.041, n=1172: +/-0.029, n=2376: +/-0.020
[PASS] half-width ~0.22 at n=20, ~0.04 at n=570, ~0.02 at n=2376
        claim=True   control(wrong)=False  
[PASS] percentile bootstrap of 10/20 gives ~[0.30, 0.70]
        claim=True   control(wrong)=False  [0.30, 0.70]

=== 7. Bias of best-of-8-epochs accuracy at n=20 ===
        true acc p=0.4125: E[max]-p = +0.158 (independent epochs), +0.098 (correlated, keep=0.8); P(best >= 0.50) = 0.93
[PASS] best-of-8 is biased upward by several points at n=20
        claim=True   control(wrong)=False  

=== 8. Capacity and loss ===
        params/example = 13204.8; loss drop epoch 1->7 = 58.1%, 1->8 = 60.7%; predicted loss at a uniform posterior = 1.855 (epoch-1 mean 1.889); smoothed-CE floor = 0.201
[PASS] uniform-posterior loss decomposition matches epoch 1 within 0.05
        claim=True   control(wrong)=False  control: CE alone explains the epoch-1 loss

=== 9. Normalized readout depends only on amplitude ratios ===
        residuals [0.01  0.02  0.015 0.012] -> P = [0.115 0.46  0.259 0.166] (max 0.460)
[PASS] P(c * residual) == P(residual): magnitude of evidence does not temper confidence
        claim=True   control(wrong)=False  

=== 10. ECE of a calibrated model at small n ===
        expected ECE when calibrated by construction: n=20: 0.241, n=570: 0.047, n=2376: 0.023
[PASS] calibrated model at n=20 still shows ECE well above 0 (> 0.1)
        claim=True   control(wrong)=False  

=== 11. Pooled-over-seeds McNemar under H0 ===
        kappa=0.0: rejection pooled=0.043, single-seed=0.033 (nominal 0.05)
        kappa=0.5: rejection pooled=0.155, single-seed=0.038 (nominal 0.05)
        kappa=0.8: rejection pooled=0.277, single-seed=0.048 (nominal 0.05)
[PASS] pooled McNemar is valid only for independent seeds and over-rejects as seeds agree
        claim=True   control(wrong)=False  

=== 12. Train-cache zero-shot statistics (n=80) vs chance 0.25 ===
        one-sided P(X <= k | 80, 0.25): k=14 (0.1750): 0.074, k=15 (0.1875): 0.121, k=16 (0.2000): 0.184;  P(X >= 27) for the LLM likelihood = 0.050
[PASS] hypothesis statistics are consistent with chance (not significantly below)
        claim=True   control(wrong)=False  

=== 13. d log P(y) / d E_kn within a channel ===
        mean pairwise cosine of centered gradient rows, support channel: 1.0000 (energy spread 0.05), 0.6226 (spread 2.0)
        support rows sum +102.9972, attack rows sum -102.9972 (opposite signs)
[PASS] near-uniform energies: every support hypothesis gets the same option gradient
        claim=True   control(wrong)=False  

SUMMARY: 28/28 checks PASS (each with its wrong-claim control failing as required)
```
