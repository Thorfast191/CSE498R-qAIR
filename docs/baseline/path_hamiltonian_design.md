# Path Hamiltonian — a joint quantum state over all reasoning paths (DESIGN, for supervisor review)

Branch `baseline` (code byte-identical to v46 `9552498`). Written 2026-10-18.
**Design document only. No model code is changed.** The only new file besides
this one is `scripts/verify_path_hamiltonian.py`, a numpy check of the
mathematics below. Its output is in Appendix A.

Labels as in `docs/baseline/issue_dossier.md`: [VERIFIED: …] = file:line or a
command run this session; [DERIVED] = mathematics shown here and checked by the
script (check numbers refer to Appendix A); [HYPOTHESIS]; [ASSUMPTION]. All
T4 costs are [ASSUMPTION] until measured.

---

## 0. Why this, and what changes relative to v46

**The idea.** qAIR's core claim is that reasoning paths interact: one path can
cancel another, or depend on another, and the answer comes out of a collapse
over *joint* configurations of paths. **v46 never implemented this.** Its
circuit processes each hypothesis separately. `models/quantum_layer.py:250-254`
reshapes the hypotheses to `(B·K, D)`, compresses each to 6 angles and runs the
6-qubit circuit once per hypothesis [VERIFIED]. No quantum state ever spans
more than one hypothesis. Paths meet only afterwards, in a classical weighted
sum (`models/interference.py`), whose limitations are C1, C2, C6 and C7 in the
dossier.

**The proposal.** One qubit per path. The state lives in the 2^K-dimensional
space of *worlds*: which paths hold and which do not. A data-dependent
Hamiltonian couples paths pairwise, so paths can exclude or require each
other. The collapse picks low-energy worlds, and the answer is read off the
resulting distribution over worlds.

**What stays and what is replaced** (CLAUDE.md rules 6 and "ask before"):
- The pipeline order is unchanged: Generator → Encoder → Quantum layer →
  Persistent Reasoner → Validator → **Energy Scoring → Collapse** → Prediction.
  The path Hamiltonian is a new *implementation* of the Energy Scoring and
  Collapse stages: the heads produce h and J (energy), and the ground or thermal
  state is the collapse. No stage is added or removed.
- **The per-hypothesis 6-qubit circuit stays** (rule 6: do not remove the
  quantum layer). It still enriches each path embedding upstream
  (`models/full_model.py:315-319`, `H = H + q_state`). Removing it would be a
  separate, later single-variable arm, and only with supervisor approval.
- The v46 path (coherent interference readout) is untouched. The new readout
  is selected by a flag, and only in the new arms.
- Honesty, stated up front: K = 8 qubits is simulated **exactly** as a 256 × 256
  matrix on a classical GPU. Nothing here is quantum hardware, and nothing here
  supports a claim of quantum advantage (CLAUDE.md).

---

## 1. Variables

- **Paths.** K = 2N hypotheses per question (`config.py:170` `GEN_MODES =
  ("support", "attack")`, K = len(GEN_MODES)·N [VERIFIED]). For a 4-option ARC
  question K = 8; 3-option questions give K = 6, and 5-option questions K = 10.
- **One qubit per path**, i = 0…K−1. Ordering used here and in the script:
  (support₀, attack₀, support₁, attack₁, …). In the model, the ordering is
  whatever the cache gives; membership comes from `polarity` (B, K) and
  `align` (B, K, N), so the ordering never matters.
- **Basis state** x ∈ {0,1}^K is one **world**. x_i = 1 means "path i holds",
  x_i = 0 means "path i fails". Integer index Σᵢ xᵢ 2^i. Pauli-Z convention:
  Z_i|x⟩ = z_i|x⟩ with z_i = 1 − 2x_i, so z_i = −1 when the path holds.
- **State space:** ℂ^(2^K). Every amplitude here is real, because H is real
  symmetric (§2).

---

## 2. Hamiltonian

$$H \;=\; \sum_{i} h_i Z_i \;+\; \sum_{i<j} J_{ij} Z_i Z_j \;-\; \Gamma \sum_i X_i
\qquad (2^K \times 2^K,\ \text{real symmetric})$$

- **Diagonal (classical Ising) part.** ⟨x|H|x⟩ = E(x) = Σᵢ hᵢzᵢ + Σ_{i<j} J_ij zᵢzⱼ.
- **Off-diagonal (transverse field).** ⟨x|H|x′⟩ = −Γ exactly when x and x′
  differ in one bit, and 0 otherwise. Hermiticity and both entry rules are
  verified in check 1.
- **Signs, verified in check 2:**
  - h_i > 0 favours x_i = 1 (path holds).
  - J_ij > 0 favours *exactly one* of i, j holding: **paths cancel**, i.e.
    exclude each other.
  - J_ij < 0 favours i and j *agreeing*: **paths need each other**. Note
    that "agreeing" includes "both fail". A Z_iZ_j coupling cannot express
    "i requires j" one-directionally; that would need a different operator
    (e.g. x_i(1−x_j)) and is out of scope.
- **Γ ≥ 0**, learned: Γ = softplus(γ), γ a scalar.

### 2.1 Heads and shapes (B = 16, d = MODEL_DIM = 128 [VERIFIED: `config.py:114`])

Inputs, all already computed in v46 forward
(`models/full_model.py:257-408` [VERIFIED]):

| Tensor | Shape | Source |
|---|---|---|
| e (post-reasoning, post-circuit, post-feedback path embeddings) | (B, K, d) | `H` after `models/full_model.py:349` |
| validator energy per path | (B, K) | `validator_out["validator_energy"]` (`models/validator.py:163-170`) |
| path–option energy | (B, K, N) | `selector["energy"]` (`models/full_model.py:369`) |
| polarity | (B, K), ±1 | input |
| align | (B, K, N), one-hot over options | input |
| H_mask, O_mask | (B, K), (B, N) | input |

**Unary head → h.**
- Own-option energy a_i = Σₙ align_in · energy_in: (B, K, N) → (B, K).
- Features u_i = [e_i, validator_energy_i, a_i, polarity_i]: (B, K, d+3) = (B, K, 131).
- MLP 131 → 64 → 1 (GELU), then h = c·tanh(·): (B, K, 131) → (B, K, 1) → (B, K).
- The LLM prior is **not** a unary feature. It stays where v46 adds it, at the
  output, and only in prior-on variants (§6), so prior-off arms remain
  comparable with A3.

**Pairwise head → J.**
- Features f_ij = [e_i + e_j, e_i ⊙ e_j, polarity_i, polarity_j,
  same_option_ij], with same_option_ij = Σₙ align_in·align_jn ∈ {0, 1}:
  (B, K, K, 2d+3) = (B, K, K, 259).
- MLP 259 → 64 → 1 (GELU), giving g: (B, K, K).
- **Symmetrize**: the [polarity_i, polarity_j] slot is ordered, so g is not
  symmetric by itself. J = c·tanh(½(g + gᵀ)): (B, K, K). Then zero the
  diagonal and set J_ij = 0 wherever either path is masked.

**Parameters** [DERIVED]: unary 131·64 + 64 + 64 + 1 = 8,513; pairwise
259·64 + 64 + 64 + 1 = 16,705; plus γ, and τ for the temperature (§4) =
**25,220**, i.e. +2.4% on v46's 1,056,382 [OBSERVED cell 22].

**Scale.** The ground state is invariant to multiplying (h, J, Γ) by any
c > 0 (the eigenvectors of cH equal those of H), so the overall scale is not
identified. Bounding h and J with c·tanh fixes it. c = 2 [ASSUMPTION; to be
set on validation in the CPU smoke test, not tuned on test].

### 2.2 Building H (shapes)

Constant buffers per K, built once:
- Z table: z (2^K, K)
- pair table: zz (K(K−1)/2, 2^K)
- transverse matrix: X_sum (2^K, 2^K), with K·2^K non-zeros

Per batch:
1. E = h @ zᵀ + J_pairs @ zz: (B, K) and (B, K(K−1)/2) → **(B, 2^K)**
2. H = diag_embed(E) − Γ·X_sum: **(B, 2^K, 2^K)** = (16, 256, 256) for K = 8

**Mixed option counts.** A batch may mix K = 6, 8 and 10. Group the items by K
and build one H per group: (B_K, 2^K, 2^K). Padding K = 6 items to 8 qubits
would quadruple the matrix and needs "frozen" qubits, which a Γ > 0 term would
still leak amplitude into. Grouping avoids both.

---

## 3. Where the classical data enters the quantum state

**Explicitly: the per-question classical data enters the quantum state here,
and only here.** It enters through three things:
1. **h**, the unary energies (from that question's path embeddings, validator
   energies and path–option energies);
2. **J**, the pairwise couplings (from pairs of that question's path
   embeddings, their polarities and whether they argue about the same option);
3. **the basis encoding of paths**: qubit i *is* path i of this question, and
   basis state x *is* a world over this question's paths.

The state is then *defined* by the data, as the ground (or thermal) state of
H(h, J, Γ). This is Hamiltonian encoding.

**Why no separate dataset-encoding stage** (no AngleEmbedding,
amplitude encoding or feature map in front of it):
- **There is no initial state to load data into.** The state is the lowest-energy
  state of a data-dependent operator, not the output of a circuit applied to an
  input state. A feature map would have nothing to act on.
- **The register is the wrong shape for features.** 8 qubits hold one bit of
  meaning per path (holds/fails). The 128-d path embeddings (1,024 numbers for
  K = 8) do not fit into a 256-dimensional state in any interpretable way.
  Encoding them would turn the "world" basis into an opaque feature space, and
  the meaning of x would be lost.
- **One channel, interpretable parameters.** With data entering only via h and
  J, every number in H has a stated meaning (a path's standalone plausibility, a
  pair's exclusion or dependence), and the diagnostics in §7 read them directly.
  A second, unconstrained encoding channel would make "the paths cancelled each
  other" untestable.
- **The per-path quantum feature map already exists.** It is the v46 6-qubit
  circuit upstream (§0), which stays. Adding another encoding stage would
  duplicate it and add a pipeline stage (CLAUDE.md "ask before").

---

## 4. Collapse

### 4.1 Ground state (as proposed)

- λ, V = `torch.linalg.eigh(H)`: (B, 2^K) and (B, 2^K, 2^K), ascending
  eigenvalues.
- ψ₀ = V[:, :, 0]: (B, 2^K); P(x) = |ψ₀(x)|²: (B, 2^K), with Σₓ P(x) = 1
  (check 8).
- At Γ = 0, ψ₀ is the one-hot minimum-energy world (check 3).
- At Γ > 0, H is **stoquastic** (all off-diagonals ≤ 0), so by Perron–Frobenius
  ψ₀ is unique and has single-signed amplitudes (check 4: 200/200 draws).
  Two consequences:
  - The ψ₀² readout is well defined, and the sign ambiguity of eigh's columns
    is harmless.
  - **There is no destructive interference of amplitudes inside ψ₀.** In this
    design, "paths cancel" means *energetic* exclusion through J > 0. It does
    not mean negative amplitudes, unlike v46's phase-π cancellation.

### 4.2 The problem with "H0 = Γ = 0, otherwise identical" under ground-state collapse

At Γ = 0, P is one-hot on the argmin world. A small change of h or J leaves
the argmin unchanged almost everywhere, so **the loss has zero gradient
almost everywhere and H0 cannot be trained**. Check 6 confirms this: the
finite-difference gradient is exactly 0 in 50/50 random draws. A ground-state
H0 is therefore not a usable control.

### 4.3 Recommended collapse for the arm pair: thermal state at learned temperature

$$P_T(x) = \frac{\langle x|e^{-H/T}|x\rangle}{\mathrm{Tr}\,e^{-H/T}},\qquad T = 0.05 + \mathrm{softplus}(\tau)$$

- At Γ = 0 this is **exactly** the classical Gibbs distribution
  softmax(−E/T) (check 7, max difference 0.0). Its gradient is non-zero (check
  6, 50/50). So H0 is "the classical Ising model over worlds".
- At Γ > 0 it is the quantum thermal state. As T → 0 it **converges to the
  ground-state collapse ψ₀²** (check 7: total-variation distance 0.244 at
  T = 1, 2.7e-16 at T = 0.01). So §4.1 is its zero-temperature limit, and a
  learned T can approach it.
- **One variable.** H0 and H1 then differ only in Γ (Γ ≡ 0 vs learned), as the
  arm specification requires.
- **Computation.**
  - H0: softmax(−E/T) directly, (B, 2^K). No matrix is needed.
  - H1: `torch.linalg.matrix_exp(−(H − c·I)/T)` in float64, take the diagonal,
    normalize. P is invariant to the shift c, so c can be a detached ground
    energy from `torch.linalg.eigvalsh` (no gradient needed). Then every
    exponent is ≤ 0 and the largest is exactly 0. A cheaper Gershgorin bound,
    c = minₓE(x) − KΓ, leaves up to KΓ/T of slack, which can underflow at small
    T [DERIVED from the Gershgorin disc radius KΓ], so it is only a fallback.
    The backward is the Fréchet
    derivative, equivalently the Daleckii–Krein divided-difference formula. It
    is smooth even when **excited** states are degenerate, which an eigh-based
    backward is not. Check 7 verifies this gradient against finite differences
    (relative error 1.8e-9).
- The ground-state collapse of §4.1 is kept as a **secondary arm, H1-GS**, so
  the proposed formulation is also tested (§6).

### 4.4 Near-degenerate ground states (H1-GS)

- **Γ > 0 guarantees a unique ground state** (check 4), but the gap can be tiny.
  With a global flip symmetry (h = 0), the gap is 0 at Γ = 0, 7.9e-8 at
  Γ = 0.01, 7.6e-4 at Γ = 0.1 and 1.04 at Γ = 1 (check 10). The eigenvector
  derivative scales as 1/(λ₀ − λ₁), so gradients blow up near such points.
- **At Γ = 0, eigh returns an arbitrary vector inside a degenerate subspace.**
  P read from one column depends on that arbitrary choice (max difference
  0.415 in check 10), while the projector average over the subspace does not.
- **Handling**, in order:
  1. Compute in float64.
  2. Floor Γ in H1-GS: Γ = 0.05 + softplus(γ) [ASSUMPTION: floor value]. H1
     (thermal) needs no floor.
  3. Broadened backward: replace 1/(λ₀ − λ_k) by
     (λ₀ − λ_k)/((λ₀ − λ_k)² + δ²) with δ = 1e-3 [ASSUMPTION]. This is a custom
     autograd function, verified against finite differences away from
     degeneracy.
  4. If λ₁ − λ₀ < δ, read P from the projector onto all eigenvectors within δ
     of λ₀, averaged: P(x) = Σ_{k: λ_k − λ₀ < δ} ψ_k(x)² / m. This is
     basis-invariant (check 10).
  5. Log the fraction of items that hit cases 3 or 4 (§7).

---

## 5. Answer and loss

- **Answer masks** (B, N, 2^K):
  m_n(x) = Π_{k: align_kn = 1, pol_k = +1} x_k · Π_{k: align_kn = 1, pol_k = −1} (1 − x_k).
  With one support and one attack per option this is
  x_sup(n) · (1 − x_att(n)), as proposed. They are built per item from `align`
  and `polarity`; masked paths and options are excluded.
- **Score:** s_n = Σₓ P(x) m_n(x): (B, 2^K) × (B, N, 2^K) → (B, N). This is the
  probability that option n's support holds and its attack fails. It lies in
  [0, 1] (check 8).
- **Normalize:** p_n = (s_n + ε)/Σ_m(s_m + ε), ε = 1e-6. The raw s_n do not sum
  to 1: several options can be "on" in one world, or none (in check 8 they
  summed to 1 in only 3 of 200 draws). After normalization Σp = 1 and every
  p_n > 0 (check 8). ε prevents log 0 when every option scores 0. Padded options
  are masked with the same −30 floor as v46 (`models/full_model.py:428-440`).
- **Prior-on variants only:** add `softplus(llm_prior_weight) · z(llm_logprob)`
  to log p and re-normalize, exactly as v46 does (`models/full_model.py:416-440`).
- **Loss.** `scores = log p` goes into the existing `training/losses.py`
  unchanged:

  smoothed cross-entropy + 0.4 · ranking + 0.1 · validator BCE.

  The path arms return `diversity = None`, so the −0.02·tanh(diversity) term
  is absent (`training/losses.py:139-145` skips it when None [VERIFIED]). It is
  identical for H0 and H1.

**Differentiable end to end.**
- loss → log p → s → P: sums, products and a normalization; smooth for ε > 0.
- P → H, thermal (H0, H1):
  - H0: softmax of −E/T, smooth.
  - H1: matrix_exp and trace, whose Fréchet derivative exists for every
    symmetric H, degenerate or not (check 7, rel. error 1.8e-9).
- P → H, ground state (H1-GS): ψ₀ by eigh. Its derivative
  dψ₀ = Σ_{k>0} ψ_k ⟨ψ_k|dH|ψ₀⟩ / (λ₀ − λ_k) exists whenever the ground state is
  non-degenerate, which Γ > 0 guarantees (check 4). It matches finite
  differences to 1.1e-8 relative error on K = 4 for all 11 parameters
  (4 h, 6 J, γ) (check 5). torch's eigh backward implements this formula
  [ASSUMPTION: from the PyTorch documentation, which also warns that it is
  unstable for close eigenvalues; hence §4.4].
- H → (h, J, Γ): linear in h and J, and in Γ through softplus.
- (h, J) → heads → e, validator energy, path–option energy → reasoner, circuit,
  encoder projection: the existing v46 graph, already differentiable.

So every parameter of the path arms receives gradient. The Tier 0
gradient-coverage test (DoD §2.2) must pass for H0, H1 and H1-GS.

---

## 6. Arms and pre-registration

| Arm | Collapse | Γ | Everything else |
|---|---|---|---|
| **H0** | thermal, P_T = softmax(−E/T) | ≡ 0 (classical Ising, Gibbs) | A3 config + path readout |
| **H1** | thermal, diag e^{−H/T}/Tr | learned, softplus(γ) | identical to H0 |
| H1-GS (secondary) | ground state ψ₀², §4.4 handling | learned, floored at 0.05 | identical to H1 |

- **Base config: A3** (quantum + validator, LLM prior off;
  `training/ablations.py:109` [VERIFIED]). With the prior off, the paths, not
  the prior, have to carry the answer.
- **Prior-on variants** (H0-F, H1-F) are run only if the supervisor wants a
  deployed-system comparison (decision P-3).
- **Comparisons**, all on full ARC-Easy, seeds 42/43/44, checkpoint selected on
  validation, test evaluated once, per-seed exact McNemar plus an item-level
  paired bootstrap of the seed-averaged difference (improvement_plan.md §0):
  - **Primary contrast: H1 vs H0** (the single variable Γ). Tested alone at
    α = 0.05.
  - **Secondary family** (Holm): H1 vs v46 A3 (reference run, Tier 0′); H0 vs
    v46 A3; H1 vs LLM cloze; H1 vs LLM letter pick; H1-GS vs H1.
- Initial values: γ such that Γ₀ = 0.5, τ such that T₀ = 1.0 [ASSUMPTION]. Γ,
  T and γ belong to the scalar LR group of improvement_plan.md T2c.

### Pre-registration (fixed now, before any run)

**Hypothesis Q ("quantum superposition between paths helps").** Holding the
energy model fixed, letting worlds superpose (Γ > 0) gives better answers than
the classical distribution over the same worlds (Γ = 0).

**Q is supported only if all of these hold:**
1. H1 − H0 test accuracy > 0, with the paired-bootstrap 95% CI excluding 0,
   and H1 ahead of H0 in all 3 seeds.
2. The learned Γ stays in use. Γ/(‖J‖_F/K) ≥ 0.05 at the selected checkpoint
   in all 3 seeds [ASSUMPTION: the 0.05 floor is a choice, fixed here]. If the
   transverse field shrinks to irrelevance, the model has chosen to be classical.
3. On validation, the median entanglement entropy of the state across the
   supports | attacks bipartition is > 0.01 nats. For thermal H1 this is
   computed on the ground state of the learned H.

**Q is falsified (on ARC-Easy, at this cache) if:**
- Γ collapses: Γ/(‖J‖_F/K) < 0.05 in ≥ 2 of 3 seeds. The data preferred the
  classical model. Or:
- H1 − H0 ≤ 0, or its CI includes 0 while the minimum detectable difference for
  the observed discordance is ≤ 3 points. The test had the power to see a
  3-point effect and saw none. Or:
- H1 > H0 significantly but condition 2 fails. The gain is then not
  attributable to superposition (an optimization or initialization difference),
  and Q is **not** supported.

**Inconclusive:** H1 − H0's CI includes 0 and the minimum detectable
difference exceeds 3 points (an under-powered test).

**Prediction** [HYPOTHESIS, stated so that it can fail]: if the Tier 1 cache
passes the gate, H1 ≥ H0 by less than 2 points, and Γ stays in use. If the
cache fails the gate, H1 ≈ H0 ≈ chance-level reasoning, and the result says
nothing about Q (§10).

**What a positive result would and would not mean.** H is stoquastic (§4.1).
Ground and thermal states of stoquastic Hamiltonians are, in general,
classically samplable by path-integral Monte Carlo, which maps a
transverse-field Ising model onto a classical Ising model with one extra
"imaginary-time" dimension [ASSUMPTION: a standard result, quoted from memory;
cite a source before using it in the thesis]. So even a clear H1 > H0 shows
that *this transverse-field relaxation of the classical collapse* helps. It
does **not** show a quantum advantage, and it must not be described as one.

---

## 7. Diagnostics (logged per epoch on validation; final on test)

| Diagnostic | Definition | What it tells |
|---|---|---|
| Γ, T | the learned scalars; also Γ/(‖J‖_F/K) | whether superposition is in use (pre-registration condition 2) |
| ‖J‖_F | mean over items of the Frobenius norm of masked J | whether pairwise structure is learned at all |
| J by pair type | mean and sign fraction of J for (support_n, attack_n of the same option), (support_n, support_m), (attack_n, attack_m), (support_n, attack_m) | the expected structure is J > 0 between a support and an attack on the same option; random signs mean no learned structure |
| ground-state gap | λ₁ − λ₀: min, median, fraction < δ | degeneracy risk; how often §4.4 handling fires |
| entanglement entropy | von Neumann entropy of ψ₀ across (i) supports \| attacks and (ii) options {0…⌈N/2⌉−1} \| rest, in nats (max 4 ln 2 = 2.77 for 4 \| 4 qubits) | entanglement between paths (pre-registration condition 3); 0 means a product state (check 9) |
| ground-state/answer agreement | the MAP world x* = argmax P(x); fraction of items where x* switches on exactly one option (m_n(x*) = 1 for one n), and fraction where that option equals the predicted argmax p_n | whether the collapse picks a coherent world or spreads mass over incoherent ones |
| world entropy | −Σₓ P(x) log P(x) / (K ln 2) | collapse selectivity, the analogue of C6 |
| H1 − H0 item disagreement | items where the two arms differ, by seed | the discordant count that drives the McNemar power |

---

## 8. Cost

**Measured on this machine** [VERIFIED: `scripts/verify_path_hamiltonian.py
--bench`, Apple M4 CPU, numpy 2.5.4]: `numpy.linalg.eigh` forward on a batch of
16 matrices takes **47 ms at K = 8 (256 × 256)** and **1.15 s at K = 10
(1024 × 1024)**, the same in float32 and float64.

**Memory** [DERIVED]:
- K = 8: one H is 256² × 4 B = 262 kB, so a batch of 16 is 4.2 MB (fp32).
- K = 10: a batch of 16 is 67 MB (fp32) or 134 MB (fp64).
- Eigenvectors or matrix_exp intermediates multiply this by a small constant.
  All of it is negligible on a 16 GB T4.

**T4 time per training step, batch 16** [ASSUMPTION; to be measured]:

| | K = 8 | K = 10 |
|---|---|---|
| H0 (softmax over worlds) | < 1 ms | ~1 ms |
| H1 thermal (eigvalsh shift + matrix_exp fwd + bwd, fp64) | ~20–120 ms | ~0.5–4 s |
| H1-GS (eigh fwd + bwd, fp64; T4 fp64 throughput is ~1/32 of fp32) | ~20–100 ms | ~0.5–3 s |

- Against an assumed 1–3 s per step for v46 (improvement_plan.md §0.2), K = 8
  adds little. K = 10 could double the step time *for the batches that contain
  5-option items*. That depends on how many ARC-Easy items have 5 options; the
  histogram is computed from the Tier 0 cache, not assumed.
- **Fallback if K = 10 is too slow:** run batched power iteration on
  (cI − H) for the ground state, or a Krylov `expm_multiply` for the thermal
  diagonal. Both avoid the full 1024³ decomposition [ASSUMPTION: only needed if
  measurement says so].

**CPU micro-benchmark plan** (before any T4 run; no data, synthetic H):
1. torch CPU, fp32 and fp64: `torch.linalg.eigh` and `torch.linalg.matrix_exp`,
   forward + backward (loss as in §5), batch 16, K = 6, 8, 10; 20 repetitions
   after 3 warm-ups; report median and IQR.
2. Correctness on the same code path: `torch.autograd.gradcheck` in fp64 at
   K = 4 for H0, H1 and H1-GS; compare with this script's analytic gradients.
3. Repeat (1) on a T4 inside the E8 timed smoke run, and record it in the
   manifest. Replace every [ASSUMPTION] in the table above.
4. Report the full-model step time with each arm on 20 questions, vs v46 A3.

---

## 9. Integration (planned; implemented only after approval)

| File | Change | v46 path affected? |
|---|---|---|
| `models/path_hamiltonian.py` (new) | `PathHamiltonian(nn.Module)`: unary and pairwise heads, constant Z/ZZ/X buffers per K, grouping by K, collapse modes `"gibbs"` (H0), `"thermal"` (H1), `"ground"` (H1-GS) with §4.4 handling; returns `scores` (B, N), `world_probs` (B, 2^K) per K-group, and the §7 diagnostics | — |
| `models/full_model.py` | constructor flag `readout ∈ {"interference", "path"}` (default `"interference"`). With `"path"`: build `PathHamiltonian`, and **do not build** `CoherentCollapse`, `CollapseController`, `EnergyFusion`, `polarity_phase` or `phase_head` (they would receive no gradient, i.e. new A1/A2-type dead parameters). The forward branches after the selector (`models/full_model.py:364`); the v46 branch is byte-identical | No |
| `training/ablations.py` | arms `H0_path_ising`, `H1_path_quantum`, `H1GS_path_ground` (A3 + `readout="path"` + collapse mode); `DEFAULTS` gains `readout="interference"` | No (existing arms resolve to the old default) |
| `training/evaluate.py` | log the §7 diagnostics when `out` has `path` keys; read v46-only keys with `.get` (`classical_weights`, `interference_ratio` etc., `training/evaluate.py:143-170` [VERIFIED]), which path arms do not return | No (same values for v46 arms) |

- **That is 4 files**, more than CLAUDE.md's 3-file threshold, so it needs
  approval before implementation.
- **Checkpoints:** v46 checkpoints are unaffected. Path-arm checkpoints carry
  their config (the A3/A5 fixes), so they are never loaded into an
  interference model.
- **Tests (CPU):**
  - port checks 1–10 to torch;
  - gradient coverage for the three arms;
  - "v46 arm outputs are unchanged" (same seed, same batch, `readout` default
    vs before the change, bitwise).

---

## 10. Prerequisites: Tier 0 and Tier 1

- **Tier 0** (improvement_plan.md). Without it, nothing here can be measured:
  - seeds must not resume each other's checkpoints (A3/A8);
  - the reference A3 run and the letter-pick and majority-vote baselines must
    exist on the same items;
  - the statistics must be item-level (B6);
  - the gradient-coverage test must exist to catch dead parameters in the new
    arms.
- **Tier 1, and why D1 blocks this design specifically.** h and J are learned
  functions of the path embeddings. If the hypotheses carry no information
  about the answer beyond the question and options (D1; v46 gate at chance
  [OBSERVED cell 14]), then by the data-processing inequality (dossier D1),
  neither h nor J can carry any either. The distribution over worlds is then
  unrelated to the answer, whatever Γ is.
- The pairwise head is **more** data-hungry than any unary readout. Whether two
  paths exclude or support each other is a property of the *pair*: it needs the
  paths' content to be informative, and it must be learned from 2251 questions ×
  K(K−1)/2 = 28 pairs each. A unary signal is a necessary condition for a
  learnable pairwise one: J = 0 is optimal when nothing distinguishes paths.
- A test of "superposition between paths helps" on a cache without signal can
  only return "no difference". That would be misreported as evidence against Q,
  when it is evidence about the cache. So H0/H1 run only on a cache that passed
  the Tier 1 gate.
- **Recommended, not required:**
  - T2a (A7) first, because h reads the validator energy, whose v46 target has
    no hypothesis axis;
  - T2c's scalar LR group for γ and τ.

---

## 11. Risks

| Risk | Mechanism | Mitigation / what we report |
|---|---|---|
| **Degenerate or near-degenerate ground states** | symmetric energies make λ₁ − λ₀ → 0; eigenvector gradients ∝ 1/gap (check 10: gap 7.9e-8 at Γ = 0.01) | thermal collapse for the primary pair; for H1-GS a Γ floor, broadened backward, projector averaging and fp64 (§4.4); log the gap distribution |
| **J does not learn without signal** | if paths are uninformative, J = 0 minimizes the loss; the pairwise head has 16,705 parameters and only 2251 questions | Tier 1 gate first; log ‖J‖_F and J by pair type; if ‖J‖_F stays near its initial value, report "no pairwise structure learned" |
| **Γ collapses to 0** | if the classical Gibbs distribution fits better, gradient descent drives softplus(γ) towards 0, and H1 becomes H0 | this is an outcome, not a bug: pre-registration condition 2 makes it a falsifier. Do not add a Γ floor to H1 to prevent it (H1-GS has a floor for numerical reasons only, and is secondary) |
| **Overfitting the pairwise head** | 25,220 new parameters; pair features are high-dimensional (259) | weight decay as in v46; report train vs validation accuracy (E6 logs train accuracy); optionally a narrower pair MLP (32 hidden) as a pre-declared variant, chosen on validation |
| Saturated collapse | low T or large ‖E‖ makes P one-hot and gradients tiny (check 6: smallest Gibbs gradient 6.1e-9 on one draw) | learned T with a 0.05 floor; tanh bound on h and J; log world entropy |
| "Both fail" ambiguity of J < 0 | Z_iZ_j coupling rewards agreement, including joint failure | stated in §2; diagnostics split J by pair type; a one-directional "requires" operator is future work |
| Over-claiming | the state is exactly simulated, and H is stoquastic | §6 last paragraph; CLAUDE.md honesty rules; never "quantum advantage" |
| Cost at K = 10 | 1024³ decompositions | group by K; measure first (§8); Krylov fallback |

---

## 12. Plain-language summary for my supervisor

The original qAIR idea is that reasoning paths should interact: one argument
can rule another out, or two arguments can only stand together, and the answer
should come from the most consistent combination of arguments. The capstone
code never did this. Its quantum circuit looked at each argument on its own,
and the arguments were only added up afterwards. This design gives each of the
8 arguments for a question one qubit, so that the quantum state describes
every possible combination of arguments being true or false (256 "worlds").
The model learns how plausible each argument is and how each pair of
arguments relates (exclusive or mutually supporting). The final state
concentrates on the most consistent worlds, and the answer is read from them.
We compare two versions that are identical except for one switch: one where
worlds can be in quantum superposition, and one where they cannot (a classical
model over the same worlds). We fix in advance what result would show that
superposition helps and what result would show it does not. It is a small,
exactly simulated system, so even a positive result is not "quantum advantage",
and we will not call it that. It only makes sense to run once the generated
arguments carry real information about the answer, which the capstone's did
not. It therefore comes after the trustworthiness and evidence-quality work
already planned.

---

## Appendix A — `scripts/verify_path_hamiltonian.py --bench` output

Run 2026-10-18 on an Apple M4 CPU, Python 3.12.12, numpy 2.5.4. Each check
evaluates the design's claim (must be True) and a deliberately wrong control
claim (must be False). [VERIFIED: command above]

```
numpy 2.5.4, Python 3.12.12

=== 1. H is Hermitian (real symmetric) and has the stated entries ===
        K=8, 256x256, 20 random draws
[PASS] H == H^T, real
        claim=True   control(wrong)=False  
[PASS] diag(H) equals the brute-force Ising energy E(x)
        claim=True   control(wrong)=False  
[PASS] off-diagonal = -Gamma exactly on Hamming-distance-1 pairs, else 0
        claim=True   control(wrong)=False  

=== 2. Sign conventions at Gamma = 0 (K = 2) ===
        h=(+1,0): minimum at x=(1, 0); J=+1 minima [(0, 1), (1, 0)]; J=-1 minima [(0, 0), (1, 1)]
[PASS] h_i > 0 favours x_i = 1 (path holds)
        claim=True   control(wrong)=False  
[PASS] J_ij > 0 favours exactly one of i, j holding (paths cancel)
        claim=True   control(wrong)=False  
[PASS] J_ij < 0 favours i, j agreeing: both hold OR both fail
        claim=True   control(wrong)=False  

=== 3. At Gamma = 0 the ground state is the minimum-energy basis state ===
        K=8, 200 random (h, J): one-hot at argmin E in 200/200
[PASS] ground state = |argmin_x E(x)>, ground energy = min E
        claim=True   control(wrong)=False  

=== 4. Gamma > 0: unique ground state with single-signed amplitudes ===
        K=8, 200 draws, Gamma in [0.05, ~3]: non-degenerate 200/200, single-signed 200/200, smallest gap 8.22e-03
[PASS] ground state non-degenerate and positive up to a global sign (stoquastic H)
        claim=True   control(wrong)=False  

=== 5. Gradient through eigh matches finite differences (K = 4) ===
        10 random draws; parameters: 4 h, 6 J, 1 a (Gamma = softplus(a)); worst relative error 1.12e-08
[PASS] analytic eigh gradient == central finite differences (rel err < 1e-5)
        claim=True   control(wrong)=False  

=== 6. Ground-state collapse at Gamma = 0 has zero gradient (H0 as specified) ===
        K=4, 50 draws: ground-state grad == 0 in 50/50; Gibbs (T=0.5) grad != 0 in 50/50 (smallest max|grad| 6.1e-09; FD noise ~1e-10)
[PASS] Gamma = 0 ground-state loss is piecewise constant (zero gradient a.e.)
        claim=True   control(wrong)=False  
[PASS] classical Gibbs collapse softmax(-E/T) has a non-zero gradient
        claim=True   control(wrong)=False  

=== 7. Thermal collapse: Gibbs at Gamma = 0, ground state as T -> 0 ===
        K=8: max|P_thermal(Gamma=0) - softmax(-E/T)| = 0.0e+00; TV(thermal, psi0^2) at T=1, 0.1, 0.01: 0.244, 0.000, 2.7e-16
[PASS] diag(exp(-H/T))/Tr at Gamma = 0 equals the classical Gibbs distribution
        claim=True   control(wrong)=False  
[PASS] thermal collapse -> ground-state collapse as T -> 0
        claim=True   control(wrong)=False  
        K=4 thermal (T=0.6): Daleckii-Krein vs finite differences, rel err 1.81e-09
[PASS] thermal-collapse gradient (divided differences, degeneracy-safe) matches FD
        claim=True   control(wrong)=False  

=== 8. Answer marginals ===
        K=8, 200 draws; raw scores summed to 1 in 3/200
[PASS] sum_x P(x) = 1
        claim=True   control(wrong)=False  
[PASS] 0 <= score_n <= 1 (a marginal probability)
        claim=True   control(wrong)=False  
[PASS] normalized p_n sums to 1, every p_n > 0
        claim=True   control(wrong)=False  

=== 9. Entanglement entropy of psi0 across supports | attacks ===
        S(J=0, Gamma=0.8) = -1.33e-15; S(Gamma=0) = -0.00e+00; S(J random, Gamma=0.8) = 0.1352 nats (max 2.7726)
[PASS] no couplings -> product state, S = 0
        claim=True   control(wrong)=False  
[PASS] Gamma = 0 -> basis state, S = 0
        claim=True   control(wrong)=False  
[PASS] couplings and Gamma > 0 -> S > 0 (entanglement between paths)
        claim=True   control(wrong)=False  

=== 10. Near-degenerate ground states ===
        h = 0 (global flip symmetry), gap at Gamma = 0, 0.01, 0.1, 1: 0.00e+00, 7.88e-08, 7.61e-04, 1.04e+00
        Gamma = 0 degenerate subspace dim 2; |P(first vector) - P(rotated first vector)|max = 0.415
[PASS] exact degeneracy at Gamma = 0, h = 0; Gamma > 0 opens a gap
        claim=True   control(wrong)=False  
[PASS] projector-averaged P is basis-invariant; a single eigh column is not
        claim=True   control(wrong)=False  

=== Benchmark (informational): numpy.linalg.eigh, batch 16, this CPU ===
        K= 8 (256x256) float64:     47.6 ms per batch of 16 (forward only)
        K= 8 (256x256) float32:     47.3 ms per batch of 16 (forward only)
        K=10 (1024x1024) float64:   1151.4 ms per batch of 16 (forward only)
        K=10 (1024x1024) float32:   1165.8 ms per batch of 16 (forward only)

SUMMARY: 22/22 checks PASS (each with its wrong-claim control failing as required)
```

## Appendix B — decisions this design needs

| # | Decision | Recommendation |
|---|---|---|
| P-1 | Approve the H0/H1 arms (thermal collapse, single variable Γ) and the secondary H1-GS arm | Approve, after the Tier 1 gate |
| P-2 | Accept the thermal collapse for the primary pair, since a ground-state H0 at Γ = 0 has zero gradient (check 6) | Accept; ground state kept as H1-GS and as the T → 0 limit |
| P-3 | Run prior-on variants (H0-F, H1-F) for a deployed-system comparison? | Only if H1 vs H0 is positive |
| P-4 | Approve the 4-file change (§9), over the 3-file threshold | Approve, as one reviewed change behind a default-off flag |
| P-5 | Keep the per-hypothesis 6-qubit circuit upstream (rule 6), or test removing it as a later single-variable arm | Keep; removal only as a separate arm |
