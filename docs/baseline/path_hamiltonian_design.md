# Path Hamiltonian — a joint quantum state over all reasoning paths (DESIGN, for supervisor review)

Branch `baseline` (code byte-identical to v46 `9552498`). Written 2026-10-18;
revised 2026-10-18 (thermal readout accepted; H2 non-stoquastic arm; directed
pair terms; verified references).
**Design document only. No model code is changed.** The only other file is
`scripts/verify_path_hamiltonian.py`, a numpy check of the mathematics. Its
output is in Appendix A.

**Labels** (as in `docs/baseline/issue_dossier.md`):
- [VERIFIED: …] = file:line, a command run this session, or a reference checked
  against its arXiv or publisher page this session (§13);
- [DERIVED] = mathematics shown here and checked by the script ("check n" refers
  to Appendix A);
- [HYPOTHESIS]; [ASSUMPTION].

All T4 costs are [ASSUMPTION] until measured.

---

## 0. Why this, and what changes relative to v46

**The idea.** qAIR's core claim is that reasoning paths interact: one path can
rule another out, or depend on another, and the answer comes out of a collapse
over *joint* configurations of paths. **v46 never implemented this.** Its
circuit processes each hypothesis separately. `models/quantum_layer.py:250-254`
reshapes the hypotheses to `(B·K, D)`, compresses each to 6 angles and runs the
6-qubit circuit once per hypothesis [VERIFIED]. No quantum state ever spans
more than one hypothesis. Paths meet only afterwards, in a classical weighted
sum (`models/interference.py`), whose limitations are C1, C2, C6 and C7 in the
dossier.

**The proposal.** One qubit per path. The state lives in the 2^K-dimensional
space of *worlds*: which paths hold and which do not. A data-dependent
Hamiltonian couples paths pairwise, so paths can exclude, require or agree
with each other. The collapse concentrates on low-energy worlds, and the
answer is read from the resulting distribution over worlds.

**Three arms isolate three claims:**
- **H0** (classical: no superposition);
- **H1** (superposition, but provably no interference between paths);
- **H2** (superposition with sign structure, so paths can interfere
  destructively).

**What stays and what is replaced** (CLAUDE.md rules 6 and "ask before"):
- The pipeline order is unchanged: Generator → Encoder → Quantum layer →
  Persistent Reasoner → Validator → **Energy Scoring → Collapse** → Prediction.
  The path Hamiltonian is a new *implementation* of the Energy Scoring and
  Collapse stages: the heads produce the energy terms, and the thermal state is
  the collapse. No stage is added or removed.
- **The per-hypothesis 6-qubit circuit stays** (rule 6). It still enriches each
  path embedding upstream (`models/full_model.py:315-319`, `H = H + q_state`).
  Removing it would be a separate, later arm, and only with approval.
- The v46 readout (coherent interference) is untouched. The new readout is
  selected by a flag, in the new arms only.
- Honesty: K = 8 qubits are simulated **exactly** as 256 × 256 matrices on a
  classical GPU. Nothing here is quantum hardware or evidence of quantum
  advantage (CLAUDE.md).

---

## 1. Variables

- **Paths.** K = 2N hypotheses per question (`config.py:170`
  `GEN_MODES = ("support", "attack")` [VERIFIED]). K = 8 for 4 options, K = 6 for
  3, and K = 10 for 5.
- **One qubit per path**, i = 0…K−1. Ordering in this document and the script:
  (support₀, attack₀, support₁, attack₁, …). In the model, membership comes
  from `polarity` (B, K) and `align` (B, K, N), so the cache's ordering never
  matters.
- **Basis state** x ∈ {0,1}^K is one **world**. x_i = 1 means "path i holds",
  x_i = 0 means "path i fails". Integer index Σᵢ xᵢ 2^i.
- **Pauli convention.** Z_i|x⟩ = z_i|x⟩ with z_i = 1 − 2x_i. X_i flips bit i.
  Y_i|b⟩ = i(−1)^b|1−b⟩.
- **State space:** ℂ^(2^K). Every Hamiltonian here is real symmetric (checks 1,
  11), so all amplitudes and density-matrix entries are real.

---

## 2. Hamiltonians

$$H_1 = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j - \Gamma\sum_i X_i,
\qquad
H_2 = H_1 + \sum_{i<j} K_{ij} X_i X_j\ \ \big(+ \sum_{i<j} L_{ij} Y_i Y_j\big)$$

All are 2^K × 2^K real symmetric matrices.

| Term | Matrix entries [DERIVED; checks 1, 11] | Meaning |
|---|---|---|
| h_i Z_i, J_ij Z_iZ_j | diagonal: E(x) = Σ hᵢzᵢ + Σ J_ij zᵢzⱼ | the classical energy of world x (§2.1) |
| −Γ X_i | −Γ between worlds differing in bit i | lets each path be in superposition of holding and failing; Γ = softplus(γ) ≥ 0 |
| K_ij X_iX_j | +K_ij between worlds differing in **both** bits i and j | couples worlds in which two paths flip together; sign unconstrained |
| L_ij Y_iY_j (optional) | −L_ij z_i z_j between the same worlds: −L_ij if x_i = x_j, +L_ij if they differ | the same flip, with a sign that depends on the world; any L_ij ≠ 0 produces both signs |

**Sign conventions at Γ = 0** [DERIVED; check 2]:
- h_i > 0 favours x_i = 1.
- J_ij > 0 favours exactly one of i, j holding (exclusion).
- J_ij < 0 favours agreement: both hold *or* both fail.

### 2.1 Directed pair terms in the 0/1 basis (replaces the symmetric Z_iZ_j head)

A symmetric Z_iZ_j coupling *alone* cannot say "i requires j" in one
direction. The previous version of this document stated that such a term
was out of scope. That was too strong: a directed penalty is a quadratic
function of the bits, so it maps **exactly** onto fields plus couplings. The
head now produces terms in the 0/1 basis:

$$E_{01}(x) = \sum_i a_i x_i + \sum_{i<j} b_{ij}\, x_i x_j + \sum_{i\neq j} r_{ij}\, x_i(1-x_j),\qquad r_{ij}\ge 0$$

- a_i: unary cost of path i holding (a_i < 0 means plausible);
- b_ij: symmetric co-occurrence term (b > 0: exclusion; b < 0: agreement);
- **r_ij: "i requires j"**, a penalty when i holds and j fails; r_ij ≠ r_ji in
  general.

**Mapping to Ising form** [DERIVED; check 13]. Expand
x_i(1 − x_j) = x_i − x_i x_j, collect

$$c_i = a_i + \sum_{j\ne i} r_{ij},\qquad q_{ij} = b_{ij} - r_{ij} - r_{ji}\quad(i<j,\ q \text{ symmetric}),$$

and substitute x_i = (1 − z_i)/2:

$$h_i = -\tfrac12 c_i - \tfrac14\sum_{j\ne i} q_{ij},\qquad J_{ij} = \tfrac14 q_{ij},\qquad \text{const} = \tfrac12\sum_i c_i + \tfrac14\sum_{i<j} q_{ij}.$$

- The constant shifts H by a multiple of the identity and leaves every state
  unchanged.
- Check 13 reproduces E₀₁ for all 256 worlds to 1.1e-14 on 50 random draws.
- It keeps the direction: with only r₀₁ = 1, the world (x₀ = 1, x₁ = 0) costs
  1.00; with only r₁₀ = 1 it costs 0.00.

**What this does and does not add.** Every quadratic function of K bits is
an Ising energy and vice versa, so the directed head does **not** enlarge the
set of representable energies. It changes the *parametrization*: "i requires j"
becomes one non-negative number predicted from ordered features. With a
symmetric head, the same constraint needs a coordinated pair of h_i, h_j and
J_ij values. This is an inductive-bias claim, and the ablation tests it.

**Symmetric-only ablation:** the same head with r ≡ 0 (output removed), one
variable.

### 2.2 Heads and shapes (B = 16, d = MODEL_DIM = 128 [VERIFIED: `config.py:114`])

Inputs, all already computed in the v46 forward pass
(`models/full_model.py:257-408` [VERIFIED]):

| Tensor | Shape | Source |
|---|---|---|
| e (post-reasoning, post-circuit, post-feedback path embeddings) | (B, K, d) | `H` after `models/full_model.py:349` |
| validator energy per path | (B, K) | `validator_out["validator_energy"]` (`models/validator.py:163-170`) |
| path–option energy | (B, K, N) | `selector["energy"]` (`models/full_model.py:369`) |
| polarity | (B, K), ±1 | input |
| align | (B, K, N), one-hot over options | input |
| H_mask, O_mask | (B, K), (B, N) | input |

**Unary head → a.**
- Own-option energy α_i = Σₙ align_in · energy_in: (B, K, N) → (B, K).
- Features [e_i, validator_energy_i, α_i, polarity_i]: (B, K, 131).
- MLP 131 → 64 → 1 (GELU); a = −c_a·tanh(·): (B, K).
- The LLM prior is not a unary feature. It stays at the output, as in v46, and
  only in prior-on variants (§6).

**Directed pair head → r, b (and k for H2).**
- Ordered features f_ij = [e_i, e_j, e_i ⊙ e_j, pol_i, pol_j, same_option_ij],
  with same_option_ij = Σₙ align_in·align_jn: (B, K, K, 3d+3) = (B, K, K, 387).
  The order of e_i, e_j makes f_ij ≠ f_ji.
- MLP 387 → 64 → n_out (GELU), with n_out = 2 (H0/H1) or 3 (H2):
  (B, K, K, 387) → (B, K, K, n_out).
- **r_ij** = c_r·sigmoid(·) ≥ 0: (B, K, K), not symmetrized.
- **b_ij** = c_b·tanh(½(s_ij + s_ji)): symmetrized.
- **H2 only: K_ij** = c_K·tanh(½(t_ij + t_ji)), symmetrized, sign
  unconstrained. **H2s** (stoquastic capacity control):
  K_ij = −c_K·sigmoid(½(t_ij + t_ji)) ≤ 0.
- Zero the diagonals; zero every term that touches a masked path.
- **Map to Ising** with the formulas of §2.1: (a, b, r) → h (B, K),
  J (B, K, K).

**Scale.** c_a = c_b = c_r = c_K = 2 [ASSUMPTION; set on validation during
the CPU smoke test, never on test]. Bounded heads fix the overall energy
scale, which T would otherwise trade off against.

**Parameters** [DERIVED]:

| Variant | Count | vs v46's 1,056,382 [OBSERVED cell 22] |
|---|---|---|
| Unary head (all variants) | 131·64 + 64 + 64 + 1 = 8,513 | |
| Pair trunk (all variants) | 387·64 + 64 = 24,832 | |
| Symmetric-only (n_out = 1) | 33,410 + scalars | |
| H0/H1 directed (n_out = 2) | 33,475 + scalars | **+3.2%** |
| H2 directed (n_out = 3) | 33,540 + scalars | **+3.2%** |

Scalars: τ (all), plus γ (H1, H2). The H2 output layer adds 65 parameters
over H1.

### 2.3 Building H (shapes)

Constant buffers per K, built once:
- Z table: z (2^K, K)
- pair table: zz (K(K−1)/2, 2^K)
- transverse matrix: X_sum (2^K, 2^K)
- H2 only: the pair-flip index table, K(K−1)/2 index maps of length 2^K

Per batch:
1. E = h @ zᵀ + J_pairs @ zz: **(B, 2^K)**
2. H₁ = diag_embed(E) − Γ·X_sum: **(B, 2^K, 2^K)**
3. H₂ = H₁ + Σ_{i<j} K_ij·XX_ij, as a scatter-add of K_ij into the pair-flip
   entries: **(B, 2^K, 2^K)**

**Mixed option counts:** group items by K (6, 8, 10) and build one H per group,
instead of padding with frozen qubits.

---

## 3. Where the classical data enters the quantum state

**Explicitly: per-question classical data enters the quantum state here and
only here.** It enters through three things:
1. the unary terms (a → h);
2. the pair terms (b, r → J and h; K for H2), all computed from that
   question's path embeddings, validator energies, path–option energies,
   polarities and option alignment;
3. the **basis encoding of paths**: qubit i *is* path i of this question, and
   basis state x *is* a world over its paths.

The state is defined by the data, as the thermal state of H(data). This is
Hamiltonian encoding.

**Why no separate dataset-encoding stage** (no AngleEmbedding, amplitude
encoding or feature map in front of it):
- **No initial state to load into.** The state is a function of a
  data-dependent operator, not the output of a circuit applied to an input
  state.
- **The register is the wrong shape for features.** 8 qubits carry one bit of
  meaning per path. The 1,024 numbers of 8 × 128-d embeddings have no
  interpretable home in a 256-dimensional state, and encoding them would destroy
  the "world" meaning of the basis.
- **One channel, interpretable numbers.** Every term in H has a stated meaning
  (plausibility, exclusion, requirement, interference coupling), and §7 reads
  them directly. A second encoding channel would make "paths interfered"
  untestable.
- **The per-path quantum feature map already exists** upstream (the v46
  circuit). Another one would duplicate it and add a pipeline stage (CLAUDE.md
  "ask before").

---

## 4. Collapse, stoquasticity and interference

### 4.1 Thermal-state readout at learned T — **accepted (P-2, 2026-10-18)**

$$P_T(x) = \frac{\langle x|e^{-H/T}|x\rangle}{\mathrm{Tr}\,e^{-H/T}},\qquad T = 0.05 + \mathrm{softplus}(\tau),\ T_0 = 1.0\ [\text{ASSUMPTION}]$$

This readout is used for **H0, H1, H2 and H2s**.
- **H0 (Γ ≡ 0, K ≡ 0):** P_T = softmax(−E/T) exactly (check 7, max difference
  0.0). The gradient is non-zero (check 6, 50/50). This is the classical Ising
  model over worlds.
- **As T → 0:** P_T → the ground-state readout ψ₀² (check 7: total-variation
  distance 0.244 at T = 1, 2.7e-16 at T = 0.01).
- **Why not a ground-state readout for the pair:** at Γ = 0 the ground state is
  the one-hot argmin world, and the loss has zero gradient almost everywhere
  (check 6, 50/50). A ground-state H0 cannot be trained, so H0/H1 would not be
  "otherwise identical".
- **Computation.** H0: softmax(−E/T), (B, 2^K). H1, H2:
  `torch.linalg.matrix_exp(−(H − c·I)/T)` in float64, then the diagonal,
  normalized.
  - c is a detached ground energy from `torch.linalg.eigvalsh`. P is invariant
    to c, so every exponent is ≤ 0 with maximum 0. The cheaper Gershgorin bound
    c = minₓE(x) − KΓ (minus Σ|K_ij| for H2) can underflow at small T and is
    only a fallback.
  - The backward is the Fréchet derivative (Daleckii–Krein divided
    differences). It is smooth even with degenerate excited states, and
    matches finite differences to 1.8e-9 for H1 (check 7) and 3.6e-9 for H2
    (check 14).
- **H1-GS** (secondary): the ground-state readout ψ₀² with Γ > 0, as originally
  proposed. Degeneracy handling is in §4.4.

### 4.2 Why H1 cannot interfere: stoquasticity and Perron–Frobenius

**Definition** [VERIFIED: Bravyi, DiVincenzo, Oliveira, Terhal 2008, abstract].
A Hamiltonian is *stoquastic* in a basis if all its off-diagonal matrix elements
in that basis are real and non-positive. Equivalently, its Gibbs density matrix
is entrywise non-negative at every temperature.

**H1 is stoquastic in the world basis.** Its only off-diagonal entries are −Γ ≤ 0
(check 1). Two consequences:
1. **Thermal state.** With c ≥ maxₓ E(x), the matrix M = cI − H₁ is entrywise
   non-negative, and
   e^{−H₁/T} = e^{−c/T}·e^{M/T} = e^{−c/T} Σₙ (M/T)ⁿ/n!.
   Every term is entrywise ≥ 0 [DERIVED; check 12: no negative entry in
   200/200 draws].
   - Read as a sum over *sequences of worlds* (each power of M steps from one
     world to a neighbouring one), the diagonal P_T(x) adds up only
     non-negative contributions.
   - **No sequence can cancel another.** There is no destructive interference
     between paths through world space.
2. **Ground state** (Perron–Frobenius, a standard theorem for non-negative
   irreducible matrices). M is non-negative and, for Γ > 0, irreducible: the
   hypercube of single-bit flips is connected. So the top eigenvector of M,
   which is the ground state of H₁, is unique and strictly positive (check 4:
   200/200; check 12: H1 ground states with a negative amplitude 0/200).

So **H1 adds superposition but no interference**: its readout equals that of
its own sign-stripped version (check 12: total-variation distance 0.0 in every
draw). Any H1-vs-H0 difference is a superposition (transverse-field) effect,
not an interference effect.

### 4.3 Why H2 can interfere

- K_ij X_iX_j puts **+K_ij** on the entries joining worlds that differ in bits i
  and j. With K_ij > 0 these entries are positive, so H₂ is **non-stoquastic in
  the world basis**. The optional Y_iY_j term has entries of both signs for any
  L_ij ≠ 0.
- Then M = cI − H₂ has negative entries. Sequences of worlds through those
  entries contribute **negative** terms to P_T(x), so contributions cancel:
  destructive interference between paths.
- Measured on random draws (K = 8, T = 0.5) [DERIVED; check 12]:

  | | H1 | H2 (unconstrained K) | H2s (K ≤ 0) |
  |---|---|---|---|
  | ground state has negative amplitudes | 0/200 | **189/200** | 0/200 |
  | negative entries in e^{−H/T} | 0/200 | **190/200** | 0/200 |

  - H2's negative amplitude mass: median 0.128, max 0.499.
  - H2's readout differs from its sign-stripped version by a median
    total-variation distance of 0.519.
  - These are **random**, untrained draws. They show what H2 *can* do, not
    what training will choose.
- **H2s is a capacity control.** It has the same head and the same XX terms,
  with the sign constrained to K ≤ 0. That keeps it stoquastic: it has every
  parameter H2 has, but cannot interfere. H2 vs H2s isolates the sign from the
  extra couplings.

**Basis caveat** [VERIFIED reference]. "Non-stoquastic" is a property of a
basis. Some Hamiltonians that are non-stoquastic in one basis become
stoquastic after a local change of basis, and finding such a "curing"
transformation is NP-complete even when restricted to single-qubit Clifford or
orthogonal transformations (Marvian, Lidar & Hen 2019).
- Our claim is deliberately basis-specific. The answer is **read in the world
  basis**, so the sign structure that matters for the readout is the one in
  that basis. The §7 diagnostics measure exactly that.
- We do not claim that a learned H2 is "intrinsically" non-stoquastic.

### 4.4 Near-degenerate ground states (H1-GS only)

- Γ > 0 makes the H1 ground state unique (check 4), but the gap can be tiny. In
  a flip-symmetric case (h = 0) the gap is 0, 7.9e-8, 7.6e-4 and 1.04 at
  Γ = 0, 0.01, 0.1 and 1 (check 10). Eigenvector gradients scale as 1/gap.
- At Γ = 0, `eigh` returns an arbitrary vector inside a degenerate subspace.
  Reading P from one column depends on that choice (max difference 0.415); the
  projector average does not (check 10).
- **Handling:**
  1. compute in float64;
  2. floor Γ at 0.05 [ASSUMPTION];
  3. broadened backward: (λ₀ − λ_k)/((λ₀ − λ_k)² + δ²), δ = 1e-3 [ASSUMPTION];
  4. if λ₁ − λ₀ < δ, use the projector average over eigenvectors within δ of
     λ₀;
  5. log how often 3 or 4 fires.
- The thermal arms (H0, H1, H2, H2s) need none of this (§4.1).

---

## 5. Answer and loss

- **Answer masks** (B, N, 2^K):
  m_n(x) = Π_{k: align_kn = 1, pol_k = +1} x_k · Π_{k: align_kn = 1, pol_k = −1} (1 − x_k),
  i.e. x_sup(n)(1 − x_att(n)) for one support and one attack per option.
- **Score:** s_n = Σₓ P_T(x) m_n(x): (B, N), in [0, 1] (check 8). It is the
  probability that option n's support holds and its attack fails.
- **Normalize:** p_n = (s_n + ε)/Σ_m(s_m + ε), ε = 1e-6. The raw s_n summed to
  1 in only 3 of 200 draws (check 8). Padded options use v46's −30 floor
  (`models/full_model.py:428-440`).
- **Prior-on variants only:** add softplus(llm_prior_weight)·z(llm_logprob) to
  log p and re-normalize, as v46 does (`models/full_model.py:416-440`).
- **Loss.** `scores = log p` goes into `training/losses.py` unchanged:

  smoothed cross-entropy + 0.4 · ranking + 0.1 · validator BCE.

  The path arms return `diversity = None`, so the diversity reward is skipped
  (`training/losses.py:139-145` [VERIFIED]). It is identical across H0, H1 and
  H2.

**Differentiable end to end.**
- loss → p → s → P_T: smooth for ε > 0.
- P_T → H:
  - H0: softmax, smooth.
  - H1, H2: matrix_exp and trace, Fréchet-differentiable for every symmetric H
    (checks 7, 14).
  - H1-GS: eigh, differentiable for a non-degenerate ground state (check 5:
    1.1e-8 vs finite differences).
- H → (h, J, Γ, K): linear in h, J and K; softplus in Γ.
- (h, J) → (a, b, r): linear (§2.1).
- Heads → e, validator energy, path–option energy → reasoner, circuit,
  encoder projection: the existing v46 graph.

The Tier 0 gradient-coverage test (DoD §2.2) must pass for every path arm.

---

## 6. Arms and pre-registration

| Arm | Γ | K (XX) | Pair head | Role |
|---|---|---|---|---|
| **H0** | ≡ 0 | ≡ 0 | directed (r, b) | classical Ising over worlds (Gibbs) |
| **H1** | learned | ≡ 0 | directed | superposition, stoquastic: **no interference** |
| **H2** | learned | learned, sign unconstrained | directed + k | superposition **with** sign structure |
| H2s | learned | learned, K ≤ 0 | directed + k | capacity control for H2 (stoquastic) |
| H1-sym | learned | ≡ 0 | r ≡ 0 (symmetric only) | ablation of the directed terms |
| H1-GS | learned, ≥ 0.05 | ≡ 0 | directed | ground-state readout (secondary) |

- All arms use the thermal readout at learned T (except H1-GS) and the A3 base
  config (LLM prior off; `training/ablations.py:109` [VERIFIED]).
- Prior-on variants only if a primary contrast is positive (decision P-3).
- **Protocol for every comparison:** full ARC-Easy; seeds 42/43/44; checkpoint
  selected on validation; test evaluated once; per-seed exact McNemar plus an
  item-level paired bootstrap of the seed-averaged difference
  (improvement_plan.md §0).
- Initial values: Γ₀ = 0.5, T₀ = 1.0, K = 0 at init (zero-initialized last
  layer of the k output) [ASSUMPTION]. γ and τ are in the T2c scalar LR group.

**Contrasts:**

| Contrast | Single variable | Status |
|---|---|---|
| **H1 vs H0** | Γ | primary, claim Q (α = 0.05) |
| **H2 vs H1** | K term | primary, claim I (α = 0.05) |
| H2 vs H2s | sign of K | attribution for claim I (required, see below) |
| H1 vs H1-sym | directed terms r | ablation (Holm family) |
| H0, H1, H2 vs v46 A3 (Tier 0′); vs LLM cloze; vs LLM letter pick | — | secondary (Holm family) |
| H1-GS vs H1 | readout | secondary |

### Pre-registration (fixed now, before any run)

*Thresholds marked † are choices fixed here [ASSUMPTION], not derived.*

**Claim Q — "superposition between paths helps."** Supported only if **all**
of these hold:
1. H1 − H0 > 0 on test, with the paired 95% CI excluding 0, and H1 ahead in
   all 3 seeds.
2. Γ stays in use: Γ/(‖J‖_F/K) ≥ 0.05† at the selected checkpoint in all
   seeds.
3. The ground state of the learned H is entangled across supports | attacks:
   median > 0.01† nats on validation.

Falsified if Γ collapses (condition 2 fails in ≥ 2 seeds), or if H1 − H0 ≤ 0 or
its CI includes 0 while the minimum detectable difference for the observed
discordance is ≤ 3 points. Inconclusive if the CI includes 0 and the test is
under-powered. Prediction [HYPOTHESIS]: on a gate-passing cache, H1 ≥ H0 by
< 2 points.

**Claim I — "interference between paths helps"** (the H2 question).
- **Supported** only if **all** of these hold:
  1. **H2 > H1**: H2 − H1 > 0 on test, paired 95% CI excluding 0, H2 ahead in
     all 3 seeds.
  2. **Attribution to sign:** H2 − H2s > 0 with a paired 95% CI excluding 0.
     Otherwise any gain comes from the extra couplings, not from interference.
  3. **Sign structure is in use** on validation at the selected checkpoint, in
     all seeds:
     - median negative-amplitude mass of the learned H's ground state ≥ 0.01†;
     - median total-variation distance between P_T and the sign-stripped
       readout ≥ 0.01†.
- **Falsified (H2 ≈ H1):**
  - H2 − H1 ≤ 0, or its CI includes 0 with a minimum detectable difference
    ≤ 3 points; **or**
  - sign structure unused (condition 3 fails in ≥ 2 seeds; training chose a
    stoquastic-looking H2); **or**
  - H2 > H1 but H2 ≈ H2s: the gain is capacity, not interference.
- **Inconclusive:** the CI includes 0 and the test is under-powered.
- **Prediction** [HYPOTHESIS]: on a gate-passing cache, H2 ≈ H1 (|Δ| < 2
  points). With one support and one attack per option, the useful pairwise
  logic is mostly exclusion and requirement, which the diagonal (h, J, r)
  already expresses.

**Directed terms (ablation, no claim attached).** H1 − H1-sym is reported with
its CI; no direction is predicted.

**What a positive result would and would not mean.**
- For H1 [VERIFIED references, §13]: H1 is a transverse-field Ising model, and
  for such models Suzuki (1976) maps the quantum system onto a classical Ising
  model with one extra dimension. Stoquastic Hamiltonians have no sign problem
  for such Monte Carlo methods, but that does **not** make them efficiently
  simulable in general:
  - estimating stoquastic ground-state energies is MA-complete under a guiding
    promise, and poly-time only for special cases such as the *ferromagnetic*
    transverse-field Ising model (Bravyi 2015). Our learned J has both signs,
    so that case does not apply;
  - quantum Monte Carlo can fail to equilibrate in polynomial time even with no
    sign problem (Hastings 2013).
- At K ≤ 10 none of this matters computationally: everything is computed
  exactly. A positive H1 result shows that this transverse-field relaxation of
  the classical collapse helps on ARC-Easy.
- A positive H2 result shows that sign structure in the world basis helps.
- Neither is a quantum advantage, and neither may be described as one.

---

## 7. Diagnostics (logged per epoch on validation; final on test)

| Diagnostic | Definition | Tells |
|---|---|---|
| Γ, T | learned scalars; Γ/(‖J‖_F/K) | superposition in use (Q-2) |
| ‖J‖_F, ‖K‖_F | means over items | pairwise structure learned at all |
| J, r, K by pair type | mean and sign fraction for (support_n, attack_n), (support_n, support_m), (attack_n, attack_m), (support_n, attack_m) | expected: b > 0 or J > 0 between the support and attack of the same option; r structure between options; random signs = nothing learned |
| directed asymmetry | median of ‖r − rᵀ‖_F / ‖r + rᵀ‖_F | whether the directed head uses direction at all |
| **negative-amplitude mass** (H2, H2s, H1) | ground state ψ₀ of the learned H with its global sign fixed so the majority mass is positive; f_neg = Σ_{x: ψ₀(x)<0} ψ₀(x)² ∈ [0, 0.5] | sign structure in the computational (world) basis (I-3); must be 0 for H1 and H2s (check 12) |
| negative thermal weight (H2) | Σ_{x≠x′} max(−ρ_xx′, 0) / Σ_{x≠x′} \|ρ_xx′\| for ρ = e^{−H/T}/Tr | sign structure in the actual readout state |
| sign-stripped TV (H2) | ½ Σₓ \|P_T(x) − P_T⁺(x)\|, with P_T⁺ from H with off-diagonals → −\|·\| | how much the signs change the readout (I-3); exactly 0 for H1 (check 12) |
| ground-state gap | λ₁ − λ₀: min, median, fraction < δ | degeneracy; H1-GS handling |
| entanglement entropy | von Neumann entropy of ψ₀ across (i) supports \| attacks, (ii) options {0…⌈N/2⌉−1} \| rest; max 4 ln 2 = 2.77 nats at 4 \| 4 | entanglement between paths (Q-3); 0 = product state (check 9) |
| ground-state/answer agreement | MAP world x* = argmax P_T(x): fraction where x* switches on exactly one option, and where that option equals the prediction | whether the collapse picks a coherent world |
| world entropy | −Σₓ P_T log P_T / (K ln 2) | collapse selectivity (the analogue of C6) |
| arm disagreements | items where H1 ≠ H0 and H2 ≠ H1, per seed | the discordant counts behind McNemar power |

---

## 8. Cost

**Measured** [VERIFIED: `scripts/verify_path_hamiltonian.py --bench`, Apple M4
CPU, numpy 2.5.4]: `numpy.linalg.eigh` forward on 16 matrices takes **47 ms at
K = 8** (256 × 256) and **1.15 s at K = 10** (1024 × 1024).

**Memory** [DERIVED]: a batch of 16 H matrices is 4.2 MB at K = 8 (fp32), and
67 MB (fp32) or 134 MB (fp64) at K = 10. H2 has the same dense size.

**T4 time per training step, batch 16** [ASSUMPTION; to be measured]:

| | K = 8 | K = 10 |
|---|---|---|
| H0 (softmax over worlds) | < 1 ms | ~1 ms |
| H1, H2, H2s thermal (eigvalsh shift + matrix_exp fwd + bwd, fp64) | ~20–120 ms | ~0.5–4 s |
| H1-GS (eigh fwd + bwd, fp64; T4 fp64 ≈ 1/32 of fp32) | ~20–100 ms | ~0.5–3 s |

- Directed head: the (B, K, K, 387) features are 16·64·387 ≈ 0.4 M floats at
  K = 8. Negligible.
- Against v46's assumed 1–3 s/step, K = 8 adds little. K = 10 batches may double
  in time; the share of 5-option items comes from the Tier 0 cache histogram.
- Fallback: a Krylov `expm_multiply` for the needed diagonal, or batched power
  iteration [ASSUMPTION: only if measurement requires it].

**CPU micro-benchmark plan** (synthetic H, before any T4 run):
1. torch CPU, fp32 and fp64: `matrix_exp`, `eigvalsh` and `eigh`, forward +
   backward with the §5 loss, batch 16, K = 6, 8, 10, for H1 and H2; 3 warm-ups,
   20 repetitions; median and IQR.
2. `torch.autograd.gradcheck` (fp64, K = 4) for H0, H1, H2, H1-GS; compare
   with this script's analytic gradients.
3. Repeat (1) on a T4 in the E8 smoke run and record it in the manifest.
4. Full-model step time per arm on 20 questions, vs v46 A3.

---

## 9. Integration (planned; implemented only after approval)

| File | Change | v46 path affected? |
|---|---|---|
| `models/path_hamiltonian.py` (new) | `PathHamiltonian(nn.Module)`: unary head; directed pair head with outputs (r, b[, k]); the (a, b, r) → (h, J) mapping; constant Z/ZZ/X/pair-flip buffers per K; grouping by K. Modes: `transverse ∈ {off, learned}`, `pair_flip ∈ {off, free, nonpositive}`, `directed ∈ {on, off}`, `readout ∈ {thermal, ground}`. Returns `scores` (B, N), per-group `world_probs` (B, 2^K) and the §7 diagnostics | — |
| `models/full_model.py` | flag `readout ∈ {"interference", "path"}` (default `"interference"`) plus a `path_cfg` dict. With `"path"`, the unused v46 readout modules (`CoherentCollapse`, `CollapseController`, `EnergyFusion`, `polarity_phase`, `phase_head`) are not built, so no new dead parameters appear (A1/A2). The forward branches after the selector (`models/full_model.py:364`); the v46 branch stays byte-identical | No |
| `training/ablations.py` | arms `H0_path_ising`, `H1_path_tfim`, `H2_path_signed`, `H2s_path_nonpos`, `H1sym_path_tfim`, `H1GS_path_ground` (A3 + path config); `DEFAULTS` gains `readout="interference"` | No |
| `training/evaluate.py` | log the §7 diagnostics when path keys are present; read v46-only keys with `.get` (`training/evaluate.py:143-170` [VERIFIED]) | No |

- **4 files: over the 3-file threshold, so this needs approval.**
- Path-arm checkpoints carry their config (A3/A5 fixes); v46 checkpoints are
  unaffected.
- CPU tests:
  - checks 1–14 ported to torch;
  - gradient coverage per arm;
  - bitwise "v46 arms unchanged" before vs after.

---

## 10. Prerequisites: Tier 0 and Tier 1

- **Tier 0** (improvement_plan.md) is needed for anything to be measurable:
  - no cross-seed resume (A3/A8);
  - the A3 reference and the letter-pick and majority-vote baselines on the
    same items;
  - item-level statistics (B6);
  - the gradient-coverage test (it catches dead heads in the new arms).
- **Tier 1, and why D1 blocks this design.**
  - Every energy term (h, J, r, K) is a learned function of the path
    embeddings. If the hypotheses carry no information about the answer beyond
    the question and options (D1; v46 gate at chance [OBSERVED cell 14]), then by
    the data-processing inequality (dossier D1) none of these terms can either.
  - The distribution over worlds is then unrelated to the answer, whatever Γ
    or K is.
  - Pairwise terms need *more* signal than unary ones: whether two paths
    exclude, require or interfere with each other is a property of the pair,
    learned from 2,251 questions × 28 pairs (K = 8).
  - On a no-signal cache, H0 ≈ H1 ≈ H2 for reasons unrelated to superposition
    or interference. That would be misreported as evidence against Q and I.
  - So **these arms run only on a cache that passed the Tier 1 gate.**
- **Recommended:** T2a (A7) first, because the unary head reads the validator
  energy. T2c for γ and τ.

---

## 11. Risks

| Risk | Mechanism | Mitigation / report |
|---|---|---|
| Degenerate ground states (H1-GS) | gap → 0; gradients ∝ 1/gap (check 10) | thermal readout for the primary arms; §4.4 for H1-GS; log the gap |
| J, r, K not learning without signal | zero pair terms are optimal when paths are uninformative; the pair trunk has 24,832 parameters | Tier 1 gate; log norms and per-type means; report "no pairwise structure" if they stay at init |
| Γ collapses to 0 | the classical model fits better | a pre-registered falsifier of Q; H1 has no floor |
| H2 learns K ≤ 0 (stoquastic) | sign structure not useful | a pre-registered falsifier of I (I-3); reported, not prevented |
| H2 sign structure removable by a basis change | basis-dependence of stoquasticity (Marvian et al. 2019) | the claim is explicitly world-basis-specific (§4.3) |
| H2 gain from capacity, not sign | XX couplings add flexibility | H2s control (same capacity, K ≤ 0) is required for claim I |
| Overfitting the pair head | ~33.5 k new parameters; ordered 387-d features | v46 weight decay; train vs validation accuracy logged (E6); optional 32-unit pair MLP as a pre-declared variant chosen on validation |
| Directed terms add no expressivity | quadratic functions = Ising energies (§2.1) | stated; H1 vs H1-sym tests inductive bias only |
| Saturated collapse | low T or large energies make P one-hot (check 6: smallest Gibbs gradient 6.1e-9) | T floor 0.05; bounded heads; log world entropy |
| Over-claiming | exact classical simulation; stoquastic or basis-specific signs | §4.2–4.3, §6 last paragraph; never "quantum advantage" |
| Cost at K = 10 | 1024³ operations | group by K; measure (§8); Krylov fallback |

---

## 12. Plain-language summary for my supervisor

The original qAIR idea is that reasoning paths should interact: one argument
can rule another out, require another, or cancel another. The answer should
come from the most consistent combination of arguments. The capstone never
did this: its quantum circuit looked at each argument alone. This design gives
each of the 8 arguments for a question one qubit, so the quantum state covers
every combination of arguments being true or false (256 "worlds"). The model
learns how plausible each argument is and how pairs relate, including
one-directional "this argument only stands if that one does" links. The answer
is read from the state.

We test three versions that differ in one switch at a time:
- a classical model over the same worlds;
- a quantum version with superposition but, provably, no cancellation between
  paths;
- a quantum version whose amplitudes can carry signs, so that paths can cancel
  each other.

For each step we fix in advance which result supports the idea and which
refutes it, including a control that rules out "it only helped because it had
more parameters". Everything is simulated exactly on a normal GPU, so even
positive results are not "quantum advantage", and we will say so. It only
makes sense to run once the generated arguments carry real information about
the answer, so it comes after the trustworthiness and evidence-quality work
already planned.

---

## 13. References (each checked this session against its arXiv or publisher page)

1. S. Bravyi, D. P. DiVincenzo, R. I. Oliveira, B. M. Terhal, "The Complexity
   of Stoquastic Local Hamiltonian Problems," *Quantum Information &
   Computation* **8**(5), 361–385 (2008); arXiv:quant-ph/0606140. Used for:
   the definition (off-diagonal elements real and non-positive in the standard
   basis), the connection to Perron–Frobenius, and the entrywise non-negative
   Gibbs density matrix.
2. S. Bravyi, "Monte Carlo simulation of stoquastic Hamiltonians," *Quantum
   Information & Computation* **15**(13/14), 1122–1140 (2015); arXiv:1402.2295.
   Used for: the guided stoquastic ground-energy problem is MA-complete; the
   *ferromagnetic* transverse-field Ising model is classically simulable in
   polynomial time.
3. M. B. Hastings (appendix by M. H. Freedman), "Obstructions to classically
   simulating the quantum adiabatic algorithm," *Quantum Information &
   Computation* **13**, 1038 (2013); arXiv:1302.5733. Used for: quantum Monte
   Carlo can fail to equilibrate in polynomial time even for systems with no
   sign problem, such as the transverse-field Ising model. (The abstract says
   "no sign problem"; it does not use the word "stoquastic".)
4. M. Suzuki, "Relationship between d-Dimensional Quantal Spin Systems and
   (d+1)-Dimensional Ising Systems: Equivalence, Critical Exponents and
   Systematic Approximants of the Partition Function and Spin Correlations,"
   *Progress of Theoretical Physics* **56**(5), 1454–1469 (1976);
   doi:10.1143/PTP.56.1454. Used for: the quantum-to-classical (d+1)
   mapping of the transverse-field Ising model. Verified from the abstract
   only.
5. M. Marvian, D. A. Lidar, I. Hen, "On the computational complexity of curing
   non-stoquastic Hamiltonians," *Nature Communications* **10**, 1571 (2019);
   doi:10.1038/s41467-019-09501-6; arXiv:1802.03408 (preprint title "On the
   Computational Complexity of Curing the Sign Problem"). Used for: finding a
   curing single-qubit Clifford or orthogonal transformation is NP-complete.

**Not cited from a source:** the Perron–Frobenius theorem (standard linear
algebra; checked numerically in checks 4 and 12). **Withdrawn** from the
previous version of this document: "ground and thermal states of stoquastic
Hamiltonians are, in general, classically samplable by path-integral Monte
Carlo". References 2 and 3 show that this is false as a general statement.

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

=== 11. H2 = H1 + K XX (+ L YY) is Hermitian with the stated entries ===
        K=6, 10 draws; YY entry = -1 where x_i = x_j, +1 where they differ
[PASS] H2 == H2^T, real (XX and YY terms included)
        claim=True   control(wrong)=False  
[PASS] <x xor e_i xor e_j| X_i X_j |x> = +1 for every x
        claim=True   control(wrong)=False  
[PASS] <x xor e_i xor e_j| Y_i Y_j |x> = -z_i z_j (both signs occur)
        claim=True   control(wrong)=False  

=== 12. Sign structure: H1 never, H2 sometimes, H2s (K <= 0) never ===
        K=8, 200 draws, T=0.5: ground state with negative amplitudes H1 0/200, H2 189/200, H2s 0/200
        negative entries in exp(-H/T): H1 0/200, H2 190/200, H2s 0/200
        H2 negative amplitude mass: median 0.128, max 0.499
        TV(P, P of sign-stripped H): H1 max 0.0e+00; H2 median 0.519, max 0.787
[PASS] H1 (stoquastic): no ground state has a negative amplitude
        claim=True   control(wrong)=False  
[PASS] H2 (unconstrained K): some ground states have negative amplitudes
        claim=True   control(wrong)=False  
[PASS] H2s (K <= 0, stoquastic capacity control): no negative amplitudes
        claim=True   control(wrong)=False  
[PASS] exp(-H/T) entrywise >= 0 for H1 and H2s; negative entries occur for H2
        claim=True   control(wrong)=False  
[PASS] signs change the readout for H2 only (sign-stripped TV: H1 = 0, H2 > 0)
        claim=True   control(wrong)=False  

=== 13. Directed 0/1-basis terms map exactly to Ising form ===
        K=8, 50 random (a, b, r), all 256 worlds: max |E01 - E_ising - const| = 1.1e-14
        world (x0=1, x1=0): penalty 1.00 under r_01 = 1, 0.00 under r_10 = 1
[PASS] x_i(1 - x_j) terms reproduced exactly for every world
        claim=True   control(wrong)=False  
[PASS] the mapped Ising form keeps the direction (r_01 != r_10)
        claim=True   control(wrong)=False  

=== 14. Thermal-collapse gradient for H2 matches finite differences (K = 4) ===
        5 draws; parameters: 4 h, 6 J, 1 a, 6 K_xx; worst relative error 3.58e-09
[PASS] H2 thermal gradient (Daleckii-Krein) == central finite differences (rel err < 1e-5)
        claim=True   control(wrong)=False  

=== Benchmark (informational): numpy.linalg.eigh, batch 16, this CPU ===
        K= 8 (256x256) float64:     47.7 ms per batch of 16 (forward only)
        K= 8 (256x256) float32:     47.6 ms per batch of 16 (forward only)
        K=10 (1024x1024) float64:   1145.3 ms per batch of 16 (forward only)
        K=10 (1024x1024) float32:   1149.5 ms per batch of 16 (forward only)

SUMMARY: 33/33 checks PASS (each with its wrong-claim control failing as required)
```

## Appendix B — decisions this design needs

| # | Decision | Status / recommendation |
|---|---|---|
| P-1 | Approve the path arms H0/H1 (claim Q) | Recommend approval, after the Tier 1 gate |
| P-2 | Thermal-state readout at learned T for H0 and H1; H1-GS kept as secondary | **Accepted 2026-10-18** |
| P-3 | Prior-on variants for a deployed-system comparison | Only if a primary contrast is positive |
| P-4 | Approve the 4-file integration (§9) | Recommend approval, behind a default-off flag |
| P-5 | Keep the per-hypothesis 6-qubit circuit upstream (rule 6) | Keep; removal only as a separate arm |
| P-6 | Approve H2 (non-stoquastic XX couplings) with the H2s capacity control (claim I); XX only, or XX + YY | Recommend XX only (one sign-controlled term, so H2s is well defined); YY as a later variant |
| P-7 | Directed (a, b, r) pair head as the default, with the symmetric-only H1-sym ablation | Recommend approval |
