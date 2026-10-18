"""
Numerical checks for docs/baseline/path_hamiltonian_design.md.

CPU only, numpy + the standard library; no torch, no model, no data. As in
scripts/verify_dossier_math.py, each check evaluates the design's claim (must
PASS) and a deliberately wrong "control" claim (must FAIL), so a check that
accepts everything cannot slip through.

Conventions (identical to the design document):
  * K paths, basis state x in {0,1}^K, integer index sum_i x_i 2^i.
  * x_i = 1 means "path i holds"; Z_i |x> = z_i |x> with z_i = 1 - 2 x_i.
  * H1 = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j - Gamma sum_i X_i (stoquastic).
  * H2 = H1 + sum_{i<j} K_ij X_i X_j (+ L_ij Y_i Y_j), signs unconstrained
    (non-stoquastic in the world basis whenever some K_ij > 0 or L_ij != 0).
  * Directed 0/1-basis energy E01(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j
    + sum_{i != j} r_ij x_i (1 - x_j), mapped exactly to Ising (h, J, const).
  * Paths are ordered (support_0, attack_0, support_1, attack_1, ...), so
    option n has support 2n and attack 2n + 1.
  * score_n = sum_x P(x) x_sup(n) (1 - x_att(n)); p_n = (score_n + eps) / sum.

    python scripts/verify_path_hamiltonian.py          # exit code 1 on failure
    python scripts/verify_path_hamiltonian.py --bench  # also time numpy eigh
"""

import math
import sys
import time

import numpy as np

EPS = 1e-6
RNG = np.random.default_rng(0)
FAILURES = []
CHECKS = []


def report(name, claim, control, detail=""):
    ok = bool(claim) and not bool(control)
    CHECKS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        claim={bool(claim)}   control(wrong)={bool(control)}  {detail}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def bits(K):
    """(2^K, K) array: bits(K)[x, i] = x_i."""
    idx = np.arange(2 ** K)
    return (idx[:, None] >> np.arange(K)[None, :]) & 1


def ising_energy(h, J, K):
    """Diagonal of H: E(x) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j, shape (2^K,)."""
    z = 1 - 2 * bits(K)
    pair = np.einsum("xi,ij,xj->x", z, np.triu(J, 1), z)
    return z @ h + pair


def x_sum(K):
    """sum_i X_i as a dense (2^K, 2^K) matrix: X_i flips bit i."""
    D = 2 ** K
    M = np.zeros((D, D))
    for x in range(D):
        for i in range(K):
            M[x, x ^ (1 << i)] = 1.0
    return M


def hamiltonian(h, J, gamma, K):
    return np.diag(ising_energy(h, J, K)) - gamma * x_sum(K)


def pair_flip(K, i, j, kind="XX"):
    """X_i X_j or Y_i Y_j as a dense real (2^K, 2^K) matrix.

    Both flip bits i and j. <x'|X_i X_j|x> = 1. Since Y|b> = i(-1)^b |1-b>,
    <x'|Y_i Y_j|x> = i^2 (-1)^(x_i + x_j) = -z_i z_j: real, sign depends on x.
    """
    D = 2 ** K
    b = bits(K)
    M = np.zeros((D, D))
    for x in range(D):
        y = x ^ (1 << i) ^ (1 << j)
        if kind == "XX":
            M[y, x] = 1.0
        else:
            M[y, x] = -(1 - 2 * b[x, i]) * (1 - 2 * b[x, j])
    return M


def hamiltonian2(h, J, gamma, Kxx, K, Lyy=None):
    """H2 = H1 + sum_{i<j} K_ij X_i X_j (+ L_ij Y_i Y_j)."""
    Hm = hamiltonian(h, J, gamma, K)
    for i in range(K):
        for j in range(i + 1, K):
            if Kxx[i, j] != 0:
                Hm = Hm + Kxx[i, j] * pair_flip(K, i, j, "XX")
            if Lyy is not None and Lyy[i, j] != 0:
                Hm = Hm + Lyy[i, j] * pair_flip(K, i, j, "YY")
    return Hm


def negative_mass(psi):
    """Amplitude mass on the minority sign, global sign fixed: in [0, 0.5]."""
    pos, neg = (psi[psi > 0] ** 2).sum(), (psi[psi < 0] ** 2).sum()
    return float(min(pos, neg))


def directed_to_ising(a, b, r):
    """Exact map of E01 = a.x + sum_{i<j} b_ij x_i x_j + sum_{i!=j} r_ij x_i(1-x_j)
    to sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const, with x = (1 - z)/2."""
    K = len(a)
    c = a + r.sum(axis=1)                       # r_ij x_i contributes to x_i
    q = np.triu(b, 1) - np.triu(r, 1) - np.triu(r.T, 1)
    q = q + q.T                                 # symmetric, zero diagonal
    h = -c / 2 - q.sum(axis=1) / 4
    J = q / 4
    const = c.sum() / 2 + np.triu(q, 1).sum() / 4
    return h, J, const


def answer_masks(K):
    """(N, 2^K): m_n(x) = x_sup(n) (1 - x_att(n))."""
    b = bits(K)
    N = K // 2
    return np.stack([b[:, 2 * n] * (1 - b[:, 2 * n + 1]) for n in range(N)])


def ground_state(Hm):
    lam, V = np.linalg.eigh(Hm)
    return lam, V, V[:, 0]


def answer_probs(P, K):
    s = answer_masks(K) @ P
    return s, (s + EPS) / (s + EPS).sum()


def loss_ground(h, J, gamma, K, y):
    _, _, psi = ground_state(hamiltonian(h, J, gamma, K))
    _, p = answer_probs(psi ** 2, K)
    return -math.log(p[y])


def softplus(a):
    return math.log1p(math.exp(a))


def sym(K):
    J = RNG.normal(size=(K, K))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    return J


def entanglement_entropy(psi, K, part_a):
    """Von Neumann entropy (nats) of psi across the bipartition part_a | rest."""
    part_b = [i for i in range(K) if i not in part_a]
    # numpy C-order: axis a of reshape([2]*K) is bit K-1-a.
    t = psi.reshape([2] * K)
    axes = [K - 1 - i for i in part_a] + [K - 1 - i for i in part_b]
    M = np.transpose(t, axes).reshape(2 ** len(part_a), 2 ** len(part_b))
    sv = np.linalg.svd(M, compute_uv=False)
    w = sv ** 2
    w = w[w > 1e-15]
    return float(-(w * np.log(w)).sum())


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_hermitian():
    print("\n=== 1. H is Hermitian (real symmetric) and has the stated entries ===")
    K = 8
    ok_sym, ok_diag, ok_off = True, True, True
    for _ in range(20):
        h, J, g = RNG.normal(size=K), sym(K), abs(RNG.normal())
        Hm = hamiltonian(h, J, g, K)
        ok_sym &= np.allclose(Hm, Hm.T, atol=0) and np.isrealobj(Hm)
        # brute-force diagonal
        z = 1 - 2 * bits(K)
        brute = np.array([z[x] @ h + sum(J[i, j] * z[x, i] * z[x, j]
                                         for i in range(K) for j in range(i + 1, K))
                          for x in range(2 ** K)])
        ok_diag &= np.allclose(np.diag(Hm), brute)
        off = Hm - np.diag(np.diag(Hm))
        ham1 = np.array([[bin(a ^ b).count("1") == 1 for b in range(2 ** K)]
                         for a in range(2 ** K)])
        ok_off &= np.allclose(off[ham1], -g) and np.allclose(off[~ham1], 0)
    # control: X applied on one side only (a non-Hermitian construction)
    h, J, g = RNG.normal(size=K), sym(K), 0.7
    bad = np.diag(ising_energy(h, J, K)) - g * np.triu(x_sum(K))
    print(f"        K=8, 256x256, 20 random draws")
    report("H == H^T, real", ok_sym, np.allclose(bad, bad.T))
    report("diag(H) equals the brute-force Ising energy E(x)", ok_diag, False)
    report("off-diagonal = -Gamma exactly on Hamming-distance-1 pairs, else 0",
           ok_off, False)


def check_conventions():
    print("\n=== 2. Sign conventions at Gamma = 0 (K = 2) ===")
    K = 2
    E_h = ising_energy(np.array([1.0, 0.0]), np.zeros((2, 2)), K)
    favoured_h = tuple(int(v) for v in bits(K)[np.argmin(E_h)])
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    E_J = ising_energy(np.zeros(2), J, K)
    minima_pos = {tuple(int(v) for v in bits(K)[x]) for x in np.flatnonzero(np.isclose(E_J, E_J.min()))}
    E_Jn = ising_energy(np.zeros(2), -J, K)
    minima_neg = {tuple(int(v) for v in bits(K)[x]) for x in np.flatnonzero(np.isclose(E_Jn, E_Jn.min()))}
    print(f"        h=(+1,0): minimum at x={favoured_h}; "
          f"J=+1 minima {sorted(minima_pos)}; J=-1 minima {sorted(minima_neg)}")
    report("h_i > 0 favours x_i = 1 (path holds)", favoured_h[0] == 1, favoured_h[0] == 0)
    report("J_ij > 0 favours exactly one of i, j holding (paths cancel)",
           minima_pos == {(1, 0), (0, 1)}, minima_pos == {(0, 0), (1, 1)})
    report("J_ij < 0 favours i, j agreeing: both hold OR both fail",
           minima_neg == {(0, 0), (1, 1)}, minima_neg == {(1, 1)})


def check_gamma0_ground_state():
    print("\n=== 3. At Gamma = 0 the ground state is the minimum-energy basis state ===")
    K, hits, ctrl = 8, 0, 0
    draws = 200
    for _ in range(draws):
        h, J = RNG.normal(size=K), sym(K)
        lam, V, psi = ground_state(hamiltonian(h, J, 0.0, K))
        E = ising_energy(h, J, K)
        x_star = int(np.argmin(E))
        hits += np.isclose(psi[x_star] ** 2, 1.0) and np.isclose(lam[0], E.min())
        ctrl += np.isclose(psi[int(np.argmax(E))] ** 2, 1.0)
    print(f"        K=8, {draws} random (h, J): one-hot at argmin E in {hits}/{draws}")
    report("ground state = |argmin_x E(x)>, ground energy = min E", hits == draws, ctrl == draws)


def check_perron_frobenius():
    print("\n=== 4. Gamma > 0: unique ground state with single-signed amplitudes ===")
    K, uniq, signed, min_gap = 8, 0, 0, np.inf
    draws = 200
    for _ in range(draws):
        h, J, g = RNG.normal(size=K), sym(K), 0.05 + abs(RNG.normal())
        lam, V, psi = ground_state(hamiltonian(h, J, g, K))
        uniq += lam[1] - lam[0] > 1e-9
        signed += np.all(psi > 0) or np.all(psi < 0)
        min_gap = min(min_gap, lam[1] - lam[0])
    print(f"        K=8, {draws} draws, Gamma in [0.05, ~3]: non-degenerate {uniq}/{draws}, "
          f"single-signed {signed}/{draws}, smallest gap {min_gap:.2e}")
    report("ground state non-degenerate and positive up to a global sign (stoquastic H)",
           uniq == draws and signed == draws, signed == 0)


def grad_eigh_analytic(h, J, a, K, y):
    """d loss / d (h, J_upper, a) by first-order eigenvector perturbation; Gamma = softplus(a)."""
    g = softplus(a)
    Hm = hamiltonian(h, J, g, K)
    lam, V = np.linalg.eigh(Hm)
    psi = V[:, 0]
    P = psi ** 2
    M = answer_masks(K)
    s = M @ P
    S = (s + EPS).sum()
    dL_ds = -np.eye(len(s))[y] / (s[y] + EPS) + 1.0 / S          # (N,)
    dL_dP = M.T @ dL_ds                                           # (2^K,)
    dL_dpsi = 2 * psi * dL_dP

    def directional(dH):
        # d psi0 = sum_{k>0} psi_k <psi_k| dH |psi0> / (lam0 - lam_k)
        coef = (V[:, 1:].T @ (dH @ psi)) / (lam[0] - lam[1:])
        return dL_dpsi @ (V[:, 1:] @ coef)

    z = 1 - 2 * bits(K)
    gh = np.array([directional(np.diag(z[:, i])) for i in range(K)])
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    gJ = np.array([directional(np.diag(z[:, i] * z[:, j])) for i, j in pairs])
    sig = 1 / (1 + math.exp(-a))
    ga = directional(-x_sum(K)) * sig
    return np.concatenate([gh, gJ, [ga]]), pairs


def grad_fd(loss, h, J, a, K, pairs, step=1e-6):
    out = []
    for i in range(K):
        e = np.zeros(K); e[i] = step
        out.append((loss(h + e, J, a) - loss(h - e, J, a)) / (2 * step))
    for i, j in pairs:
        E = np.zeros((K, K)); E[i, j] = E[j, i] = step
        out.append((loss(h, J + E, a) - loss(h, J - E, a)) / (2 * step))
    out.append((loss(h, J, a + step) - loss(h, J, a - step)) / (2 * step))
    return np.array(out)


def check_eigh_gradient():
    print("\n=== 5. Gradient through eigh matches finite differences (K = 4) ===")
    K, worst = 4, 0.0
    for trial in range(10):
        h, J, a, y = RNG.normal(size=K), sym(K), RNG.normal(), int(RNG.integers(K // 2))
        ga, pairs = grad_eigh_analytic(h, J, a, K, y)
        gf = grad_fd(lambda h_, J_, a_: loss_ground(h_, J_, softplus(a_), K, y),
                     h, J, a, K, pairs)
        worst = max(worst, np.max(np.abs(ga - gf)) / (np.max(np.abs(gf)) + 1e-12))
    # control: treat psi0 as constant (no eigenvector derivative) -> gradient 0
    h, J, a, y = RNG.normal(size=K), sym(K), 0.3, 0
    _, pairs = grad_eigh_analytic(h, J, a, K, y)
    gf = grad_fd(lambda h_, J_, a_: loss_ground(h_, J_, softplus(a_), K, y), h, J, a, K, pairs)
    print(f"        10 random draws; parameters: 4 h, 6 J, 1 a (Gamma = softplus(a)); "
          f"worst relative error {worst:.2e}")
    report("analytic eigh gradient == central finite differences (rel err < 1e-5)",
           worst < 1e-5, np.allclose(np.zeros_like(gf), gf, atol=1e-6))


def check_gamma0_untrainable():
    print("\n=== 6. Ground-state collapse at Gamma = 0 has zero gradient (H0 as specified) ===")
    K, zero_gs, nonzero_gibbs, min_gibbs = 4, 0, 0, np.inf
    draws = 50
    T = 0.5
    for _ in range(draws):
        h, J, y = RNG.normal(size=K), sym(K), int(RNG.integers(K // 2))
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        g_gs = grad_fd(lambda h_, J_, a_: loss_ground(h_, J_, 0.0, K, y), h, J, 0.0, K, pairs)[:-1]

        def loss_gibbs(h_, J_, a_):
            E = ising_energy(h_, J_, K)
            P = np.exp(-(E - E.min()) / T); P /= P.sum()
            return -math.log(answer_probs(P, K)[1][y])

        g_gb = grad_fd(loss_gibbs, h, J, 0.0, K, pairs)[:-1]
        zero_gs += np.max(np.abs(g_gs)) < 1e-6
        # Central-difference noise is ~1e-16 / 1e-6 = 1e-10, so 1e-9 is 10x above it.
        # Small values are a saturated softmax (one world dominates at this T), not a zero.
        nonzero_gibbs += np.max(np.abs(g_gb)) > 1e-9
        min_gibbs = min(min_gibbs, np.max(np.abs(g_gb)))
    print(f"        K=4, {draws} draws: ground-state grad == 0 in {zero_gs}/{draws}; "
          f"Gibbs (T={T}) grad != 0 in {nonzero_gibbs}/{draws} "
          f"(smallest max|grad| {min_gibbs:.1e}; FD noise ~1e-10)")
    report("Gamma = 0 ground-state loss is piecewise constant (zero gradient a.e.)",
           zero_gs == draws, zero_gs == 0)
    report("classical Gibbs collapse softmax(-E/T) has a non-zero gradient",
           nonzero_gibbs == draws, nonzero_gibbs == 0)


def thermal_P(Hm, T):
    lam, V = np.linalg.eigh(Hm)
    w = np.exp(-(lam - lam[0]) / T)
    rho = (V * w) @ V.T
    return np.diag(rho) / np.trace(rho)


def check_thermal():
    print("\n=== 7. Thermal collapse: Gibbs at Gamma = 0, ground state as T -> 0 ===")
    K = 8
    h, J = RNG.normal(size=K), sym(K)
    E = ising_energy(h, J, K)
    T = 0.7
    P_th = thermal_P(hamiltonian(h, J, 0.0, K), T)
    P_gibbs = np.exp(-(E - E.min()) / T); P_gibbs /= P_gibbs.sum()
    _, _, psi = ground_state(hamiltonian(h, J, 0.8, K))
    tv = [0.5 * np.abs(thermal_P(hamiltonian(h, J, 0.8, K), t) - psi ** 2).sum()
          for t in (1.0, 0.1, 0.01)]
    print(f"        K=8: max|P_thermal(Gamma=0) - softmax(-E/T)| = "
          f"{np.max(np.abs(P_th - P_gibbs)):.1e}; "
          f"TV(thermal, psi0^2) at T=1, 0.1, 0.01: {tv[0]:.3f}, {tv[1]:.3f}, {tv[2]:.1e}")
    report("diag(exp(-H/T))/Tr at Gamma = 0 equals the classical Gibbs distribution",
           np.allclose(P_th, P_gibbs, atol=1e-12), np.allclose(P_th, np.ones(2 ** K) / 2 ** K))
    report("thermal collapse -> ground-state collapse as T -> 0", tv[2] < 1e-6, tv[0] < 1e-6)

    # Daleckii-Krein gradient of the thermal loss vs finite differences (K = 4)
    K = 4
    h, J, a, y, T = RNG.normal(size=K), sym(K), 0.2, 1, 0.6

    def loss_th(h_, J_, a_):
        return -math.log(answer_probs(thermal_P(hamiltonian(h_, J_, softplus(a_), K), T), K)[1][y])

    Hm = hamiltonian(h, J, softplus(a), K)
    lam, V = np.linalg.eigh(Hm)
    f = np.exp(-(lam - lam[0]) / T)
    df = -f / T
    L = lam[:, None] - lam[None, :]
    G = np.where(np.abs(L) > 1e-12, (f[:, None] - f[None, :]) / np.where(L == 0, 1, L),
                 df[:, None] * np.ones_like(L))
    rho = (V * f) @ V.T
    Z = np.trace(rho)
    P = np.diag(rho) / Z
    M = answer_masks(K)
    s = M @ P
    dL_ds = -np.eye(len(s))[y] / (s[y] + EPS) + 1.0 / (s + EPS).sum()
    dL_dP = M.T @ dL_ds

    def directional(dH):
        drho = V @ (G * (V.T @ dH @ V)) @ V.T
        dP = (np.diag(drho) * Z - np.diag(rho) * np.trace(drho)) / Z ** 2
        return dL_dP @ dP

    z = 1 - 2 * bits(K)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    ga = np.array([directional(np.diag(z[:, i])) for i in range(K)]
                  + [directional(np.diag(z[:, i] * z[:, j])) for i, j in pairs]
                  + [directional(-x_sum(K)) / (1 + math.exp(-a))])
    gf = grad_fd(loss_th, h, J, a, K, pairs)
    rel = np.max(np.abs(ga - gf)) / np.max(np.abs(gf))
    print(f"        K=4 thermal (T={T}): Daleckii-Krein vs finite differences, rel err {rel:.2e}")
    report("thermal-collapse gradient (divided differences, degeneracy-safe) matches FD",
           rel < 1e-5, rel > 1.0)


def check_marginals():
    print("\n=== 8. Answer marginals ===")
    K, ok_P, ok_s, ok_p, raw_one = 8, True, True, True, 0
    draws = 200
    for _ in range(draws):
        h, J, g = RNG.normal(size=K), sym(K), abs(RNG.normal())
        _, _, psi = ground_state(hamiltonian(h, J, g, K))
        P = psi ** 2
        s, p = answer_probs(P, K)
        ok_P &= np.isclose(P.sum(), 1.0)
        ok_s &= np.all(s >= 0) and np.all(s <= 1 + 1e-12)
        ok_p &= np.isclose(p.sum(), 1.0) and np.all(p > 0)
        raw_one += np.isclose(s.sum(), 1.0)
    print(f"        K=8, {draws} draws; raw scores summed to 1 in {raw_one}/{draws}")
    report("sum_x P(x) = 1", ok_P, False)
    report("0 <= score_n <= 1 (a marginal probability)", ok_s, False)
    report("normalized p_n sums to 1, every p_n > 0", ok_p, raw_one == draws)


def check_entanglement():
    print("\n=== 9. Entanglement entropy of psi0 across supports | attacks ===")
    K = 8
    sup = [0, 2, 4, 6]
    h, J = RNG.normal(size=K), sym(K)
    _, _, psi_prod = ground_state(hamiltonian(h, np.zeros((K, K)), 0.8, K))
    _, _, psi_cl = ground_state(hamiltonian(h, J, 0.0, K))
    _, _, psi_q = ground_state(hamiltonian(h, J, 0.8, K))
    S = [entanglement_entropy(p, K, sup) for p in (psi_prod, psi_cl, psi_q)]
    print(f"        S(J=0, Gamma=0.8) = {S[0]:.2e}; S(Gamma=0) = {S[1]:.2e}; "
          f"S(J random, Gamma=0.8) = {S[2]:.4f} nats (max {4 * math.log(2):.4f})")
    report("no couplings -> product state, S = 0", S[0] < 1e-10, S[0] > 0.01)
    report("Gamma = 0 -> basis state, S = 0", S[1] < 1e-10, S[1] > 0.01)
    report("couplings and Gamma > 0 -> S > 0 (entanglement between paths)", S[2] > 1e-3, S[2] < 1e-10)


def check_degeneracy():
    print("\n=== 10. Near-degenerate ground states ===")
    K = 4
    J = sym(K)
    h0 = np.zeros(K)                       # Z2 symmetric: flipping every z leaves E unchanged
    gaps = []
    for g in (0.0, 0.01, 0.1, 1.0):
        lam, _, _ = ground_state(hamiltonian(h0, J, g, K))
        gaps.append(lam[1] - lam[0])
    print("        h = 0 (global flip symmetry), gap at Gamma = 0, 0.01, 0.1, 1: "
          + ", ".join(f"{v:.2e}" for v in gaps))
    lam, V, _ = ground_state(hamiltonian(h0, J, 0.0, K))
    deg = np.flatnonzero(lam - lam[0] < 1e-9)
    U = V[:, deg]
    R = np.array([[math.cos(0.7), -math.sin(0.7)], [math.sin(0.7), math.cos(0.7)]])
    U2 = U @ R[: len(deg), : len(deg)]
    P_a, P_b = U[:, 0] ** 2, U2[:, 0] ** 2
    P_proj_a = (U ** 2).sum(1) / len(deg)
    P_proj_b = (U2 ** 2).sum(1) / len(deg)
    print(f"        Gamma = 0 degenerate subspace dim {len(deg)}; "
          f"|P(first vector) - P(rotated first vector)|max = {np.max(np.abs(P_a - P_b)):.3f}")
    report("exact degeneracy at Gamma = 0, h = 0; Gamma > 0 opens a gap",
           gaps[0] < 1e-9 and gaps[3] > 1e-3, gaps[0] > 1e-3)
    report("projector-averaged P is basis-invariant; a single eigh column is not",
           np.allclose(P_proj_a, P_proj_b) and not np.allclose(P_a, P_b),
           np.allclose(P_a, P_b))


def check_h2_hermitian():
    print("\n=== 11. H2 = H1 + K XX (+ L YY) is Hermitian with the stated entries ===")
    K = 6
    ok_sym, ok_xx, ok_yy = True, True, True
    b = bits(K)
    for _ in range(10):
        h, J, g = RNG.normal(size=K), sym(K), abs(RNG.normal())
        Kxx, Lyy = sym(K), sym(K)
        Hm = hamiltonian2(h, J, g, Kxx, K, Lyy)
        ok_sym &= np.allclose(Hm, Hm.T, atol=0) and np.isrealobj(Hm)
        i, j = 1, 4
        XX, YY = pair_flip(K, i, j, "XX"), pair_flip(K, i, j, "YY")
        for x in range(2 ** K):
            y = x ^ (1 << i) ^ (1 << j)
            ok_xx &= XX[y, x] == 1.0 and XX.sum() == 2 ** K
            same = b[x, i] == b[x, j]
            ok_yy &= YY[y, x] == (-1.0 if same else 1.0)
    bad = hamiltonian(h, J, g, K) + np.triu(pair_flip(K, 0, 1, "XX"))
    print(f"        K=6, 10 draws; YY entry = -1 where x_i = x_j, +1 where they differ")
    report("H2 == H2^T, real (XX and YY terms included)", ok_sym, np.allclose(bad, bad.T))
    report("<x xor e_i xor e_j| X_i X_j |x> = +1 for every x", ok_xx, False)
    report("<x xor e_i xor e_j| Y_i Y_j |x> = -z_i z_j (both signs occur)", ok_yy, False)


def thermal_rho(Hm, T):
    lam, V = np.linalg.eigh(Hm)
    w = np.exp(-(lam - lam[0]) / T)
    return (V * w) @ V.T


def stripped(Hm):
    """Stoquastic counterpart: off-diagonals replaced by -|.|."""
    D = np.diag(np.diag(Hm))
    return D - np.abs(Hm - D)


def check_sign_structure():
    print("\n=== 12. Sign structure: H1 never, H2 sometimes, H2s (K <= 0) never ===")
    K, draws, T = 8, 200, 0.5
    neg_gs = {"H1": 0, "H2": 0, "H2s": 0}
    neg_rho = {"H1": 0, "H2": 0, "H2s": 0}
    tv_strip = {"H1": 0.0, "H2": []}
    fneg_h2 = []
    for _ in range(draws):
        h, J, g = RNG.normal(size=K), sym(K), 0.05 + abs(RNG.normal())
        Kxx = sym(K)
        arms = {
            "H1": hamiltonian(h, J, g, K),
            "H2": hamiltonian2(h, J, g, Kxx, K),
            "H2s": hamiltonian2(h, J, g, -np.abs(Kxx), K),
        }
        for name, Hm in arms.items():
            _, _, psi = ground_state(Hm)
            f = negative_mass(psi)
            neg_gs[name] += f > 1e-9
            rho = thermal_rho(Hm, T)
            neg_rho[name] += rho.min() < -1e-12 * np.abs(rho).max()
            if name == "H2":
                fneg_h2.append(f)
        P1 = np.diag(thermal_rho(arms["H1"], T)); P1 = P1 / P1.sum()
        P1s = np.diag(thermal_rho(stripped(arms["H1"]), T)); P1s = P1s / P1s.sum()
        tv_strip["H1"] = max(tv_strip["H1"], 0.5 * np.abs(P1 - P1s).sum())
        P2 = np.diag(thermal_rho(arms["H2"], T)); P2 = P2 / P2.sum()
        P2s = np.diag(thermal_rho(stripped(arms["H2"]), T)); P2s = P2s / P2s.sum()
        tv_strip["H2"].append(0.5 * np.abs(P2 - P2s).sum())
    fneg_h2 = np.array(fneg_h2)
    tv2 = np.array(tv_strip["H2"])
    print(f"        K=8, {draws} draws, T={T}: ground state with negative amplitudes "
          f"H1 {neg_gs['H1']}/{draws}, H2 {neg_gs['H2']}/{draws}, H2s {neg_gs['H2s']}/{draws}")
    print(f"        negative entries in exp(-H/T): H1 {neg_rho['H1']}/{draws}, "
          f"H2 {neg_rho['H2']}/{draws}, H2s {neg_rho['H2s']}/{draws}")
    print(f"        H2 negative amplitude mass: median {np.median(fneg_h2):.3f}, "
          f"max {fneg_h2.max():.3f}")
    print(f"        TV(P, P of sign-stripped H): H1 max {tv_strip['H1']:.1e}; "
          f"H2 median {np.median(tv2):.3f}, max {tv2.max():.3f}")
    report("H1 (stoquastic): no ground state has a negative amplitude",
           neg_gs["H1"] == 0, neg_gs["H1"] > 0)
    report("H2 (unconstrained K): some ground states have negative amplitudes",
           neg_gs["H2"] > 0, neg_gs["H2"] == 0)
    report("H2s (K <= 0, stoquastic capacity control): no negative amplitudes",
           neg_gs["H2s"] == 0, neg_gs["H2s"] > 0)
    report("exp(-H/T) entrywise >= 0 for H1 and H2s; negative entries occur for H2",
           neg_rho["H1"] == 0 and neg_rho["H2s"] == 0 and neg_rho["H2"] > 0,
           neg_rho["H2"] == 0)
    report("signs change the readout for H2 only (sign-stripped TV: H1 = 0, H2 > 0)",
           tv_strip["H1"] < 1e-12 and np.median(tv2) > 1e-3, tv_strip["H1"] > 1e-6)


def check_directed_mapping():
    print("\n=== 13. Directed 0/1-basis terms map exactly to Ising form ===")
    K, draws, worst = 8, 50, 0.0
    b01 = bits(K)
    for _ in range(draws):
        a = RNG.normal(size=K)
        bb = sym(K)
        r = np.abs(RNG.normal(size=(K, K)))
        np.fill_diagonal(r, 0.0)
        E01 = (b01 @ a
               + np.einsum("xi,ij,xj->x", b01, np.triu(bb, 1), b01)
               + np.einsum("xi,ij,xj->x", b01, r, 1 - b01))
        h, J, const = directed_to_ising(a, bb, r)
        worst = max(worst, np.max(np.abs(E01 - (ising_energy(h, J, K) + const))))
    # control: forget the pair contribution to the fields
    h, J, const = directed_to_ising(a, bb, r)
    q = 4 * J
    E_bad = ising_energy(h + q.sum(axis=1) / 4, J, K) + const
    # asymmetry: r_01 alone vs r_10 alone give different energies
    a0, b0 = np.zeros(K), np.zeros((K, K))
    r01 = np.zeros((K, K)); r01[0, 1] = 1.0
    E_01 = ising_energy(*directed_to_ising(a0, b0, r01)[:2], K) + directed_to_ising(a0, b0, r01)[2]
    E_10 = ising_energy(*directed_to_ising(a0, b0, r01.T)[:2], K) + directed_to_ising(a0, b0, r01.T)[2]
    x_10 = 1          # x_0 = 1, x_1 = 0: path 0 holds without path 1
    print(f"        K=8, {draws} random (a, b, r), all 256 worlds: max |E01 - E_ising - const| = {worst:.1e}")
    print(f"        world (x0=1, x1=0): penalty {E_01[x_10]:.2f} under r_01 = 1, "
          f"{E_10[x_10]:.2f} under r_10 = 1")
    report("x_i(1 - x_j) terms reproduced exactly for every world", worst < 1e-12,
           np.max(np.abs(E01 - E_bad)) < 1e-12)
    report("the mapped Ising form keeps the direction (r_01 != r_10)",
           np.isclose(E_01[x_10], 1.0) and np.isclose(E_10[x_10], 0.0),
           np.isclose(E_01[x_10], E_10[x_10]))


def check_h2_thermal_gradient():
    print("\n=== 14. Thermal-collapse gradient for H2 matches finite differences (K = 4) ===")
    K, T, worst = 4, 0.6, 0.0
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    for _ in range(5):
        h, J, a, y = RNG.normal(size=K), sym(K), RNG.normal(), int(RNG.integers(K // 2))
        Kxx = sym(K)

        def loss(h_, J_, a_, K_):
            P = np.diag(thermal_rho(hamiltonian2(h_, J_, softplus(a_), K_, K), T))
            return -math.log(answer_probs(P / P.sum(), K)[1][y])

        Hm = hamiltonian2(h, J, softplus(a), Kxx, K)
        lam, V = np.linalg.eigh(Hm)
        f = np.exp(-(lam - lam[0]) / T)
        L = lam[:, None] - lam[None, :]
        G = np.where(np.abs(L) > 1e-12, (f[:, None] - f[None, :]) / np.where(L == 0, 1, L),
                     (-f / T)[:, None] * np.ones_like(L))
        rho = (V * f) @ V.T
        Z = np.trace(rho)
        P = np.diag(rho) / Z
        M = answer_masks(K)
        s = M @ P
        dL_dP = M.T @ (-np.eye(len(s))[y] / (s[y] + EPS) + 1.0 / (s + EPS).sum())

        def directional(dH):
            drho = V @ (G * (V.T @ dH @ V)) @ V.T
            return dL_dP @ ((np.diag(drho) * Z - np.diag(rho) * np.trace(drho)) / Z ** 2)

        z = 1 - 2 * bits(K)
        ga = np.array([directional(np.diag(z[:, i])) for i in range(K)]
                      + [directional(np.diag(z[:, i] * z[:, j])) for i, j in pairs]
                      + [directional(-x_sum(K)) / (1 + math.exp(-a))]
                      + [directional(pair_flip(K, i, j, "XX")) for i, j in pairs])
        step = 1e-6
        gf = list(grad_fd(lambda h_, J_, a_: loss(h_, J_, a_, Kxx), h, J, a, K, pairs, step))
        for i, j in pairs:
            E = np.zeros((K, K)); E[i, j] = E[j, i] = step
            gf.append((loss(h, J, a, Kxx + E) - loss(h, J, a, Kxx - E)) / (2 * step))
        gf = np.array(gf)
        worst = max(worst, np.max(np.abs(ga - gf)) / np.max(np.abs(gf)))
    print(f"        5 draws; parameters: 4 h, 6 J, 1 a, 6 K_xx; worst relative error {worst:.2e}")
    report("H2 thermal gradient (Daleckii-Krein) == central finite differences (rel err < 1e-5)",
           worst < 1e-5, worst > 1.0)


def bench():
    print("\n=== Benchmark (informational): numpy.linalg.eigh, batch 16, this CPU ===")
    for K in (8, 10):
        D = 2 ** K
        for dtype in (np.float64, np.float32):
            mats = []
            for _ in range(16):
                Hm = hamiltonian(RNG.normal(size=K), sym(K), 0.5, K).astype(dtype)
                mats.append(Hm)
            A = np.stack(mats)
            np.linalg.eigh(A)
            reps = 5 if K == 8 else 2
            t0 = time.perf_counter()
            for _ in range(reps):
                np.linalg.eigh(A)
            dt = (time.perf_counter() - t0) / reps
            print(f"        K={K:2d} ({D}x{D}) {np.dtype(dtype).name}: "
                  f"{dt * 1000:8.1f} ms per batch of 16 (forward only)")


def main():
    print(f"numpy {np.__version__}, Python {sys.version.split()[0]}")
    check_hermitian()
    check_conventions()
    check_gamma0_ground_state()
    check_perron_frobenius()
    check_eigh_gradient()
    check_gamma0_untrainable()
    check_thermal()
    check_marginals()
    check_entanglement()
    check_degeneracy()
    check_h2_hermitian()
    check_sign_structure()
    check_directed_mapping()
    check_h2_thermal_gradient()
    if "--bench" in sys.argv:
        bench()
    print()
    if FAILURES:
        print(f"SUMMARY: {len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print(f"SUMMARY: {len(CHECKS)}/{len(CHECKS)} checks PASS (each with its wrong-claim control failing as required)")


if __name__ == "__main__":
    main()
