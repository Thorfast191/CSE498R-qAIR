"""
Numerical checks for every checkable mathematical claim in
docs/baseline/issue_dossier.md.

CPU only, numpy + the standard library; no torch, no model, no data. Each
check evaluates the dossier's claim (must PASS) and a deliberately wrong
"control" claim (must FAIL). A check counts as verified only when both
behave as expected, so a check that passes everything cannot slip through.

`coherent_readout` mirrors models/interference.py::CoherentCollapse.forward
(v46, lines 138-222) operation for operation, without padding masks.

    python scripts/verify_dossier_math.py        # exit code 1 on any failure
"""

import math
import sys

import numpy as np

LOG8 = math.log(8)

# Observed values from the executed v46 notebook (notebooks/qAIR_v46.ipynb).
OBS = {
    "entropy": [2.0704, 2.0742, 2.0759, 2.0755, 2.0754, 2.0760, 2.0763],
    "diversity": [0.018196, 0.010570, 0.007043, 0.008168, 0.008414,
                  0.007223, 0.006711, 0.006660],
    "collapse_peak_final": 0.1406,
    "polarity_phase": [3.1421, 3.1429, 3.1427, 3.1418, 3.1410, 3.1402,
                       3.1398, 3.1396],
    "train_loss": [1.8890, 1.6500, 1.3585, 1.1235, 0.9435, 0.8551, 0.7924,
                   0.7425],
    "val_acc": [0.35, 0.35, 0.40, 0.45, 0.40, 0.40, 0.45, 0.50],
    "params": 1_056_382,
    "train_n": 80,
}

# Optimizer facts read from ckpt/qair_v46_best.pt (param_groups, scheduler).
BASE_LR, ETA_MIN, WEIGHT_DECAY, EPOCHS, STEPS_PER_EPOCH = 5e-4, 1e-6, 0.05, 8, 5

results = []


def check(name, claim, control, detail=""):
    """claim must be True, control (a deliberately wrong claim) must be False."""

    ok = bool(claim) and not bool(control)
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        claim={'True ' if claim else 'False'}  "
          f"control(wrong)={'True ' if control else 'False'}  {detail}")


def softplus(x):
    return math.log1p(math.exp(x))


# ---------------------------------------------------------------------------
# Readout, mirroring models/interference.py (batched over axis 0)
# ---------------------------------------------------------------------------

def coherent_readout(collapse_energy, answer_energy, phase_h=None, phase_kn=None,
                     t_hyp=1.0, t_opt=1.0, mode="learned"):
    """
    collapse_energy (B, K), answer_energy (B, K, N), phase_h (B, K),
    phase_kn (B, K, N). Returns a dict with the same names as the module.
    """

    log_r = -collapse_energy / (2.0 * t_hyp)
    log_r = log_r - log_r.max(axis=1, keepdims=True)
    r = np.exp(log_r)
    r = r / np.sqrt((r ** 2).sum(axis=1, keepdims=True) + 1e-8)
    collapse_probs = r ** 2
    collapse_probs = collapse_probs / (collapse_probs.sum(axis=1, keepdims=True) + 1e-8)

    B = answer_energy.shape[0]
    log_t = -answer_energy / (2.0 * t_opt)
    log_t = log_t - log_t.reshape(B, -1).max(axis=1)[:, None, None]
    t = np.exp(log_t)

    w = r[:, :, None] * t
    classical = (w ** 2).sum(axis=1)
    in_phase = w.sum(axis=1) ** 2

    if mode == "classical":
        coherent = classical
    elif mode == "zero" or phase_h is None:
        coherent = in_phase
    else:
        phase = np.broadcast_to(phase_h[:, :, None], w.shape)
        if phase_kn is not None:
            phase = phase + phase_kn
        real = (w * np.cos(phase)).sum(axis=1)
        imag = (w * np.sin(phase)).sum(axis=1)
        coherent = real ** 2 + imag ** 2

    interference = coherent - classical
    weights = np.clip(coherent, 1e-12, None)
    log_probs = np.log(weights) - np.log(weights.sum(axis=1, keepdims=True))

    return {
        "probs": np.exp(log_probs),
        "coherent": coherent,
        "classical": classical,
        "interference": interference,
        "destructive_fraction": float((interference < 0).mean()),
        "collapse_probs": collapse_probs,
    }


def polarity_phases(circuit_phase, polarity, rho):
    """models/full_model.py:390 -- +0 for support, +rho for attack."""

    return circuit_phase + 0.5 * (1.0 - polarity) * rho


def entropy(p):
    p = np.clip(p, 1e-300, None)
    return float(-(p * np.log(p)).sum())


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(b, c):
    """Same formula as evaluation/stats.py::mcnemar."""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# 1. Collapse entropy, KL to uniform, implied logit spread
# ---------------------------------------------------------------------------

def check_entropy():
    print("\n=== 1. Collapse entropy vs log 8 ===")

    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(8))
    kl = float((p * np.log(p * 8)).sum())
    check("KL(p||U) = log K - H(p) (identity)",
          abs(kl - (LOG8 - entropy(p))) < 1e-12,
          abs(kl - (entropy(p) - LOG8)) < 1e-12,
          f"KL={kl:.6f}")

    kl_first, kl_last = LOG8 - OBS["entropy"][0], LOG8 - OBS["entropy"][-1]
    print(f"        log 8 = {LOG8:.5f}; KL epoch 1 = {kl_first:.5f}, "
          f"KL final = {kl_last:.5f} nats")

    p_half = np.array([0.5] + [0.5 / 7] * 7)
    kl_half = LOG8 - entropy(p_half)
    kl_four = math.log(2)
    print(f"        selective references: peak 0.5 on one hypothesis -> H={entropy(p_half):.4f}, "
          f"KL={kl_half:.4f}; uniform on 4 of 8 -> KL={kl_four:.4f}")
    print(f"        logit gap needed for a 0.5 peak: ln 7 = {math.log(7):.4f}")
    check("observed collapse is near-uniform (KL < 0.01 nats every epoch)",
          max(LOG8 - h for h in OBS["entropy"]) < 0.01,
          kl_last >= 0.1,
          f"final KL is {kl_half / kl_last:.0f}x below the 0.5-peak reference")

    # Second-order expansion: KL(softmax(l) || U) ~ Var(l) / 2 for small spread.
    ratios = []
    for v in (0.002, 0.007, 0.018):
        l = rng.normal(0, math.sqrt(v), size=(20000, 8))
        q = np.exp(l - l.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        kl_sim = float((q * np.log(q * 8)).sum(1).mean())
        ratios.append(kl_sim / (l.var(axis=1).mean() / 2))
    check("KL ~ Var(logits)/2 for small spread (within 5%)",
          all(abs(x - 1) < 0.05 for x in ratios),
          all(abs(x - 2) < 0.05 for x in ratios),
          f"KL/(Var/2) = {', '.join(f'{x:.3f}' for x in ratios)}")

    # logits = -E/T and diversity = Var_k(E)  =>  T^2 = diversity / (2 KL)
    t2_first = OBS["diversity"][0] / (2 * kl_first)
    t2_last = OBS["diversity"][6] / (2 * kl_last)
    t_init = 0.3 + math.log(2)
    print(f"        implied collapse temperature: epoch 1 T={math.sqrt(t2_first):.3f}, "
          f"epoch 7 T={math.sqrt(t2_last):.3f}; init 0.3+softplus(0)={t_init:.4f}")
    check("entropy is explained by diversity with T ~ 1 (both epochs within 10%)",
          all(0.9 < math.sqrt(x) < 1.1 for x in (t2_first, t2_last)),
          all(abs(math.sqrt(x) - 0.5) < 0.1 for x in (t2_first, t2_last)))

    sigma = math.sqrt(OBS["diversity"][-1])
    l = rng.normal(0, sigma, size=(200000, 8))
    q = np.exp(l - l.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    peak = float(q.max(1).mean())
    l10 = rng.normal(0, math.sqrt(10) * sigma, size=(200000, 8))
    q10 = np.exp(l10 - l10.max(1, keepdims=True))
    q10 /= q10.sum(1, keepdims=True)
    check("Gaussian logits with Var=final diversity reproduce the collapse peak (+/-0.005)",
          abs(peak - OBS["collapse_peak_final"]) < 0.005,
          abs(float(q10.max(1).mean()) - OBS["collapse_peak_final"]) < 0.005,
          f"predicted {peak:.4f}, observed {OBS['collapse_peak_final']}")


# ---------------------------------------------------------------------------
# 2. Sign symmetry of the readout (C1)
# ---------------------------------------------------------------------------

def check_sign_symmetry():
    print("\n=== 2. Sign symmetry of the coherent readout ===")

    eps = 1e-9
    # Hypotheses (mode-major): sup0, sup1, att0, att1. Each mainly weighs
    # on its own option. Option 0: support 0.6, attack 0.1; option 1 swapped.
    t = np.array([[0.6, eps], [eps, 0.1], [0.1, eps], [eps, 0.6]])
    t_opt = 1.0
    answer_energy = (-2 * t_opt * np.log(t))[None]
    collapse_energy = np.zeros((1, 4))
    pol = np.array([1.0, 1.0, -1.0, -1.0])
    phase_h = polarity_phases(np.zeros(4), pol, math.pi)[None]

    out = coherent_readout(collapse_energy, answer_energy, phase_h, None, 1.0, t_opt)
    p = out["probs"][0]
    s_minus_a = (0.6 - 0.1, 0.1 - 0.6)
    print(f"        S-A per option = {s_minus_a}; coherent = {out['coherent'][0]}; "
          f"P = {p}")
    check("strongly supported and strongly refuted options tie: P0 == P1",
          abs(p[0] - p[1]) < 1e-9,
          p[0] > p[1] + 1e-6,
          "control claims the readout prefers the supported option")

    # Same configuration with a reference amplitude c added to every option
    # (coherent = (c + S - A)^2), the "with a reference term" case.
    c = 1.0
    ref = (c + np.array(s_minus_a)) ** 2
    p_ref = ref / ref.sum()
    check("with a reference term the ranking follows S - A (additive scoring)",
          p_ref[0] > p_ref[1],
          abs(p_ref[0] - p_ref[1]) < 1e-9,
          f"P_ref = {p_ref.round(4)}")

    # Ranking is a function of |S - A|: random S, A with the swap
    rng = np.random.default_rng(1)
    S, A = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)
    check("(S-A)^2 is invariant under swapping S and A for every random draw",
          np.allclose((S - A) ** 2, (A - S) ** 2),
          np.allclose(S - A, A - S))


# ---------------------------------------------------------------------------
# 3. Validator gradient under the v46 target (A7)
# ---------------------------------------------------------------------------

def check_validator_gradient():
    print("\n=== 3. Validator BCE gradient under the v46 target ===")

    rng = np.random.default_rng(2)
    K, N, y = 8, 4, 1
    logits = rng.normal(0, 1, size=(K, N))
    sigma = 1 / (1 + np.exp(-logits))

    # v46 models/validator.py:141-142  target[b, :, y] = 1.0
    target_v46 = np.zeros((K, N))
    target_v46[:, y] = 1.0
    grad_v46 = sigma - target_v46          # dBCE/dlogit

    pol = np.array([1.0] * N + [-1.0] * N)
    hyp_option = np.array(list(range(N)) * 2)
    sound = np.where(pol > 0, hyp_option == y, hyp_option != y).astype(float)
    print(f"        sound hypotheses (support on y, attacks on wrong options): {sound.astype(int)}")

    check("v46: d loss/d logit[k, y] < 0 for EVERY hypothesis k (sound or not)",
          bool((grad_v46[:, y] < 0).all()),
          bool((grad_v46[sound == 0, y] > 0).any()),
          f"grad[:, y] = {grad_v46[:, y].round(3)}")

    means = target_v46.mean(axis=1)
    check("v46: option-mean of the target is 1/N for every k (no k signal)",
          np.allclose(means, 1 / N),
          np.ptp(means) > 0,
          f"mean_n target[k] = {means}")

    # Candidate fix in the v47 patch: target gated by soundness
    target_gated = target_v46 * sound[:, None]
    grad_gated = sigma - target_gated
    check("gated target (v47 candidate): unsound hypotheses are pushed DOWN at y",
          bool((grad_gated[sound == 0, y] > 0).all() and (grad_gated[sound == 1, y] < 0).all()),
          bool((grad_gated[:, y] < 0).all()),
          f"option-means now {target_gated.mean(axis=1)}")


# ---------------------------------------------------------------------------
# 4. destructive_fraction is fixed by polarity_phase (C2)
# ---------------------------------------------------------------------------

def check_destructive_fraction():
    print("\n=== 4. destructive_fraction at polarity_phase = pi ===")

    N = 4
    K = 2 * N
    pol = np.array([1.0] * N + [-1.0] * N)
    rho_star = math.acos(-(N - 1) / N)
    print(f"        uniform-amplitude threshold rho* = arccos(-(N-1)/N) = {rho_star:.4f} rad")

    def uniform_fraction(rho):
        ph = polarity_phases(np.zeros(K), pol, rho)[None]
        out = coherent_readout(np.zeros((1, K)), np.zeros((1, K, N)), ph, np.zeros((1, K, N)))
        return out["destructive_fraction"]

    check("uniform amplitudes: fraction jumps 0 -> 1 at rho* (exact)",
          uniform_fraction(rho_star - 0.01) == 0.0 and uniform_fraction(rho_star + 0.01) == 1.0,
          uniform_fraction(0.0) == 1.0)

    # Init-scale random inputs: answer energy standardized then divided by the
    # selector temperature 0.5+softplus(1)=1.813 (std ~0.55); collapse-energy
    # variance = epoch-1 diversity; T_hyp = 0.993; T_opt = 0.3+softplus(0.5413);
    # circuit-phase spread 0.11 (sanity cell: "phase spread 0.1126");
    # compatibility-phase spread 0.16 [ASSUMPTION].
    rng = np.random.default_rng(3)
    B = 4000
    t_opt = 0.3 + softplus(0.5413)
    ae = rng.normal(0, 1 / (0.5 + softplus(1.0)), size=(B, K, N))
    ce = rng.normal(0, math.sqrt(OBS["diversity"][0]), size=(B, K))

    def random_fraction(rho, phase_sd=0.11, theta_sd=0.16):
        ph = polarity_phases(rng.normal(0, phase_sd, size=(B, K)), pol[None], rho)
        th = rng.normal(0, theta_sd, size=(B, K, N))
        return coherent_readout(ce, ae, ph, th, 0.993, t_opt)["destructive_fraction"]

    sweep = {rho: random_fraction(rho) for rho in (0.0, 1.6, 2.0, 2.4, 2.8, 3.0, math.pi)}
    print("        rho -> fraction: " + ", ".join(f"{k:.2f}:{v:.3f}" for k, v in sweep.items()))
    print("        (config.py:223-226 measured on the real model: 0.000 up to 1.6, "
          "0.494 at 2.4, 0.994 at 2.8, 1.000 at >=3.0)")
    check("init-scale random inputs: fraction >= 0.99 at pi, < 0.05 at 1.6",
          sweep[math.pi] >= 0.99 and sweep[1.6] < 0.05,
          sweep[0.0] >= 0.99)

    spreads = {sd: random_fraction(math.pi, sd, sd) for sd in (0.1, 0.5, 1.0, 2.0)}
    print("        fraction at pi vs phase spread (circuit=compat sd): "
          + ", ".join(f"{k}:{v:.3f}" for k, v in spreads.items()))


# ---------------------------------------------------------------------------
# 5. Gradient of the readout with respect to polarity_phase (C2, B8)
# ---------------------------------------------------------------------------

def check_phase_gradient():
    print("\n=== 5. d coherent / d rho and the Adam step budget ===")

    rng = np.random.default_rng(4)
    N, K = 4, 8
    pol = np.array([1.0] * N + [-1.0] * N)
    ae = rng.normal(0, 0.55, size=(1, K, N))
    ce = rng.normal(0, 0.13, size=(1, K))

    def coherent_sum(rho, delta=0.0):
        circ = np.where(pol > 0, 0.0, delta)        # misalign the attack channel by delta
        ph = polarity_phases(circ, pol, rho)[None]
        return coherent_readout(ce, ae, ph, np.zeros((1, K, N)))["coherent"][0]

    h = 1e-6
    g_pi = (coherent_sum(math.pi + h) - coherent_sum(math.pi - h)) / (2 * h)
    g_half = (coherent_sum(math.pi / 2 + h) - coherent_sum(math.pi / 2 - h)) / (2 * h)
    check("aligned channels: d coherent_n / d rho = 0 at rho = pi (all options)",
          np.abs(g_pi).max() < 1e-6,
          np.allclose(g_pi, g_half, atol=1e-6),
          f"max|grad| at pi = {np.abs(g_pi).max():.2e}, at pi/2 = {np.abs(g_half).max():.3f}")

    # Analytic: coherent = |S|^2 + |A|^2 + 2 Re(conj(S) A e^{i rho});
    # at rho = pi, d/d rho = 2 |S||A| sin(arg A - arg S)
    delta = 0.2
    g_mis = (coherent_sum(math.pi + h, delta) - coherent_sum(math.pi - h, delta)) / (2 * h)

    # Channel sums S_n, A_n built exactly as coherent_readout builds w_kn
    # (t_hyp = t_opt = 1; max-subtraction cancels in the r normalization).
    r = np.exp(-ce / 2)
    r /= np.sqrt((r ** 2).sum())
    t = np.exp(-ae / 2)
    t /= t.max()
    wkn = (r[0][:, None] * t[0])
    S = wkn[:N].sum(0)
    A = wkn[N:].sum(0)
    analytic = 2 * S * A * math.sin(delta)
    check("misaligned by delta: gradient at pi = 2|S||A| sin(delta)",
          np.allclose(g_mis, analytic, rtol=1e-4),
          np.allclose(g_mis, 0, atol=1e-6),
          f"fd {g_mis.round(5)} vs analytic {analytic.round(5)}")

    # Adam moves a parameter by ~lr per step (|m_hat/sqrt(v_hat)| <~ 1).
    lrs = [ETA_MIN + (BASE_LR - ETA_MIN) * (1 + math.cos(math.pi * e / EPOCHS)) / 2
           for e in range(EPOCHS)]
    budget = STEPS_PER_EPOCH * sum(lrs)
    traj = [math.pi] + OBS["polarity_phase"]
    path = sum(abs(b - a) for a, b in zip(traj, traj[1:]))
    net = traj[-1] - math.pi
    decay_only = -sum(STEPS_PER_EPOCH * lr * WEIGHT_DECAY * math.pi for lr in lrs)
    print(f"        Adam budget sum(lr_t) over 40 steps = {budget:.5f} rad; "
          f"observed path length {path:.5f}, net drift {net:+.5f}; "
          f"weight-decay-only drift {decay_only:+.5f}")
    check("observed polarity_phase movement is within the Adam step budget",
          path <= budget,
          path > budget)


# ---------------------------------------------------------------------------
# 6. Binomial facts, McNemar, CI half-widths, bootstrap
# ---------------------------------------------------------------------------

def check_binomial():
    print("\n=== 6. Small-sample statistics ===")

    p3 = binom_cdf(3, 20, 0.25)
    p4 = binom_cdf(4, 20, 0.25)
    check("P(X <= 3 | n=20, p=0.25) = 0.225 (A3 at 3/20 is consistent with chance)",
          abs(p3 - 0.2252) < 5e-4,
          p3 < 0.05,
          f"P(X<=3)={p3:.4f}, P(X<=4)={p4:.4f} (P0 at 4/20)")

    p = mcnemar_exact(2, 4)
    check("exact McNemar b=2, c=4 -> p = 0.6875",
          abs(p - 0.6875) < 1e-12,
          p < 0.05)

    p_best = mcnemar_exact(0, 5)
    check("5 net flips out of 20 cannot reach p < 0.05 even with no reverse flips",
          p_best > 0.05,
          p_best < 0.05,
          f"p(b=0, c=5) = {p_best:.4f}")

    hw = {n: 1.96 * math.sqrt(0.25 / n) for n in (20, 570, 1172, 2376)}
    print("        95% CI half-width at p=0.5: " + ", ".join(f"n={k}: +/-{v:.3f}" for k, v in hw.items()))
    check("half-width ~0.22 at n=20, ~0.04 at n=570, ~0.02 at n=2376",
          abs(hw[20] - 0.219) < 0.002 and abs(hw[570] - 0.041) < 0.002 and abs(hw[2376] - 0.020) < 0.001,
          hw[20] < 0.1)

    rng = np.random.default_rng(0)
    x = np.array([1] * 10 + [0] * 10, dtype=float)
    boots = x[rng.integers(0, 20, size=(10000, 20))].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    check("percentile bootstrap of 10/20 gives ~[0.30, 0.70]",
          abs(lo - 0.30) < 0.051 and abs(hi - 0.70) < 0.051,
          hi - lo < 0.2,
          f"[{lo:.2f}, {hi:.2f}]")


# ---------------------------------------------------------------------------
# 7. Selection bias of max-over-epochs validation accuracy (B3)
# ---------------------------------------------------------------------------

def check_selection_bias():
    print("\n=== 7. Bias of best-of-8-epochs accuracy at n=20 ===")

    rng = np.random.default_rng(5)
    p, n, epochs, sims = float(np.mean(OBS["val_acc"])), 20, 8, 20000

    iid = rng.binomial(n, p, size=(sims, epochs)) / n
    bias_iid = float(iid.max(1).mean() - p)

    keep = 0.8   # [ASSUMPTION] each item keeps last epoch's outcome w.p. 0.8
    out = rng.random((sims, n)) < p
    best = out.mean(1)
    for _ in range(epochs - 1):
        redraw = rng.random((sims, n)) >= keep
        out = np.where(redraw, rng.random((sims, n)) < p, out)
        best = np.maximum(best, out.mean(1))
    bias_corr = float(best.mean() - p)

    p_ge_half = float((iid.max(1) >= 0.5).mean())
    print(f"        true acc p={p:.4f}: E[max]-p = {bias_iid:+.3f} (independent epochs), "
          f"{bias_corr:+.3f} (correlated, keep=0.8); P(best >= 0.50) = {p_ge_half:.2f}")
    check("best-of-8 is biased upward by several points at n=20",
          bias_corr > 0.03 and bias_iid > bias_corr,
          abs(bias_iid) < 0.01)


# ---------------------------------------------------------------------------
# 8. Overfitting arithmetic and the loss decomposition at initialization
# ---------------------------------------------------------------------------

def check_overfitting():
    print("\n=== 8. Capacity and loss ===")

    ratio = OBS["params"] / OBS["train_n"]
    drop7 = 1 - OBS["train_loss"][6] / OBS["train_loss"][0]
    drop8 = 1 - OBS["train_loss"][7] / OBS["train_loss"][0]

    # losses.py: CE (smoothed) + 0.4 ranking + 0.1 validator BCE + 0.02 (-tanh div)
    ce0 = math.log(4)
    rank0 = 1.0               # relu(1 - 0) for each distractor at a uniform posterior
    bce0 = math.log(2)
    init_pred = ce0 + 0.4 * rank0 + 0.1 * bce0 - 0.02 * math.tanh(OBS["diversity"][0])
    eps = 0.05
    tgt = np.array([1 - eps + eps / 4] + [eps / 4] * 3)
    ce_floor = entropy(tgt)
    print(f"        params/example = {ratio:.1f}; loss drop epoch 1->7 = {100 * drop7:.1f}%, "
          f"1->8 = {100 * drop8:.1f}%; predicted loss at a uniform posterior = {init_pred:.3f} "
          f"(epoch-1 mean {OBS['train_loss'][0]}); smoothed-CE floor = {ce_floor:.3f}")
    check("uniform-posterior loss decomposition matches epoch 1 within 0.05",
          abs(init_pred - OBS["train_loss"][0]) < 0.05,
          abs(ce0 - OBS["train_loss"][0]) < 0.05,
          "control: CE alone explains the epoch-1 loss")


# ---------------------------------------------------------------------------
# 9. Scale invariance of the Born normalization (B11 mechanism)
# ---------------------------------------------------------------------------

def check_scale_invariance():
    print("\n=== 9. Normalized readout depends only on amplitude ratios ===")

    resid = np.array([0.010, 0.020, 0.015, 0.012])   # near-cancelled S - A
    p_small = resid ** 2 / (resid ** 2).sum()
    p_big = (100 * resid) ** 2 / ((100 * resid) ** 2).sum()
    print(f"        residuals {resid} -> P = {p_small.round(3)} (max {p_small.max():.3f})")
    check("P(c * residual) == P(residual): magnitude of evidence does not temper confidence",
          np.allclose(p_small, p_big),
          not np.allclose(p_small, p_big))


# ---------------------------------------------------------------------------
# 10. Finite-sample ECE of a perfectly calibrated model (B11)
# ---------------------------------------------------------------------------

def ece_15_bins(conf, hit):
    """Same estimator as training/evaluate.py::expected_calibration_error."""

    edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return ece


def check_ece_noise():
    print("\n=== 10. ECE of a calibrated model at small n ===")

    rng = np.random.default_rng(6)
    means = {}
    for n in (20, 570, 2376):
        vals = []
        for _ in range(2000 if n < 1000 else 300):
            conf = rng.uniform(0.3, 0.9, n)       # [ASSUMPTION] confidence spread
            hit = (rng.random(n) < conf).astype(float)
            vals.append(ece_15_bins(conf, hit))
        means[n] = float(np.mean(vals))
    print("        expected ECE when calibrated by construction: "
          + ", ".join(f"n={k}: {v:.3f}" for k, v in means.items()))
    check("calibrated model at n=20 still shows ECE well above 0 (> 0.1)",
          means[20] > 0.1 and means[2376] < 0.05,
          means[20] < 0.02)


# ---------------------------------------------------------------------------
# 11. Pooling seeds inflates McNemar (B6)
# ---------------------------------------------------------------------------

def check_pooled_mcnemar():
    print("\n=== 11. Pooled-over-seeds McNemar under H0 ===")

    # H0: both arms have the same per-item accuracy p_i. Each arm has a
    # per-item "base" outcome that every seed keeps with probability kappa
    # (else redraws), so kappa controls the inter-seed correlation rho of the
    # paired difference d_is. kappa=0 -> independent seeds.
    rng = np.random.default_rng(7)
    n_items, seeds, reps = 300, 5, 600

    def rejection_rates(kappa):
        rej_pooled = rej_single = 0
        for _ in range(reps):
            p = rng.uniform(0.2, 0.8, n_items)

            def outcomes():
                base = rng.random(n_items) < p
                redraw = rng.random((seeds, n_items)) >= kappa
                return np.where(redraw, rng.random((seeds, n_items)) < p, base)

            a, b = outcomes(), outcomes()
            rej_pooled += mcnemar_exact(int((a & ~b).sum()), int((~a & b).sum())) < 0.05
            rej_single += mcnemar_exact(int((a[0] & ~b[0]).sum()), int((~a[0] & b[0]).sum())) < 0.05
        return rej_pooled / reps, rej_single / reps

    rates = {k: rejection_rates(k) for k in (0.0, 0.5, 0.8)}
    for k, (rp, rs) in rates.items():
        print(f"        kappa={k}: rejection pooled={rp:.3f}, single-seed={rs:.3f} (nominal 0.05)")
    check("pooled McNemar is valid only for independent seeds and over-rejects as seeds agree",
          0.02 <= rates[0.0][0] <= 0.08 and rates[0.8][0] > 0.15 and 0.02 <= rates[0.8][1] <= 0.08,
          abs(rates[0.8][0] - 0.05) < 0.03)


# ---------------------------------------------------------------------------
# 12. Zero-shot cache statistics vs chance at n=80 (D1, B10)
# ---------------------------------------------------------------------------

def check_cache_signal():
    print("\n=== 12. Train-cache zero-shot statistics (n=80) vs chance 0.25 ===")

    p_lo = {k: binom_cdf(k, 80, 0.25) for k in (14, 15, 16)}   # 0.175, 0.1875, 0.200
    p_hi_llm = 1 - binom_cdf(26, 80, 0.25)                         # 0.3375 = 27/80
    print("        one-sided P(X <= k | 80, 0.25): "
          + ", ".join(f"k={k} ({k / 80:.4f}): {v:.3f}" for k, v in p_lo.items())
          + f";  P(X >= 27) for the LLM likelihood = {p_hi_llm:.3f}")
    check("hypothesis statistics are consistent with chance (not significantly below)",
          all(v > 0.05 for v in p_lo.values()),
          all(v < 0.05 for v in p_lo.values()))


# ---------------------------------------------------------------------------
# 13. Common-mode gradient on option energies at near-uniform amplitudes (C7)
# ---------------------------------------------------------------------------

def check_common_mode_gradient():
    print("\n=== 13. d log P(y) / d E_kn within a channel ===")

    N, K, y = 4, 8, 2
    pol = np.array([1.0] * N + [-1.0] * N)
    rng = np.random.default_rng(8)

    def grad_rows(spread):
        ae = rng.normal(0, spread, size=(1, K, N))
        ce = np.zeros((1, K))
        ph = polarity_phases(np.zeros(K), pol, math.pi)[None]

        def logp(E):
            return math.log(coherent_readout(ce, E, ph, np.zeros((1, K, N)))["probs"][0, y])

        g = np.zeros((K, N))
        h = 1e-6
        for k in range(K):
            for n in range(N):
                e = ae.copy(); e[0, k, n] += h
                f = ae.copy(); f[0, k, n] -= h
                g[k, n] = (logp(e) - logp(f)) / (2 * h)
        gs = g[:N] - g[:N].mean(1, keepdims=True)
        gs /= np.linalg.norm(gs, axis=1, keepdims=True)
        cos = gs @ gs.T
        return float(cos[np.triu_indices(N, 1)].mean()), g

    cos_small, g = grad_rows(0.05)
    cos_large, _ = grad_rows(2.0)
    print(f"        mean pairwise cosine of centered gradient rows, support channel: "
          f"{cos_small:.4f} (energy spread 0.05), {cos_large:.4f} (spread 2.0)")
    print(f"        support rows sum {g[:N].sum():+.4f}, attack rows sum {g[N:].sum():+.4f} (opposite signs)")
    check("near-uniform energies: every support hypothesis gets the same option gradient",
          cos_small > 0.99 and cos_large < cos_small,
          cos_small < 0.5)


def main():

    check_entropy()
    check_sign_symmetry()
    check_validator_gradient()
    check_destructive_fraction()
    check_phase_gradient()
    check_binomial()
    check_selection_bias()
    check_overfitting()
    check_scale_invariance()
    check_ece_noise()
    check_pooled_mcnemar()
    check_cache_signal()
    check_common_mode_gradient()

    n_ok = sum(ok for _, ok in results)
    print(f"\nSUMMARY: {n_ok}/{len(results)} checks PASS "
          f"(each with its wrong-claim control failing as required)")

    if n_ok != len(results):
        for name, ok in results:
            if not ok:
                print(f"  FAILED: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
