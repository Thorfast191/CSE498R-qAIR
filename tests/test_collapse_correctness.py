"""
DoD §2.3 -- collapse correctness on a hand-built batch
(models/interference.py CoherentCollapse; docs/baseline/definition_of_done.md).

The expected values are computed with plain `math`, independently of the
module: r_k = exp(-E_k / 2T) (hypothesis amplitudes), t_kn = exp(-A_kn / 2t)
(compatibility amplitudes), w_kn = r_k t_kn. Normalization constants cancel
in the distribution over options, so only that distribution is compared.

  * probabilities over valid options sum to 1; padded options get no mass;
  * "classical" = sum_k r_k^2 t_kn^2;
  * "zero"      = (sum_k r_k t_kn)^2;
  * "learned"   = |sum_k r_k t_kn e^{i phi_kn}|^2, and with support phases 0
    and attack phases pi it equals (S_n - T_n)^2 (C1: documented, not changed);
  * hypothesis weights |c_k|^2 = r_k^2 normalized over valid hypotheses.

    python -m pytest tests/test_collapse_correctness.py
"""

import cmath
import math

import pytest
import torch
import torch.nn.functional as F

from config import EMBEDDING_DIM
from tests.test_forward_unchanged import cpu_build  # noqa: F401  (fixture)
from tests.test_gradient_coverage import ARMS

TOL = 1e-6
SUM_TOL = 1e-5
PAD_MASS = 1e-12

# Two questions, K = 4 hypotheses (2 support, 2 attack), N = 3 options.
# Question 1 has a padded option (n = 2) and a padded hypothesis (k = 3).
COLLAPSE_ENERGY = [[0.3, -0.5, 1.1, 0.2],
                   [-0.7, 0.4, 0.9, 0.0]]
ANSWER_ENERGY = [
    [[0.5, -1.0, 0.2], [1.5, 0.3, -0.4], [-0.2, 0.8, 1.0], [0.1, -0.6, 0.7]],
    [[-0.3, 0.9, 0.0], [0.6, -0.8, 0.0], [1.2, 0.1, 0.0], [0.0, 0.0, 0.0]],
]
H_MASK = [[True, True, True, True], [True, True, True, False]]
O_MASK = [[True, True, True], [True, True, False]]
HYP_TEMPERATURE = [0.8, 1.3]
PHASES = [[0.4, -1.2, 2.0, 0.7], [1.0, 0.3, -2.5, 0.0]]
SUPPORT_ATTACK = [0.0, 0.0, math.pi, math.pi]


def module(phase_mode):
    from models.interference import CoherentCollapse

    return CoherentCollapse(phase_mode=phase_mode)


def option_temperature(collapse):
    return 0.3 + F.softplus(collapse.option_temp).item()


def run(collapse, phase_h=None):
    with torch.no_grad():
        return collapse(
            torch.tensor(COLLAPSE_ENERGY),
            torch.tensor(ANSWER_ENERGY),
            phase_h=None if phase_h is None else torch.tensor(phase_h),
            H_mask=torch.tensor(H_MASK),
            O_mask=torch.tensor(O_MASK),
            hyp_temperature=torch.tensor(HYP_TEMPERATURE).view(2, 1),
        )


def amplitudes(b, t_opt):
    """w[k][n] = r_k t_kn over valid k, n of question b, by hand."""

    ks = [k for k, valid in enumerate(H_MASK[b]) if valid]
    ns = [n for n, valid in enumerate(O_MASK[b]) if valid]
    r = [math.exp(-COLLAPSE_ENERGY[b][k] / (2 * HYP_TEMPERATURE[b])) for k in ks]
    return ks, ns, [[r[i] * math.exp(-ANSWER_ENERGY[b][k][n] / (2 * t_opt)) for n in ns]
                    for i, k in enumerate(ks)]


def distribution(weights):
    total = sum(weights)
    return [w / total for w in weights]


def by_hand(b, t_opt, rule):
    ks, ns, w = amplitudes(b, t_opt)
    return distribution([rule(ks, [w[i][j] for i in range(len(ks))], n)
                         for j, n in enumerate(ns)])


def classical_rule(ks, column, n):
    return sum(x ** 2 for x in column)


def zero_rule(ks, column, n):
    return sum(column) ** 2


def phase_rule(phases):
    def rule(ks, column, n):
        return abs(sum(x * cmath.exp(1j * phases[k]) for k, x in zip(ks, column))) ** 2
    return rule


def probs(out, b):
    n = sum(O_MASK[b])
    return out["log_probs"][b].exp()[:n].tolist(), out["log_probs"][b].exp()[n:]


def assert_close(got, expected):
    assert got == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("phase_mode", ["classical", "zero", "learned"])
def test_valid_options_sum_to_one_and_padding_gets_no_mass(phase_mode):
    out = run(module(phase_mode), phase_h=PHASES)

    for b in range(2):
        valid, padded = probs(out, b)
        assert abs(sum(valid) - 1.0) <= SUM_TOL
        assert (padded <= PAD_MASS).all()


def test_classical_mode_is_the_decohered_mixture():
    collapse = module("classical")
    out = run(collapse, phase_h=PHASES)  # phases must be ignored

    for b in range(2):
        assert_close(probs(out, b)[0], by_hand(b, option_temperature(collapse), classical_rule))


def test_zero_mode_is_the_in_phase_coherent_sum():
    collapse = module("zero")
    out = run(collapse, phase_h=PHASES)  # phases must be ignored

    for b in range(2):
        assert_close(probs(out, b)[0], by_hand(b, option_temperature(collapse), zero_rule))


def test_learned_mode_is_the_coherent_sum_with_phases():
    collapse = module("learned")
    out = run(collapse, phase_h=PHASES)

    for b in range(2):
        expected = by_hand(b, option_temperature(collapse), phase_rule(PHASES[b]))
        assert_close(probs(out, b)[0], expected)


def test_support_attack_phases_give_the_squared_difference():
    """C1, documented not changed: at phases 0 / pi the readout is (S - T)^2."""

    collapse = module("learned")
    out = run(collapse, phase_h=[SUPPORT_ATTACK, SUPPORT_ATTACK])

    for b in range(2):
        ks, ns, w = amplitudes(b, option_temperature(collapse))
        support = [sum(w[i][j] for i, k in enumerate(ks) if k < 2) for j in range(len(ns))]
        attack = [sum(w[i][j] for i, k in enumerate(ks) if k >= 2) for j in range(len(ns))]
        expected = distribution([(s - a) ** 2 for s, a in zip(support, attack)])

        assert_close(probs(out, b)[0], expected)


def test_hypothesis_weights_are_normalized_over_valid_hypotheses():
    out = run(module("learned"), phase_h=PHASES)

    for b in range(2):
        ks = [k for k, valid in enumerate(H_MASK[b]) if valid]
        expected = distribution([
            math.exp(-COLLAPSE_ENERGY[b][k] / HYP_TEMPERATURE[b]) for k in ks
        ])
        got = out["collapse_probs"][b]

        assert_close(got[ks].tolist(), expected)
        assert (got[[k for k in range(4) if k not in ks]] <= PAD_MASS).all()


# ---------------------------------------------------------------------------
# The same two properties on the full model's scores, for every arm
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ragged_batch():
    g = torch.Generator().manual_seed(5)
    B, N = 3, 4
    K = 2 * N
    O_mask = torch.ones(B, N, dtype=torch.bool)
    O_mask[1, 3] = False                     # a 3-option question
    hyp_option = torch.arange(K) % N
    return {
        "H": torch.randn(B, K, EMBEDDING_DIM, generator=g),
        "O": torch.randn(B, N, EMBEDDING_DIM, generator=g),
        "Q": torch.randn(B, EMBEDDING_DIM, generator=g),
        "H_mask": O_mask[:, hyp_option],
        "O_mask": O_mask,
        "polarity": torch.where(torch.arange(K) < N, 1.0, -1.0).expand(B, K).clone(),
        "align": F.one_hot(hyp_option, N).float().expand(B, K, N).clone(),
        "llm_logprob": torch.randn(B, N, generator=g),
    }


@pytest.mark.parametrize("arm", ARMS)
def test_model_scores_are_a_distribution_over_valid_options(arm, ragged_batch, cpu_build):
    pytest.importorskip("pennylane")

    torch.manual_seed(0)
    model = cpu_build.build_model(cpu_build.ABLATIONS[arm]).eval()

    with torch.no_grad():
        p = model(**ragged_batch)["scores"].exp()

    mask = ragged_batch["O_mask"]
    assert ((p * mask).sum(dim=1) - 1.0).abs().max() <= SUM_TOL
    assert (p[~mask] <= PAD_MASS).all()
