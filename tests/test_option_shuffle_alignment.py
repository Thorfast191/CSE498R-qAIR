"""
DoD §2.6 -- `collate_fn` with option shuffling keeps every option,
hypothesis and label aligned (training/dataset.py collate_fn).

Every tensor of a hand-built sample carries a code saying which original
option (and mode) it belongs to, so after shuffling each alignment can be
read back and checked. Layout as the cache writes it: mode-major,
K = len(modes) * N, hyp_option[b*N + i] = i (models/generator.py
generate_batch_with_flags).

    python -m pytest tests/test_option_shuffle_alignment.py
"""

import pytest
import torch
from torch.utils.data import DataLoader

from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)
from training.dataset import collate_fn

DIM = 8
MODES = 2           # support, attack
TRIALS = 50

# Embedding slots that carry the codes.
OPTION, MODE, FALLBACK = 0, 1, 2


def option_code(n):
    return 10.0 + n


def llm_code(n):
    return -0.1 * (n + 1)


def sample(n_options, y):
    k = MODES * n_options
    hyp_option = torch.arange(n_options).repeat(MODES)
    mode = torch.arange(k) // n_options
    fallback = (hyp_option + mode) % 3 == 0

    O = torch.zeros(n_options, DIM)
    O[:, OPTION] = option_code(torch.arange(n_options).float())

    H = torch.zeros(k, DIM)
    H[:, OPTION] = option_code(hyp_option.float())
    H[:, MODE] = mode.float()
    H[:, FALLBACK] = fallback.float()

    return {
        "H": H,
        "O": O,
        "Q": torch.full((DIM,), 7.0),
        "y": y,
        "hyp_option": hyp_option,
        "polarity": torch.where(mode == 0, 1.0, -1.0),
        "is_fallback": fallback,
        "llm_logprob": llm_code(torch.arange(n_options).float()),
    }


def ragged_batch():
    """3-, 4- and 5-option questions, as in ARC."""
    return [sample(4, 2), sample(3, 0), sample(5, 4), sample(4, 3)]


def check_alignment(batch, originals):

    for b, original in enumerate(originals):
        n = original["O"].shape[0]
        k = original["H"].shape[0]

        O = batch["O"][b]
        H = batch["H"][b]
        align = batch["align"][b]

        assert batch["O_mask"][b].sum() == n and batch["H_mask"][b].sum() == k
        assert torch.equal(align[k:], torch.zeros_like(align[k:]))
        assert torch.equal(align[:k, n:], torch.zeros_like(align[:k, n:]))
        assert torch.equal(align[:k].sum(dim=1), torch.ones(k))

        # Same options, same hypotheses, possibly reordered.
        assert sorted(O[:n, OPTION].tolist()) == sorted(original["O"][:, OPTION].tolist())
        assert sorted(map(tuple, H[:k].tolist())) == sorted(map(tuple, original["H"].tolist()))

        # Option <-> its LLM score.
        codes = O[:n, OPTION] - 10.0
        assert torch.allclose(batch["llm_logprob"][b, :n], llm_code(codes))

        # Label still points at the same option.
        assert O[batch["y"][b], OPTION] == option_code(original["y"])

        # Hypothesis <-> the option it is about, its polarity and fallback flag.
        about = align[:k].argmax(dim=1)
        assert torch.equal(O[about, OPTION], H[:k, OPTION])
        assert torch.equal(batch["polarity"][b, :k], torch.where(H[:k, MODE] == 0, 1.0, -1.0))
        assert torch.equal(batch["is_fallback"][b, :k], H[:k, FALLBACK].bool())

        # Mode-major layout survives: block m, row i is about option i.
        assert torch.equal(about, torch.arange(n).repeat(MODES))


def test_unshuffled_batch_is_aligned():
    originals = ragged_batch()
    check_alignment(collate_fn(originals, shuffle_options=False), originals)


def test_shuffled_batches_stay_aligned():
    originals = ragged_batch()
    torch.manual_seed(0)
    moved = 0

    for _ in range(TRIALS):
        batch = collate_fn(originals, shuffle_options=True)
        check_alignment(batch, originals)
        moved += sum(
            int(batch["y"][b] != original["y"]) for b, original in enumerate(originals)
        )

    assert moved > 0, "shuffling never moved a label: the test checked nothing"


@pytest.fixture
def cached_samples(smoke_cache):
    from training.dataset import QAIRDataset

    ds = QAIRDataset(split="train", max_samples=16, cache_dir=str(smoke_cache),
                     arc_config=ARC)
    return [ds[i] for i in range(len(ds))]


def test_cached_samples_stay_aligned(cached_samples):
    """The cache's own layout: hypotheses about the correct option are kept."""

    def correct_option_rows(batch, b):
        about = batch["align"][b].argmax(dim=1)[batch["H_mask"][b]]
        rows = batch["H"][b][batch["H_mask"][b]][about == batch["y"][b]]
        return sorted(map(tuple, rows.tolist()))

    plain = collate_fn(cached_samples, shuffle_options=False)
    torch.manual_seed(0)
    shuffled = next(iter(DataLoader(cached_samples, batch_size=len(cached_samples),
                                    collate_fn=collate_fn)))

    for b, s in enumerate(cached_samples):
        n = s["O"].shape[0]
        assert s["H"].shape[0] == MODES * n
        assert torch.equal(s["hyp_option"], torch.arange(n).repeat(MODES))

        assert torch.equal(shuffled["O"][b, shuffled["y"][b]], s["O"][s["y"]])
        assert correct_option_rows(shuffled, b) == correct_option_rows(plain, b)
