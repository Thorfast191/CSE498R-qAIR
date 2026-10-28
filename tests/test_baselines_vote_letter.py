"""
B5 -- evidence-weighted majority vote; B4 -- LLM letter pick
(docs/baseline/issues.md B4, B5; pre-registration in issue_dossier.md).

    python -m pytest tests/test_baselines_vote_letter.py
    QAIR_SLOW=1 python -m pytest tests/test_baselines_vote_letter.py   # real Qwen

The fast tests use hand-built samples, the fake cache of
tests/test_checkpoint_provenance and a character-level fake tokenizer /
model; the slow test downloads Qwen2.5-0.5B-Instruct.
"""

import json
import os
import re

import pytest
import torch

from evaluation import baselines
from tests.test_cache_determinism import ARC
from tests.test_checkpoint_provenance import smoke_cache  # noqa: F401  (fixture)


# ---------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------

def hand_sample(sup_cos, att_cos, y):
    """3 options; H_k is built to have the given cosine with its option."""

    n, dim = len(sup_cos), 8
    O = torch.eye(dim)[:n]
    other = torch.eye(dim)[n]                       # orthogonal to every option

    def at(cosine, option):
        return cosine * O[option] + (1 - cosine ** 2) ** 0.5 * other

    H = torch.stack([at(c, i) for i, c in enumerate(sup_cos)]
                    + [at(c, i) for i, c in enumerate(att_cos)])
    return {"H": H, "O": O, "y": y, "options": ["a", "b", "c"],
            "hyp_option": torch.tensor([0, 1, 2, 0, 1, 2]),
            "polarity": torch.tensor([1.0, 1, 1, -1, -1, -1])}


def test_vote_is_support_minus_attack_cosine():
    sample = hand_sample(sup_cos=[0.2, 0.9, 0.5], att_cos=[0.1, 0.3, 0.6], y=1)

    votes = baselines.evidence_vote(sample)

    assert torch.allclose(votes, torch.tensor([0.1, 0.6, -0.1]), atol=1e-6)
    assert int(votes.argmax()) == 1


def test_vote_does_not_depend_on_hypothesis_order():
    sample = hand_sample(sup_cos=[0.2, 0.9, 0.5], att_cos=[0.1, 0.3, 0.6], y=1)
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    shuffled = dict(sample, H=sample["H"][order], hyp_option=sample["hyp_option"][order],
                    polarity=sample["polarity"][order])

    assert torch.allclose(baselines.evidence_vote(shuffled), baselines.evidence_vote(sample))


def test_ties_go_to_the_lowest_index():
    sample = hand_sample(sup_cos=[0.5, 0.5, 0.5], att_cos=[0.2, 0.2, 0.2], y=2)
    assert int(baselines.evidence_vote(sample).argmax()) == 0


def test_vote_matches_the_cache_gate_margin(smoke_cache, capsys):
    """On the mode-major cache the vote is the gate's support - attack margin."""

    acc, records, predictions = baselines.run_vote(str(smoke_cache), arc_config=ARC)
    out = capsys.readouterr().out

    gate = float(re.findall(r"support - attack margin\s+([0-9.]+)", out)[0])
    assert acc == pytest.approx(gate, abs=1e-4)
    assert len(records) == len(predictions) > 0
    assert "degenerate" in out


# ---------------------------------------------------------------------------
# B4 (fake tokenizer / model)
# ---------------------------------------------------------------------------

PAD = 0


class CharTokenizer:
    """One token per character (id = ord); left/right padding with PAD."""

    padding_side = "right"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, texts, return_tensors, padding):
        ids = [self.encode(t) for t in texts]
        width = max(map(len, ids))
        rows, mask = [], []
        for seq in ids:
            pad = [PAD] * (width - len(seq))
            rows.append(pad + seq if self.padding_side == "left" else seq + pad)
            m = [0] * len(pad)
            mask.append(m + [1] * len(seq) if self.padding_side == "left" else [1] * len(seq) + m)
        return Batch(input_ids=torch.tensor(rows), attention_mask=torch.tensor(mask))


class Batch(dict):
    def to(self, device):
        return self


class Output:
    def __init__(self, logits):
        self.logits = logits


class RightAnswerModel:
    """
    At each row's last real token only, puts its mass on the letter of the
    option whose text is RIGHT; everywhere else on 'A'. Reading any other
    position (e.g. a pad position under right padding) gives 'A'.
    """

    def __call__(self, input_ids, attention_mask):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, 128)
        logits[:, :, ord("A")] = 5.0
        for b in range(B):
            text = "".join(chr(int(t)) for t in input_ids[b] if t != PAD)
            letter = re.search(r"\n([A-H])\. RIGHT\n", text + "\n").group(1)
            last = int(attention_mask[b].nonzero().max())
            logits[b, last] = 0.0
            logits[b, last, ord(letter)] = 10.0
        return Output(logits)


def test_letter_pick_reads_the_answer_position():
    tokenizer = CharTokenizer()
    questions = ["Short?", "A much longer question that forces padding?"]
    options = [["wrong", "wrong", "RIGHT"], ["RIGHT", "wrong", "wrong", "wrong"]]

    scores = baselines.letter_pick_scores(RightAnswerModel(), tokenizer, questions, options, "cpu")

    assert [int(s.argmax()) for s in scores] == [2, 0]
    assert [len(s) for s in scores] == [3, 4]
    assert tokenizer.padding_side == "right"         # restored


def test_prompt_letters_every_option():
    text = baselines.letter_prompt(CharTokenizer(), "Why?", ["x", "y", "z"])
    assert "Question: Why?\nA. x\nB. y\nC. z\nAnswer:" in text


def test_multi_token_letter_is_refused():
    class Tok(CharTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [1, 2] if text == "C" else [ord(text)]

    with pytest.raises(ValueError, match="'C' is 2 tokens"):
        baselines.letter_token_ids(Tok(), 4)


class FakeLetterGenerator:
    def __init__(self, revision="fake-rev", name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model, self.tokenizer, self.device = AlwaysC(), CharTokenizer(), "cpu"
        self.revision, self.name = revision, name

    def provenance(self):
        return {"generator": self.name, "generator_revision": self.revision}


class AlwaysC:
    def __call__(self, input_ids, attention_mask):
        logits = torch.zeros(*input_ids.shape, 128)
        logits[:, -1, ord("C")] = 1.0
        return Output(logits)


def test_run_letter_pick_records(smoke_cache, tmp_path):
    from training.dataset import QAIRDataset

    ys = [s["y"] for s in QAIRDataset(split="validation", cache_dir=str(smoke_cache),
                                       arc_config=ARC).samples]
    cached = QAIRDataset(split="validation", cache_dir=str(smoke_cache), arc_config=ARC).provenance

    acc, records, predictions = baselines.run_letter_pick(
        str(smoke_cache), arc_config=ARC,
        generator=FakeLetterGenerator(revision=cached["generator_revision"],
                                      name=cached["generator"]))

    assert predictions == [2] * len(ys)
    assert records == [int(y == 2) for y in ys]

    path = baselines.write_records(str(tmp_path), "letter", "validation", ARC, acc, records,
                                   predictions)
    with open(path) as f:
        saved = json.load(f)
    assert saved["records"] == records and saved["predictions"] == predictions


@pytest.mark.parametrize("field", ["generator", "generator_revision"])
def test_other_generator_or_revision_is_refused(smoke_cache, field):
    from training.dataset import QAIRDataset

    cached = QAIRDataset(split="validation", cache_dir=str(smoke_cache), arc_config=ARC).provenance
    generator = FakeLetterGenerator(revision=cached["generator_revision"], name=cached["generator"])
    if field == "generator":
        generator.name = "Qwen/Qwen2.5-7B-Instruct"
    else:
        generator.revision = "another-rev"

    with pytest.raises(baselines.GeneratorMismatchError, match=f"^{field}:"):
        baselines.run_letter_pick(str(smoke_cache), arc_config=ARC, generator=generator)


# ---------------------------------------------------------------------------
# B4 with the real model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("QAIR_SLOW"),
                    reason="set QAIR_SLOW=1 (downloads Qwen2.5-0.5B-Instruct, ARC-Easy)")
def test_real_letter_pick_is_deterministic_on_20_items():
    from benchmarks.arc import load_arc
    from models.generator import HypothesisGenerator

    gen = HypothesisGenerator(device="cpu")
    assert len(set(baselines.letter_token_ids(gen.tokenizer, len(baselines.LETTERS)))) == 8

    rows = load_arc("ARC-Easy")["validation"].select(range(20))
    questions = [r["question"] for r in rows]
    options = [r["choices"]["text"] for r in rows]

    first = baselines.letter_pick_scores(gen.model, gen.tokenizer, questions, options, "cpu")
    second = baselines.letter_pick_scores(gen.model, gen.tokenizer, questions, options, "cpu")

    assert all(torch.equal(a, b) for a, b in zip(first, second))
    print("letter-pick predictions:", [int(s.argmax()) for s in first])
