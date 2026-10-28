"""
Reference baselines. Run these BEFORE trusting any qAIR number.

The v44 audit found the full 8.2M-parameter model scoring 0.3383 on
validation while a 30-line MLP over option embeddings alone scored
0.3585 -- the architecture was net-NEGATIVE against its own simplest
baseline, and nothing in the repo would have revealed that.

Three baselines and one cache-level check, cheapest first:

  --mode llm      The generator's own per-option log-likelihood, read
                  straight out of the cache. Free, since v45 stores it at
                  build time. THIS IS THE NUMBER THAT DECIDES WHETHER THE
                  PROJECT HAS A RESULT: if the pipeline cannot beat the
                  model it is built on top of, it is a lossy compression
                  of that model's knowledge and no reviewer will look
                  past it. It went unmeasured for the project's entire
                  history.

  --mode asymmetry
                  Not a baseline -- a check on the premise the whole v45
                  design rests on: that the attacking hypothesis for the
                  CORRECT option is systematically weaker than the attacks
                  on the distractors. Free, cache-level, no training and
                  no GPU. If that asymmetry is absent there is nothing for
                  the coherent collapse to discriminate on, and no amount
                  of architecture work reaches it -- the fix belongs in
                  the attack prompt in models/generator.py.

  --mode probe    Trainable probes on the cached embeddings. Establishes
                  the DATA ceiling: no architecture over this cache can
                  do much better, so a qAIR number below these is a
                  problem with qAIR, not with the task.

                  Since B7 the epoch count is chosen on a train holdout
                  and validation is evaluated once. Measured on the v44
                  cache before B7 (best epoch on validation, optimistic):
                      O only                      0.3585
                      diag(H_n, O_n)              0.3527
                      full pairwise H_k x O_n     0.3979
                      Q x O pairwise              0.4849   <- question!
                      Q x O + H                   0.4408   <- H HURT

  --mode direct   Re-scores the options with a freshly loaded LLM. Slow,
                  and should reproduce --mode llm exactly; kept as a
                  check that the cached scores are what they claim to be.

  --mode vote     Evidence-weighted majority vote over the cached
                  hypotheses (B5). Free, no parameters.

  --mode letter   LLM letter pick (B4): the generator sees the question and
                  all lettered options; the answer is the letter with the
                  highest next-token probability. Needs the LLM.

  --mode all      llm + asymmetry + probe + vote.

--out-dir writes the per-item records of llm, vote and letter as JSON
(evaluation/stats.py and the Step 7 tables read them).

    python -m evaluation.baselines --mode all --split validation
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import (
    CACHE_DIR,
    ARC_CONFIG,
    EMBEDDING_DIM as D,
    resolve_device,
)
from training.dataset import QAIRDataset
from training.seed import set_seed


# ==========================================================
# CACHED LLM BASELINE  (free)
# ==========================================================

def run_cached_llm_baseline(cache_dir, split="validation", max_samples=None,
                            arc_config=ARC_CONFIG):
    """
    argmax over the per-option log-likelihoods cached at build time.
    """

    ds = QAIRDataset(
        split=split, max_samples=max_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )

    correct = 0
    records = []

    for s in ds.samples:

        lp = s.get("llm_logprob")

        if lp is None or lp.numel() != len(s["options"]):
            continue

        hit = int(lp.argmax().item() == s["y"])

        correct += hit
        records.append(hit)

    acc = correct / max(len(records), 1)

    print(f"\n=== DIRECT LLM BASELINE (cached, {arc_config} {split}) ===")
    print(f"  length-normalized option likelihood : {acc:.4f}  "
          f"({correct}/{len(records)})")
    print(
        "\n  If this beats the trained qAIR pipeline, the architecture is a\n"
        "  lossy compression of knowledge the generator already had. That is\n"
        "  the first thing a reviewer will check."
    )

    return acc, records


# ==========================================================
# EVIDENCE-WEIGHTED MAJORITY VOTE  (B5; free, no parameters)
# ==========================================================

def evidence_vote(sample):
    """
    Pre-registered in docs/baseline/issue_dossier.md B5 (2026-10-17),
    before any ARC-Easy result existed:

        v_n = sum_k [hyp_option_k == n] * polarity_k * cos(H_k, O_n)

    i.e. support evidence for option n minus attack evidence against it,
    every hypothesis included (fallbacks too: qAIR sees them as well).
    With one support and one attack per option this is the cache gate's
    "support - attack margin" statistic. A plain count vote is degenerate
    -- sum_k polarity_k [hyp_option_k == n] = 0 for every option -- so the
    vote is weighted by evidence. Prediction: argmax_n v_n; ties go to the
    lowest index (torch.argmax returns the first maximum).
    """

    H = F.normalize(sample["H"].float(), dim=-1)
    O = F.normalize(sample["O"].float(), dim=-1)
    owner = sample["hyp_option"]

    cos = (H * O[owner]).sum(-1)                        # cos(H_k, O_{option of k})
    votes = torch.zeros(O.shape[0])
    votes.index_add_(0, owner, sample["polarity"].float() * cos)
    return votes


def run_vote(cache_dir, split="validation", max_samples=None, arc_config=ARC_CONFIG):
    """Evidence-weighted vote (B5) and, for reference, the degenerate count vote."""

    ds = QAIRDataset(split=split, max_samples=max_samples, cache_dir=cache_dir,
                     arc_config=arc_config)

    predictions, records, count_records = [], [], []
    for s in ds.samples:
        votes = evidence_vote(s)
        pred = int(votes.argmax().item())
        predictions.append(pred)
        records.append(int(pred == s["y"]))

        counts = torch.zeros(len(s["options"]))
        counts.index_add_(0, s["hyp_option"], s["polarity"].float())
        count_records.append(int(int(counts.argmax().item()) == s["y"]))

    acc = sum(records) / max(len(records), 1)
    count_acc = sum(count_records) / max(len(count_records), 1)
    chance = sum(1 / len(s["options"]) for s in ds.samples) / max(len(ds.samples), 1)

    print(f"\n=== EVIDENCE-WEIGHTED MAJORITY VOTE (B5, {arc_config} {split}) ===")
    print(f"  support - attack cosine vote : {acc:.4f}  ({sum(records)}/{len(records)})  "
          f"chance {chance:.4f}")
    print(f"  plain count vote (degenerate: all ties, picks option 0) : {count_acc:.4f}")

    return acc, records, predictions


# ==========================================================
# LLM LETTER PICK  (B4; needs the LLM)
# ==========================================================

LETTERS = "ABCDEFGH"

# Fixed before any test-split result (B4: "fix the prompt before seeing test").
LETTER_PROMPT = (
    "Answer the following multiple-choice question. Reply with the letter "
    "of the correct option only.\n\n"
    "Question: {question}\n"
    "{options}\n"
    "Answer:"
)


def letter_prompt(tokenizer, question, options):
    """The chat-formatted prompt; the next token is the answer letter."""

    lettered = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(options))
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": LETTER_PROMPT.format(question=question, options=lettered)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def letter_token_ids(tokenizer, n):
    """Token id of each of the first n letters; each must be a single token."""

    ids = []
    for letter in LETTERS[:n]:
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"letter {letter!r} is {len(tokens)} tokens for this tokenizer")
        ids.append(tokens[0])
    return ids


@torch.no_grad()
def letter_pick_scores(model, tokenizer, questions, options_list, device, batch_size=16):
    """
    log P(letter | prompt) at the answer position, for each option's letter.
    Left padding, so the last position is the answer position in every row.
    """

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    scores = []

    try:
        for start in range(0, len(questions), batch_size):
            qs = questions[start:start + batch_size]
            opts = options_list[start:start + batch_size]
            enc = tokenizer([letter_prompt(tokenizer, q, o) for q, o in zip(qs, opts)],
                            return_tensors="pt", padding=True).to(device)
            logprobs = torch.log_softmax(model(**enc).logits[:, -1].float(), dim=-1)
            for row, options in enumerate(opts):
                ids = torch.tensor(letter_token_ids(tokenizer, len(options)), device=logprobs.device)
                scores.append(logprobs[row, ids].cpu())
    finally:
        tokenizer.padding_side = previous_side

    return scores


class GeneratorMismatchError(RuntimeError):
    """The loaded LLM is not the model / revision the cache was generated with."""


def run_letter_pick(cache_dir, split="validation", max_samples=None, arc_config=ARC_CONFIG,
                    limit=None, generator=None):
    """
    B4. The generator's own model and revision: refuses when the cache
    header records a different generator or revision.
    """

    from models.generator import HypothesisGenerator

    ds = QAIRDataset(split=split, max_samples=max_samples, cache_dir=cache_dir,
                     arc_config=arc_config)
    samples = ds.samples[:limit] if limit else ds.samples

    gen = generator or HypothesisGenerator(device=resolve_device())
    loaded, cached = gen.provenance(), ds.provenance or {}
    for key in ("generator", "generator_revision"):
        if cached.get(key) and loaded.get(key) and cached[key] != loaded[key]:
            raise GeneratorMismatchError(
                f"{key}: cache {cached[key]!r}, loaded {loaded[key]!r}")

    scores = letter_pick_scores(gen.model, gen.tokenizer, [s["question"] for s in samples],
                                [s["options"] for s in samples], gen.device)

    predictions = [int(sc.argmax().item()) for sc in scores]
    records = [int(p == s["y"]) for p, s in zip(predictions, samples)]
    acc = sum(records) / max(len(records), 1)

    print(f"\n=== LLM LETTER PICK (B4, {loaded.get('generator')}, {arc_config} {split}) ===")
    print(f"  letter of max next-token probability : {acc:.4f}  ({sum(records)}/{len(records)})")

    return acc, records, predictions


def write_records(out_dir, mode, split, arc_config, acc, records, predictions=None):
    """{out_dir}/{mode}_{split}.json with the per-item records."""

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{mode}_{split}.json")
    with open(path, "w") as f:
        json.dump({"mode": mode, "split": split, "arc_config": arc_config, "accuracy": acc,
                   "n": len(records), "records": records, "predictions": predictions},
                  f, indent=1)
        f.write("\n")
    print(f"  records: {path}")
    return path


# ==========================================================
# SUPPORT / ATTACK ASYMMETRY  (free, cache-level)
# ==========================================================

def run_asymmetry(cache_dir, split="validation", max_samples=None,
                  arc_config=ARC_CONFIG):
    """
    Measures the premise the v45 evidence design rests on: that the LLM
    objects to a genuinely wrong option more specifically than to a
    correct one, so the attacking channel carries an asymmetry the
    coherent collapse can discriminate on.

    The statistic is cos(H_k, O_{hyp_option[k]}) -- how closely each
    hypothesis tracks the option it was generated to argue about -- split
    by whether that option is the correct one. Supports are reported as
    the reference: a gap that appears in BOTH channels is a property of
    the options and the encoder, not of the attack prompt.

    Fallback rates come with it because a templated fallback is generic
    text carrying no evidence about any option, so if fallbacks
    concentrate in one group the cosine gap is partly measuring them.

    With the hypothesis phases at 0 and pi the coherent sum reduces to
    (sum_support w - sum_attack w)^2, so an attack gap indistinguishable
    from zero means the collapse has nothing to discriminate on -- and the
    fix is in the attack prompt in models/generator.py, not in the model.
    """

    ds = QAIRDataset(
        split=split, max_samples=max_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )

    # group -> is-the-correct-option -> per-hypothesis values
    cos = {"attack": {True: [], False: []}, "support": {True: [], False: []}}
    fell_back = {"attack": {True: [], False: []}, "support": {True: [], False: []}}

    n_samples = 0

    for s in ds.samples:

        pol = s.get("polarity")
        opt = s.get("hyp_option")

        if pol is None or opt is None:
            continue

        H = F.normalize(s["H"].float(), dim=-1)
        O = F.normalize(s["O"].float(), dim=-1)

        flags = s["is_fallback"]

        # Each hypothesis against the option it argues about, not against
        # every option -- alignment is positional in a v45 cache, so this
        # pairing is exactly the one generation used.
        d = (H * O[opt]).sum(-1)

        n_samples += 1

        for k in range(H.shape[0]):

            group = "attack" if pol[k].item() < 0 else "support"
            on_correct = bool(opt[k].item() == s["y"])

            cos[group][on_correct].append(d[k].item())
            fell_back[group][on_correct].append(float(flags[k].item()))

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    print(f"\n=== SUPPORT/ATTACK ASYMMETRY (cached, {arc_config} {split}) ===")
    print(f"  {n_samples} samples, split by whether the hypothesis argues "
          f"about the correct option")

    out = {}

    for key, header, table in (
        ("cos", "cos(H_k, O_hyp_option[k])", cos),
        ("fallback", "fallback rate", fell_back),
    ):

        print(f"\n  {header}")
        print(f"    {'':<9s}{'correct':>10s}{'distractor':>12s}{'gap':>10s}"
              f"{'n correct':>12s}{'n distractor':>14s}")

        for group in ("attack", "support"):

            hit = table[group][True]
            miss = table[group][False]

            c, w = mean(hit), mean(miss)

            print(f"    {group:<9s}{c:>10.4f}{w:>12.4f}{c - w:>+10.4f}"
                  f"{len(hit):>12d}{len(miss):>14d}")

            out[f"{group}_{key}_correct"] = c
            out[f"{group}_{key}_distractor"] = w
            out[f"{group}_{key}_gap"] = c - w

    print(
        "\n  The attack rows carry the claim; the support rows are the "
        "reference, since a\n  gap present in both is a property of the "
        "options rather than of the attack\n  prompt. With the hypothesis "
        "phases at 0 and pi the coherent sum reduces to\n  (sum_support w - "
        "sum_attack w)^2, so an attack gap indistinguishable from zero\n"
        "  means the coherent collapse has no amplitude asymmetry to work "
        "with."
    )

    return out


# ==========================================================
# PROBES ON CACHED EMBEDDINGS  (the data ceiling)
# ==========================================================

def _stack(ds, n_opt=4):
    """
    Returns (H_sup, H_att, O, Q, llm, y) for the samples with exactly
    n_opt options and the v45 two-channel hypothesis layout.
    """

    rows = [
        s for s in ds.samples
        if s["O"].shape[0] == n_opt
        and s["H"].shape[0] == 2 * n_opt
        and "Q" in s
    ]

    if not rows:
        raise RuntimeError(
            f"no {n_opt}-option samples with support+attack hypotheses found "
            f"-- is this a v45 cache?"
        )

    H = torch.stack([r["H"].float() for r in rows])

    return (
        H[:, :n_opt],
        H[:, n_opt:],
        torch.stack([r["O"].float() for r in rows]),
        torch.stack([r["Q"].float() for r in rows]),
        torch.stack([r["llm_logprob"].float() for r in rows]),
        torch.tensor([r["y"] for r in rows]),
    )


class Probe(nn.Module):

    def __init__(self, n_feat, n_scalar=0):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_feat * D + n_scalar, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, X):
        return self.f(X).squeeze(-1)


def _qexp(Q, O):
    return Q.unsqueeze(1).expand_as(O)


FEATURES = {
    "O only": (1, 0, lambda sup, att, O, Q, lp: O),

    "Q x O": (4, 0, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O], -1)),

    "Q x O + support": (6, 0, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O, sup, sup * O], -1)),

    "Q x O + support + attack": (8, 0, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O,
         sup, sup * O, att, att * O], -1)),

    "Q x O + evidence + llm": (8, 1, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O,
         sup, sup * O, att, att * O, _zscore(lp).unsqueeze(-1)], -1)),
}


def _zscore(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)


PROBE_HOLDOUT = 0.2      # fraction of train held out to choose the epoch count (B7)
PROBE_SPLIT_SEED = 0


def _probe_epoch(model, opt, X, y):
    model.train()
    perm = torch.randperm(y.numel())
    for i in range(0, perm.numel(), 128):
        b = perm[i:i + 128]
        loss = F.cross_entropy(model(X[b]), y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()


def _accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def _fresh_probe(n_feat, n_scalar, lr):
    set_seed(0)
    model = Probe(n_feat, n_scalar)
    return model, torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)


def train_holdout_split(n):
    """(fit, holdout) indices of the train split; fixed, independent of validation."""

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(PROBE_SPLIT_SEED))
    n_hold = max(1, int(round(PROBE_HOLDOUT * n)))
    return perm[n_hold:], perm[:n_hold]


def fit_probe(n_feat, n_scalar, Xtr, ytr, Xva, yva, epochs, lr):
    """
    B7. The epoch count is chosen on a holdout of the TRAIN split (first
    best), the probe is retrained from the same seed on all of train for
    that many epochs, and validation is evaluated exactly once. Before B7
    the reported number was the best validation accuracy over 60 epochs,
    which selects on the evaluated items (optimistic, like B3).
    """

    fit, hold = train_holdout_split(ytr.numel())
    model, opt = _fresh_probe(n_feat, n_scalar, lr)
    holdout = []
    for _ in range(epochs):
        _probe_epoch(model, opt, Xtr[fit], ytr[fit])
        holdout.append(_accuracy(model, Xtr[hold], ytr[hold]))
    best_epochs = holdout.index(max(holdout)) + 1

    model, opt = _fresh_probe(n_feat, n_scalar, lr)
    for _ in range(best_epochs):
        _probe_epoch(model, opt, Xtr, ytr)

    return {"val_acc": _accuracy(model, Xva, yva), "epochs": best_epochs,
            "holdout_acc": holdout[best_epochs - 1]}


def run_probes(cache_dir, epochs=60, lr=3e-3, train_samples=None,
               val_samples=None, arc_config=ARC_CONFIG):
    """
    train_samples/val_samples MUST match whatever cap the cache was built
    with. A cache built with max_samples is stored complete=False by
    design (so a later full build can resume and finish the split), which
    means calling QAIRDataset with max_samples=None against it does not
    read 80 samples -- it starts generating the rest of the split.

    epochs is the maximum; fit_probe chooses the count on a train holdout.
    """

    tr = _stack(QAIRDataset(
        split="train", max_samples=train_samples, cache_dir=cache_dir,
        arc_config=arc_config))
    va = _stack(QAIRDataset(
        split="validation", max_samples=val_samples, cache_dir=cache_dir,
        arc_config=arc_config))

    print(f"\ntrain={tr[5].numel()}  val={va[5].numel()}  chance=0.2500")
    print(f"\n=== DATA CEILING (probes on cached embeddings; epochs chosen on a "
          f"{PROBE_HOLDOUT:.0%} train holdout, validation evaluated once) ===")

    out = {}

    for name, (n_feat, n_scalar, fn) in FEATURES.items():

        result = fit_probe(n_feat, n_scalar, fn(*tr[:5]), tr[5], fn(*va[:5]), va[5],
                           epochs, lr)
        out[name] = result
        print(f"  {name:<28s} val acc = {result['val_acc']:.4f}  "
              f"(epochs {result['epochs']}, holdout {result['holdout_acc']:.4f})")

    print(
        "\n  'Q x O + support' vs 'Q x O' is the question the whole project "
        "turns on:\n  does the generated evidence add anything to the "
        "question and the options?\n  On the v44 cache it was NEGATIVE "
        "(0.4849 -> 0.4408, best-epoch-on-validation numbers, before B7)."
    )

    return out


# ==========================================================
# LIVE LLM BASELINE  (verification of the cached scores)
# ==========================================================

@torch.no_grad()
def run_direct(cache_dir, split="validation", limit=None, max_samples=None,
               arc_config=ARC_CONFIG):
    """
    Scores each option by its mean token log-likelihood under the same
    LLM used for hypothesis generation, conditioned on the question.
    Length-normalized so long options aren't penalized.

    Should reproduce run_cached_llm_baseline; it exists so the cached
    numbers can be checked rather than trusted.
    """

    from models.generator import HypothesisGenerator, LLM_NAME

    gen = HypothesisGenerator(device=resolve_device())

    ds = QAIRDataset(
        split=split, max_samples=max_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )

    samples = ds.samples[:limit] if limit else ds.samples

    correct = 0
    records = []

    batch = 8

    for i in tqdm(range(0, len(samples), batch), desc=f"{LLM_NAME} direct"):

        chunk = samples[i:i + batch]

        scored = gen.score_options(
            [s["question"] for s in chunk],
            [s["options"] for s in chunk],
        )

        for s, scores in zip(chunk, scored):
            hit = int(max(range(len(scores)), key=lambda j: scores[j]) == s["y"])
            correct += hit
            records.append(hit)

    acc = correct / max(len(samples), 1)

    print(f"\n=== DIRECT LLM BASELINE (live, {LLM_NAME}, {split}) ===")
    print(f"  length-normalized option likelihood : {acc:.4f}  "
          f"({correct}/{len(samples)})")

    return acc, records


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["llm", "asymmetry", "probe", "direct", "vote", "letter", "all"],
        default="all",
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument("--arc-config", type=str, default=ARC_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--train-samples", type=int, default=None,
        help="Must match the cap the train cache was built with.",
    )
    parser.add_argument(
        "--val-samples", type=int, default=None,
        help="Must match the cap the validation cache was built with.",
    )
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Write per-item records of llm / vote / letter as JSON.")
    args = parser.parse_args()

    set_seed(0)

    max_for_split = (
        args.val_samples if args.split == "validation" else args.train_samples
    )

    def save(mode, acc, records, predictions=None):
        if args.out_dir:
            write_records(args.out_dir, mode, args.split, args.arc_config, acc, records,
                          predictions)

    if args.mode in ("llm", "all"):
        save("llm", *run_cached_llm_baseline(
            args.cache_dir, split=args.split, max_samples=max_for_split,
            arc_config=args.arc_config,
        ))

    if args.mode in ("vote", "all"):
        save("vote", *run_vote(
            args.cache_dir, split=args.split, max_samples=max_for_split,
            arc_config=args.arc_config,
        ))

    if args.mode == "letter":
        save("letter", *run_letter_pick(
            args.cache_dir, split=args.split, max_samples=max_for_split,
            arc_config=args.arc_config, limit=args.limit,
        ))

    if args.mode in ("asymmetry", "all"):
        run_asymmetry(
            args.cache_dir, split=args.split, max_samples=max_for_split,
            arc_config=args.arc_config,
        )

    if args.mode in ("probe", "all"):
        run_probes(
            args.cache_dir,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            arc_config=args.arc_config,
        )

    if args.mode == "direct":
        run_direct(
            args.cache_dir, split=args.split, limit=args.limit,
            max_samples=max_for_split, arc_config=args.arc_config,
        )


if __name__ == "__main__":
    main()
