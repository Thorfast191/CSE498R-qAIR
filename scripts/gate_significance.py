"""
Significance of cache-quality gate accuracies against chance (D1, D5).

    python scripts/gate_significance.py --split train \\
        --acc "support agreement=0.3350" --acc "attack disagreement=0.1804"

Chance differs per question (1 / number of options), so under "no signal"
the number of hits follows a Poisson-binomial distribution; the p-value
is exact and two-sided. Reads only the option counts of the requested ARC
split; refuses the test split (it is never gated, CLAUDE.md rule 5).
"""

import argparse
import math


def poisson_binomial(ps):
    """P(X = k) for k = 0..len(ps), X a sum of independent Bernoulli(p_i)."""

    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            new[k] += v * (1 - p)
            new[k + 1] += v * p
        dist = new
    return dist


def significance(acc, ps):
    """Hits, binomial SE, z and exact two-sided p of an accuracy vs chance ps."""

    n = len(ps)
    hits = round(acc * n)
    dist = poisson_binomial(ps)
    p_two = min(1.0, 2 * min(sum(dist[:hits + 1]), sum(dist[hits:])))
    sd0 = math.sqrt(sum(p * (1 - p) for p in ps))
    return {"hits": hits, "n": n, "acc": acc, "chance": sum(ps) / n,
            "se": math.sqrt(acc * (1 - acc) / n), "z": (hits - sum(ps)) / sd0, "p": p_two}


def option_counts(arc_config, split):
    from datasets import load_dataset

    if split == "test":
        raise ValueError("the test split is never gated (CLAUDE.md rule 5)")
    return [len(r["choices"]["text"]) for r in load_dataset("allenai/ai2_arc", arc_config,
                                                            split=split)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arc-config", default="ARC-Easy")
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    parser.add_argument("--acc", action="append", required=True, help='"name=accuracy"')
    args = parser.parse_args(argv)

    ps = [1 / c for c in option_counts(args.arc_config, args.split)]
    print(f"{args.arc_config} {args.split}: n={len(ps)}, chance {sum(ps) / len(ps):.4f}")
    for item in args.acc:
        name, value = item.rsplit("=", 1)
        r = significance(float(value), ps)
        print(f"  {name:<25s} {r['acc']:.4f} = {r['hits']}/{r['n']}  SE {r['se']:.4f}  "
              f"95% CI [{r['acc'] - 1.96 * r['se']:.4f}, {r['acc'] + 1.96 * r['se']:.4f}]  "
              f"z {r['z']:+.2f}  exact two-sided p {r['p']:.2e}")


if __name__ == "__main__":
    main()
