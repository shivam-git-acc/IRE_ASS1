"""
Q4 offline evaluation harness.

Accuracy metrics (per impression, averaged): AUC, MRR, nDCG@5, nDCG@10.
Beyond-accuracy (per ranked list): intra-list diversity, novelty, coverage.
Bootstrap 95% CIs for each metric. Head/tail (and cold/warm) slicing.

An "impression result" is (scores, labels) for the candidates of one impression,
where labels are 0/1 (clicked). Beyond-accuracy metrics additionally take the ranked
candidate ids plus article metadata (category embeddings / popularity) as needed.

Reads the per-impression (scores, labels) cache written by `src.rerank`:

  python -m src.rerank   --dataset mind
  python -m src.evaluate --dataset mind
"""
import argparse
import os
import pickle

import numpy as np


# ---- accuracy metrics -----------------------------------------------------------------
def auc(scores, labels):
    labels = np.asarray(labels)
    pos = labels == 1
    neg = labels == 0
    np_, nn = pos.sum(), neg.sum()
    if np_ == 0 or nn == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def mrr(scores, labels):
    labels = np.asarray(labels)
    order = np.argsort(-scores)
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg(scores, labels, k):
    labels = np.asarray(labels)
    order = np.argsort(-scores)[:k]
    gains = labels[order]
    ideal = np.sort(labels)[::-1][:k]
    idcg = dcg(ideal)
    return float(dcg(gains) / idcg) if idcg > 0 else 0.0


# ---- beyond-accuracy metrics ----------------------------------------------------------
def intra_list_diversity(ranked_vectors, k=10):
    """1 - mean pairwise cosine among the top-k recommended item vectors."""
    v = ranked_vectors[:k]
    if len(v) < 2:
        return 0.0
    sims = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            sims.append(float(v[i] @ v[j]))
    return 1.0 - (np.mean(sims) if sims else 0.0)


def novelty(ranked_ids, popularity_lookup, k=10):
    """Mean -log2(popularity_prob) of the top-k items (higher = more novel/less popular)."""
    top = ranked_ids[:k]
    vals = []
    total = max(sum(popularity_lookup.values()), 1)
    for a in top:
        p = (popularity_lookup.get(a, 0) + 1) / total
        vals.append(-np.log2(p))
    return float(np.mean(vals)) if vals else 0.0


def coverage(all_recommended_ids, catalogue_size):
    """Fraction of the catalogue that appears across all recommendations."""
    return len(set(all_recommended_ids)) / max(catalogue_size, 1)


# ---- bootstrap CI ---------------------------------------------------------------------
def bootstrap_ci(per_impression_values, n_boot=1000, alpha=0.05, seed=0):
    """95% CI of the mean via bootstrap resampling over impressions."""
    vals = np.array([v for v in per_impression_values if v is not None])
    if len(vals) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(vals.mean()), lo, hi)


# ---- driver ---------------------------------------------------------------------------
def evaluate(impression_results, slices=None, n_boot=1000):
    """impression_results: list of dicts, each with keys 'scores','labels', and
    optionally 'slice' (e.g. 'head'/'tail' or 'cold'/'warm').

    Returns a dict of metric -> (mean, lo, hi), overall and per slice.
    """
    def collect(rows):
        a = [auc(r["scores"], r["labels"]) for r in rows]
        m = [mrr(r["scores"], r["labels"]) for r in rows]
        n5 = [ndcg(r["scores"], r["labels"], 5) for r in rows]
        n10 = [ndcg(r["scores"], r["labels"], 10) for r in rows]
        return {
            "AUC": bootstrap_ci(a, n_boot),
            "MRR": bootstrap_ci(m, n_boot),
            "nDCG@5": bootstrap_ci(n5, n_boot),
            "nDCG@10": bootstrap_ci(n10, n_boot),
            "n": len(rows),
        }

    out = {"overall": collect(impression_results)}
    if slices:
        for name in slices:
            rows = [r for r in impression_results if r.get("slice") == name]
            if rows:
                out[name] = collect(rows)
    return out


def format_report(results):
    lines = []
    for group, metrics in results.items():
        lines.append(f"\n[{group}]  (n={metrics['n']})")
        for k in ("AUC", "MRR", "nDCG@5", "nDCG@10"):
            mean, lo, hi = metrics[k]
            if mean is not None:
                lines.append(f"  {k:8s} {mean:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    ap.add_argument("--cache", default=None, help="default: artifacts/<dataset>/eval_cache.pkl")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    cache_path = args.cache or os.path.join("artifacts", args.dataset, "eval_cache.pkl")
    if not os.path.exists(cache_path):
        raise SystemExit(
            f"No eval cache at {cache_path}. Run `python -m src.rerank --dataset "
            f"{args.dataset}` first to train the reranker and produce it."
        )
    with open(cache_path, "rb") as f:
        impression_results = pickle.load(f)

    results = evaluate(impression_results, slices=["head", "tail"], n_boot=args.n_boot)
    print(format_report(results))


if __name__ == "__main__":
    main()
