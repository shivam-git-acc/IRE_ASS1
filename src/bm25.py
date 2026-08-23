"""
Q2 lexical candidate generation (BM25).

Builds a BM25 index over article title+abstract, constructs a query from the user's
recent click-history titles, retrieves top-K from the FULL article pool, and reports
recall@K for K in {50,100,200}.

Since ~99% of impressions have a single clicked article, recall@K reduces to hit@K:
the fraction of impressions whose clicked article appears in the top-K retrieved.

  python -m src.bm25 --dataset mind   --k 50 100 200
  python -m src.bm25 --dataset ebnerd --k 50 100 200 --split demo
"""
import argparse
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ablations"))
import features as F  # noqa
import config as C  # noqa


def build_query(user_history, title_tok, max_hist=30):
    q = []
    for aid in user_history[-max_hist:]:
        q.extend(title_tok.get(aid, []))
    return q


def recall_at_k(bm25, eval_rows, id_to_row, hist, title_tok, ks=(50, 100, 200), max_hist=30):
    """eval_rows: list of (user_id, clicked_article_id) with non-empty history."""
    hits = {k: 0 for k in ks}
    n = 0
    for uid, clicked in eval_rows:
        h = hist.get(uid, [])
        if not h:
            continue
        q = build_query(h, title_tok, max_hist)
        scores = bm25.scores_all(q)
        order = np.argsort(-scores)
        crow = id_to_row.get(clicked)
        if crow is None:
            continue
        pos = np.where(order == crow)[0]
        if len(pos) == 0:
            continue
        rank = pos[0]
        for k in ks:
            if rank < k:
                hits[k] += 1
        n += 1
    return {k: hits[k] / n for k in ks}, n


def _clicked_eval_rows(impressions):
    """impressions: iterable of (uid, T, candidates, labels) -> [(uid, clicked_id), ...]

    Takes the first positive label per impression (recall@K reduces to hit@K since
    ~99% of impressions have exactly one click).
    """
    rows = []
    for uid, T, cand, labs in impressions:
        labs = list(labs)
        if 1 in labs:
            rows.append((uid, cand[labs.index(1)]))
    return rows


def _lexical_index(ids, texts):
    id_to_row = {x: i for i, x in enumerate(ids)}
    corpus = [F.tok(texts[x]) for x in ids]
    title_tok = {x: F.tok(texts[x]) for x in ids}
    return id_to_row, corpus, title_tok


def run(dataset, cfg, ks, split=None, sample=None, seed=0):
    import common as CL  # ablations/common.py

    if dataset == "mind":
        train_dir = C.resolve(cfg["paths"]["train"])
        dev_dir = C.resolve(cfg["paths"]["dev"])
        b = CL.load_mind(train_dir, dev_dir)
        eval_rows = _clicked_eval_rows(CL.mind_impressions(b["b_dv"]))
        id_to_row, corpus, title_tok, hist = b["id_to_row"], b["corpus"], b["title_tok"], b["hist"]
    else:
        split = split or "demo"
        base = C.resolve(cfg["paths"][split])
        b = CL.load_ebnerd(base)
        eval_rows = _clicked_eval_rows(CL.ebnerd_impressions(base, "validation"))
        ids = list(b["texts"].keys())
        id_to_row, corpus, title_tok = _lexical_index(ids, b["texts"])
        hist = b["load_hist"]("validation")

    if sample and len(eval_rows) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(eval_rows), sample, replace=False)
        eval_rows = [eval_rows[i] for i in idx]

    bm25 = F.BM25(corpus)
    recall, n = recall_at_k(bm25, eval_rows, id_to_row, hist, title_tok, ks=tuple(ks))
    return recall, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--split", default=None, help="ebnerd only: demo|small|large (default demo)")
    ap.add_argument("--sample", type=int, default=2000,
                     help="subsample eval impressions for speed (0 = use all)")
    args = ap.parse_args()

    cfg = C.load_config(args.config or f"configs/{args.dataset}.yaml")
    recall, n = run(args.dataset, cfg, args.k, split=args.split,
                     sample=(args.sample or None))

    print(f"[bm25] dataset={args.dataset} eval_impressions={n}")
    for k in sorted(recall):
        print(f"  recall@{k:<4d} {recall[k]:.4f}")


if __name__ == "__main__":
    main()
