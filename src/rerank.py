"""
LightGBM learning-to-rank reranker.

Builds the per-candidate feature matrix (content + behavioural, all leakage-safe via
src.features) and trains a LambdaMART ranker. The reranker scores the GIVEN in-view
slate (this is the leaderboard task) — distinct from retrieval, which searches the pool.

Feature set (MIND):  bm25, emb_mean, emb_best, recency, pop_1h/24h/7d, pop_vel,
                      ctr_24h, ctr_total, cat_aff, position, slate_size
Feature set (EB-NeRD):recency, pageviews, inviews, read_time, ctr_proxy, cat_match,
                      emb_mean, emb_best, sentiment, position, slate_size

Trains on the labeled train split and scores the labeled dev/validation split, saving
the model and a per-impression (scores, labels) cache for `src.evaluate` to consume:

  python -m src.rerank   --dataset mind
  python -m src.evaluate --dataset mind
"""
import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ablations"))
import features as F  # noqa
import config as C  # noqa


def make_ranker(cfg_reranker):
    return lgb.LGBMRanker(
        objective=cfg_reranker.get("objective", "lambdarank"),
        n_estimators=cfg_reranker.get("n_estimators", 500),
        learning_rate=cfg_reranker.get("learning_rate", 0.03),
        num_leaves=cfg_reranker.get("num_leaves", 31),
        min_child_samples=cfg_reranker.get("min_child_samples", 50),
        importance_type="gain",
        verbose=-1,
    )


def build_mind_features(uid, T, cand, ctx):
    """ctx: dict with bm25, id_to_row, title_tok, emb_by_id, cat, log, first_seen, hist."""
    hist = ctx["hist"].get(uid, [])
    q = []
    for a in hist[-30:]:
        q.extend(ctx["title_tok"].get(a, []))
    hv = [ctx["emb_by_id"][a] for a in hist[-30:] if a in ctx["emb_by_id"]]
    um = F.mean_pool(hv)
    # user category affinity
    cats = [ctx["cat"].get(a) for a in hist[-30:]]
    tot = len([c for c in cats if c])
    cd = {k: v / tot for k, v in Counter(c for c in cats if c).items()} if tot else {}

    m = len(cand)
    rows = []
    for i, c in enumerate(cand):
        bm = ctx["bm25"].score(q, ctx["id_to_row"][c]) if c in ctx["id_to_row"] else 0.0
        cv = ctx["emb_by_id"].get(c)
        em = float(um @ cv) if (um is not None and cv is not None) else 0.0
        eb = F.best_match(cv, hv)
        rec = F.recency(ctx["first_seen"].get(c), T, tau_hours=6.0)
        p1 = ctx["log"].popularity(c, T, 1)
        p24 = ctx["log"].popularity(c, T, 24)
        p7 = ctx["log"].popularity(c, T, 168)
        pv = p1 / (p24 + 1)
        c24 = ctx["log"].ctr(c, T, 24)
        ct = ctx["log"].ctr(c, T)
        ca = cd.get(ctx["cat"].get(c, ""), 0.0)
        rows.append([bm, em, eb, rec, p1, p24, p7, pv, c24, ct, ca, i / max(1, m - 1), m])
    X = np.array(rows)
    for col in (0, 4, 5, 6, 7):  # min-max the raw-count columns
        X[:, col] = F.minmax(X[:, col])
    return X


def build_ebnerd_features(uid, T, cand, ctx):
    """ctx: dict with emb_by_id, cat, hist, pub, pv, iv, rt, sent, recency_tau_hours.

    Uses EB-NeRD's provided article-level totals (total_pageviews/inviews/read_time) as
    the popularity/CTR signal instead of a per-event log (see configs/ebnerd.yaml
    `use_provided_totals`).
    """
    hist = ctx["hist"].get(uid, [])
    hv = [ctx["emb_by_id"][a] for a in hist[-30:] if a in ctx["emb_by_id"]]
    um = F.mean_pool(hv)
    cats = [ctx["cat"].get(a) for a in hist[-30:]]
    tot = len([c for c in cats if c])
    cd = {k: v / tot for k, v in Counter(c for c in cats if c).items()} if tot else {}

    tau = ctx.get("recency_tau_hours", 24.0)
    m = len(cand)
    rows = []
    for i, c in enumerate(cand):
        cv = ctx["emb_by_id"].get(c)
        em = float(um @ cv) if (um is not None and cv is not None) else 0.0
        eb = F.best_match(cv, hv)
        rec = F.recency(ctx["pub"].get(c), T, tau_hours=tau)
        pv = ctx["pv"].get(c, 0)
        iv = ctx["iv"].get(c, 0)
        rt = ctx["rt"].get(c, 0)
        ctr = pv / (iv + 1)
        sent = ctx["sent"].get(c, 0.0)
        ca = cd.get(ctx["cat"].get(c, ""), 0.0)
        rows.append([rec, pv, iv, rt, ctr, ca, em, eb, sent, i / max(1, m - 1), m])
    X = np.array(rows)
    for col in (1, 2, 3):  # min-max the raw-count columns
        X[:, col] = F.minmax(X[:, col])
    return X


def train(ranker, X, y, groups):
    ranker.fit(X, y, group=groups)
    return ranker


def predict_ranks(ranker, X):
    """Return 1-based ranks per candidate (rank 1 = highest score), in candidate order."""
    sc = ranker.predict(X)
    order = np.argsort(-sc)
    ranks = np.empty(len(sc), dtype=int)
    ranks[order] = np.arange(1, len(sc) + 1)
    return ranks


def _subsample(rows, n, seed=0):
    if not n or len(rows) <= n:
        return rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), n, replace=False)
    return [rows[i] for i in idx]


def run_mind(cfg, train_sample=None, eval_sample=None, seed=0):
    import common as CL  # ablations/common.py

    train_dir = C.resolve(cfg["paths"]["train"])
    dev_dir = C.resolve(cfg["paths"]["dev"])
    b = CL.load_mind(train_dir, dev_dir)
    emb_by_id, _ = CL.encode_texts(cfg["embedding_model"], b["ids"], b["texts"],
                                    prefix=cfg.get("embedding_prefix", "none"))
    bm25 = F.BM25(b["corpus"])
    ctx = dict(bm25=bm25, id_to_row=b["id_to_row"], title_tok=b["title_tok"],
               emb_by_id=emb_by_id, cat=b["cat"], log=b["log"],
               first_seen=b["first_seen"], hist=b["hist"])

    train_rows = _subsample(list(CL.mind_impressions(b["b_tr"])), train_sample, seed)
    Xs, ys, groups = [], [], []
    for uid, T, cand, labs in train_rows:
        Xs.append(build_mind_features(uid, T, cand, ctx))
        ys.append(labs)
        groups.append(len(cand))
    ranker = make_ranker(cfg["reranker"])
    train(ranker, np.vstack(Xs), np.concatenate(ys), groups)

    dev_rows = _subsample(list(CL.mind_impressions(b["b_dv"])), eval_sample, seed)
    results, clicked_pop = [], []
    dev_max_T = max((T for _, T, _, _ in dev_rows), default=None)
    for uid, T, cand, labs in dev_rows:
        scores = ranker.predict(build_mind_features(uid, T, cand, ctx))
        results.append({"scores": scores, "labels": labs})
        clicked = cand[list(labs).index(1)]
        clicked_pop.append(ctx["log"].popularity(clicked, dev_max_T))
    _tag_head_tail(results, clicked_pop)
    return ranker, results


def run_ebnerd(cfg, split="small", train_sample=None, eval_sample=None, seed=0):
    import common as CL  # ablations/common.py

    base = C.resolve(cfg["paths"][split])
    b = CL.load_ebnerd(base)
    ids = list(b["texts"].keys())
    emb_by_id, _ = CL.encode_texts(cfg["embedding_model"], ids, b["texts"],
                                    prefix=cfg.get("embedding_prefix", "none"))
    tau = cfg.get("features", {}).get("recency_tau_hours", 24.0)
    ctx_tr = dict(emb_by_id=emb_by_id, cat=b["cat"], hist=b["load_hist"]("train"),
                  pub=b["pub"], pv=b["pv"], iv=b["iv"], rt=b["rt"], sent=b["sent"],
                  recency_tau_hours=tau)
    ctx_va = dict(ctx_tr, hist=b["load_hist"]("validation"))

    train_rows = _subsample(list(CL.ebnerd_impressions(base, "train")), train_sample, seed)
    Xs, ys, groups = [], [], []
    for uid, T, cand, labs in train_rows:
        Xs.append(build_ebnerd_features(uid, T, cand, ctx_tr))
        ys.append(labs)
        groups.append(len(cand))
    ranker = make_ranker(cfg["reranker"])
    train(ranker, np.vstack(Xs), np.concatenate(ys), groups)

    dev_rows = _subsample(list(CL.ebnerd_impressions(base, "validation")), eval_sample, seed)
    results, clicked_pv = [], []
    for uid, T, cand, labs in dev_rows:
        scores = ranker.predict(build_ebnerd_features(uid, T, cand, ctx_va))
        results.append({"scores": scores, "labels": labs})
        clicked = cand[list(labs).index(1)]
        clicked_pv.append(ctx_va["pv"].get(clicked, 0))
    _tag_head_tail(results, clicked_pv)
    return ranker, results


def _tag_head_tail(results, clicked_popularity):
    """Slice each impression by whether its clicked article is above/below the median
    popularity among clicked articles in this eval set."""
    if not clicked_popularity:
        return
    med = float(np.median(clicked_popularity))
    for r, p in zip(results, clicked_popularity):
        r["slice"] = "head" if p >= med else "tail"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None, help="ebnerd only: demo|small|large (default small)")
    ap.add_argument("--train-sample", type=int, default=5000,
                     help="subsample train impressions for speed (0 = use all)")
    ap.add_argument("--eval-sample", type=int, default=2000,
                     help="subsample dev/validation impressions for speed (0 = use all)")
    ap.add_argument("--out-dir", default=None, help="default: artifacts/<dataset>")
    args = ap.parse_args()

    cfg = C.load_config(args.config or f"configs/{args.dataset}.yaml")
    out_dir = args.out_dir or os.path.join("artifacts", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "mind":
        ranker, results = run_mind(cfg, train_sample=(args.train_sample or None),
                                    eval_sample=(args.eval_sample or None))
    else:
        ranker, results = run_ebnerd(cfg, split=(args.split or "small"),
                                      train_sample=(args.train_sample or None),
                                      eval_sample=(args.eval_sample or None))

    model_path = os.path.join(out_dir, "reranker.txt")
    ranker.booster_.save_model(model_path)
    cache_path = os.path.join(out_dir, "eval_cache.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[rerank] dataset={args.dataset} eval_impressions={len(results)}")
    print(f"[rerank] model saved to {model_path}")
    print(f"[rerank] eval cache saved to {cache_path}")
    print(f"[rerank] next: python -m src.evaluate --dataset {args.dataset}")


if __name__ == "__main__":
    main()
