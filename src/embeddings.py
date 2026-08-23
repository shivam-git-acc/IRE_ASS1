"""
Q3 semantic candidate generation (embeddings).

Encodes articles with a sentence-transformer (per-language model), builds an ANN index
(FAISS inner-product if available, else exact brute-force — allowed for small scale),
forms a mean-pooled user vector from click history, retrieves top-K from the full pool,
reports recall@K.

Model choice tracks language:
  MIND (English)  -> all-MiniLM-L6-v2         (beat multilingual E5: 0.67 vs 0.60)
  EB-NeRD (Danish)-> paraphrase-multilingual-MiniLM-L12-v2

  python -m src.embeddings --dataset mind   --k 50 100 200
  python -m src.embeddings --dataset ebnerd --k 50 100 200 --split demo
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ablations"))
import config as C  # noqa

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False


def normalize(v):
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-12)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1e-12, n)


class ANNIndex:
    """FAISS inner-product index with exact brute-force fallback."""

    def __init__(self, embeddings, ids, use_approximate=False):
        self.ids = list(ids)
        self.emb = np.ascontiguousarray(normalize(embeddings).astype(np.float32))
        self.index = None
        if HAS_FAISS:
            dim = self.emb.shape[1]
            if use_approximate and len(ids) > 10000:
                idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efSearch = 64
            else:
                idx = faiss.IndexFlatIP(dim)
            idx.add(self.emb)
            self.index = idx

    def search(self, query_vec, k):
        q = normalize(query_vec.astype(np.float32)).reshape(1, -1)
        if self.index is not None:
            scores, idx = self.index.search(q, min(k, len(self.ids)))
            return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        # brute-force
        sims = (self.emb @ q[0])
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


def user_vector(history, emb_by_id, max_hist=30):
    vecs = [emb_by_id[a] for a in history[-max_hist:] if a in emb_by_id]
    if not vecs:
        return None
    m = np.mean(vecs, axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def recall_at_k(index, emb_by_id, eval_rows, hist, ks=(50, 100, 200), max_hist=30):
    hits = {k: 0 for k in ks}
    n = 0
    for uid, clicked in eval_rows:
        uv = user_vector(hist.get(uid, []), emb_by_id, max_hist)
        if uv is None:
            continue
        got = index.search(uv, max(ks))
        got_ids = [g[0] for g in got]
        for k in ks:
            if clicked in got_ids[:k]:
                hits[k] += 1
        n += 1
    return {k: hits[k] / n for k in ks}, n


def _clicked_eval_rows(impressions):
    rows = []
    for uid, T, cand, labs in impressions:
        labs = list(labs)
        if 1 in labs:
            rows.append((uid, cand[labs.index(1)]))
    return rows


def run(dataset, cfg, ks, model_name=None, prefix=None, split=None, sample=None, seed=0):
    import common as CL  # ablations/common.py

    model_name = model_name or cfg["embedding_model"]
    prefix = prefix if prefix is not None else cfg.get("embedding_prefix", "none")

    if dataset == "mind":
        train_dir = C.resolve(cfg["paths"]["train"])
        dev_dir = C.resolve(cfg["paths"]["dev"])
        b = CL.load_mind(train_dir, dev_dir)
        eval_rows = _clicked_eval_rows(CL.mind_impressions(b["b_dv"]))
        ids, texts, hist = b["ids"], b["texts"], b["hist"]
    else:
        split = split or "demo"
        base = C.resolve(cfg["paths"][split])
        b = CL.load_ebnerd(base)
        eval_rows = _clicked_eval_rows(CL.ebnerd_impressions(base, "validation"))
        ids, texts = list(b["texts"].keys()), b["texts"]
        hist = b["load_hist"]("validation")

    if sample and len(eval_rows) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(eval_rows), sample, replace=False)
        eval_rows = [eval_rows[i] for i in idx]

    emb_by_id, mat = CL.encode_texts(model_name, ids, texts, prefix=prefix)
    index = ANNIndex(mat, ids)
    recall, n = recall_at_k(index, emb_by_id, eval_rows, hist, ks=tuple(ks))
    return recall, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="override the config's embedding_model")
    ap.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--split", default=None, help="ebnerd only: demo|small|large (default demo)")
    ap.add_argument("--sample", type=int, default=2000,
                     help="subsample eval impressions for speed (0 = use all)")
    args = ap.parse_args()

    cfg = C.load_config(args.config or f"configs/{args.dataset}.yaml")
    print("FAISS available:", HAS_FAISS)
    recall, n = run(args.dataset, cfg, args.k, model_name=args.model, split=args.split,
                     sample=(args.sample or None))

    print(f"[embeddings] dataset={args.dataset} eval_impressions={n}")
    for k in sorted(recall):
        print(f"  recall@{k:<4d} {recall[k]:.4f}")


if __name__ == "__main__":
    main()
