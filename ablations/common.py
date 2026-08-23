"""
Shared helpers for ablation scripts: dataset loading (MIND + EB-NeRD) into a common
in-memory form, plus a configurable feature builder so each ablation varies exactly one
thing (popularity mode, embedding model, pooling, feature subset, etc.).

All feature computation routes through src.features (leakage-safe primitives).
"""
import glob
import os
import sys
import datetime as dt
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features as F  # noqa: E402


# ======================================================================================
# MIND loading (TSV)
# ======================================================================================
MIND_NEWS = ["news_id", "category", "subcategory", "title", "abstract", "url", "te", "ae"]
MIND_BEH = ["impression_id", "user_id", "time", "history", "impressions"]


def mind_pfx(x):
    return f"mind:{x}"


def load_mind(train_dir, dev_dir, extra_news_dirs=()):
    """Return a dict bundle with articles, behaviours, history, event log.

    extra_news_dirs: additional directories (e.g. an unlabeled test split) whose
    news.tsv should be merged into the article pool for retrieval/features, without
    contributing to the behavioural event log (which must stay leakage-safe / labeled).
    """
    news = pd.concat([
        pd.read_csv(f"{d}/news.tsv", sep="\t", header=None, names=MIND_NEWS, quoting=3,
                    usecols=["news_id", "category", "title", "abstract"])
        for d in (train_dir, dev_dir, *extra_news_dirs)
    ]).drop_duplicates("news_id").reset_index(drop=True)
    news["title"] = news["title"].fillna("")
    news["abstract"] = news["abstract"].fillna("")

    ids = [mind_pfx(r.news_id) for r in news.itertuples()]
    id_to_row = {x: i for i, x in enumerate(ids)}
    cat = {mind_pfx(r.news_id): (r.category if isinstance(r.category, str) else "")
           for r in news.itertuples()}
    title_tok = {mind_pfx(r.news_id): F.tok(r.title) for r in news.itertuples()}
    corpus = [F.tok(f"{r.title} {r.abstract}") for r in news.itertuples()]
    texts = {mind_pfx(r.news_id): f"{r.title} {r.abstract}".strip() for r in news.itertuples()}

    def load_beh(p):
        b = pd.read_csv(f"{p}/behaviors.tsv", sep="\t", header=None, names=MIND_BEH, quoting=3)
        b["t"] = pd.to_datetime(b["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        return b

    b_tr, b_dv = load_beh(train_dir), load_beh(dev_dir)

    hist = {}
    for b in (b_tr, b_dv):
        for u, h in zip(b["user_id"], b["history"]):
            if isinstance(h, str) and h:
                hist[mind_pfx(u)] = [mind_pfx(x) for x in h.split()]

    first_seen = {}
    log = F.EventLog()
    for b in (b_tr, b_dv):
        for t, imps in zip(b["t"], b["impressions"]):
            if pd.isna(t) or not isinstance(imps, str):
                continue
            for tk in imps.split():
                parts = tk.split("-")
                nid = mind_pfx(parts[0])
                if nid not in first_seen or t < first_seen[nid]:
                    first_seen[nid] = t
                if len(parts) == 2:
                    log.add(nid, t, parts[1] == "1")
    log.finalize()

    return dict(ids=ids, id_to_row=id_to_row, cat=cat, title_tok=title_tok,
                corpus=corpus, texts=texts, b_tr=b_tr, b_dv=b_dv, hist=hist,
                first_seen=first_seen, log=log, pfx=mind_pfx)


def mind_impressions(b_df):
    """Yield (uid, T, candidates, labels) for impressions with a click."""
    for u, t, imps in zip(b_df["user_id"], b_df["t"], b_df["impressions"]):
        if not isinstance(imps, str) or pd.isna(t):
            continue
        cand, labs = [], []
        for tk in imps.split():
            p = tk.split("-")
            if len(p) == 2:
                cand.append(mind_pfx(p[0]))
                labs.append(int(p[1]))
        if cand and sum(labs) > 0:
            yield mind_pfx(u), t, cand, np.array(labs)


def mind_test_impressions(b_df):
    """Yield (impression_id, uid, T, candidates) for an UNLABELED test split (MIND test
    impressions are bare news ids, no '-0/-1' suffix). No click filter — every row is
    needed for submission."""
    for iid, u, t, imps in zip(b_df["impression_id"], b_df["user_id"], b_df["t"],
                                b_df["impressions"]):
        if not isinstance(imps, str) or pd.isna(t):
            continue
        cand = [mind_pfx(tk.split("-")[0]) for tk in imps.split()]
        if cand:
            yield iid, mind_pfx(u), t, cand


# ======================================================================================
# EB-NeRD loading (parquet)
# ======================================================================================
def eb_pfx(x):
    return f"eb:{x}"


def load_ebnerd(base_dir, testset=False):
    """base_dir contains articles.parquet + train/ + validation/ (or test/)."""
    import polars as pl

    a = pl.read_parquet(f"{base_dir}/articles.parquet")
    cols = a.columns
    def col(name):
        return a[name].to_list() if name in cols else [None] * a.height

    aid = a["article_id"].to_list()
    pub = {eb_pfx(r): p for r, p in zip(aid, col("published_time"))}
    pv = {eb_pfx(r): (v or 0) for r, v in zip(aid, col("total_pageviews"))}
    iv = {eb_pfx(r): (v or 0) for r, v in zip(aid, col("total_inviews"))}
    rt = {eb_pfx(r): (v or 0) for r, v in zip(aid, col("total_read_time"))}
    cat = {eb_pfx(r): (c or "") for r, c in zip(aid, col("category_str"))}
    sent = {eb_pfx(r): (s or 0.0) for r, s in zip(aid, col("sentiment_score"))}
    texts = {eb_pfx(r): f"{t or ''} {s or ''}".strip()
             for r, t, s in zip(aid, col("title"), col("subtitle"))}

    def load_hist(sub):
        h = pl.read_parquet(f"{base_dir}/{sub}/history.parquet")
        return {eb_pfx(u): [eb_pfx(x) for x in (arts or [])]
                for u, arts in zip(h["user_id"].to_list(), h["article_id_fixed"].to_list())}

    return dict(pub=pub, pv=pv, iv=iv, rt=rt, cat=cat, sent=sent, texts=texts,
                load_hist=load_hist, base_dir=base_dir, pfx=eb_pfx)


def ebnerd_impressions(base_dir, split):
    import polars as pl
    b = pl.read_parquet(f"{base_dir}/{split}/behaviors.parquet")
    for uid, T, inv, clk in zip(b["user_id"].to_list(), b["impression_time"].to_list(),
                                b["article_ids_inview"].to_list(),
                                b["article_ids_clicked"].to_list()):
        cand = [eb_pfx(x) for x in inv]
        labs = set(eb_pfx(x) for x in (clk or []))
        yy = np.array([1 if c in labs else 0 for c in cand])
        if cand and yy.sum() > 0:
            yield eb_pfx(uid), T, cand, yy


def ebnerd_test_impressions(base_dir, split):
    """Yield (impression_id, uid, T, candidates) for an UNLABELED test split. No click
    filter — every row is needed for submission."""
    import polars as pl
    b = pl.read_parquet(f"{base_dir}/{split}/behaviors.parquet")
    for iid, uid, T, inv in zip(b["impression_id"].to_list(), b["user_id"].to_list(),
                                 b["impression_time"].to_list(),
                                 b["article_ids_inview"].to_list()):
        cand = [eb_pfx(x) for x in inv]
        if cand:
            yield iid, eb_pfx(uid), T, cand


# ======================================================================================
# Embedding helper
# ======================================================================================
def encode_texts(model_name, id_list, text_lookup, prefix="none", batch_size=512):
    """Encode articles with a sentence-transformer. prefix in {none, passage, query}."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    if prefix == "none":
        texts = [text_lookup[i] if text_lookup[i] else "news" for i in id_list]
    else:
        texts = [f"{prefix}: {text_lookup[i]}".strip() for i in id_list]
    mat = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    return {id_list[i]: mat[i] for i in range(len(id_list))}, mat


# ======================================================================================
# AUC over impressions (small utility used by ablations)
# ======================================================================================
def auc(scores, labels):
    labels = np.asarray(labels)
    pos = labels == 1; neg = labels == 0
    np_, nn = pos.sum(), neg.sum()
    if np_ == 0 or nn == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def mean_auc(per_impression):
    vals = [a for a in per_impression if a is not None]
    return float(np.mean(vals)) if vals else None
