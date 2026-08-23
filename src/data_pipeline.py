"""
Q1 data pipeline: raw files -> unified schema -> temporal split -> feature-store inputs.

Unified schema (both datasets, IDs prefixed 'mind:' / 'eb:'):
  articles     : article_id, title, abstract, body, category, entities, published_time
  impressions  : impression_id, user_id, time, candidates[], labels[]
  history      : user_id, clicked_article_ids[], (timestamps[] where available)

MIND and EB-NeRD keep SEPARATE pools (different languages, separate leaderboards) —
never merged into one retrieval index.

Temporal split: both datasets ship pre-split by time (MIND train Nov9-14 / dev Nov15;
EB-NeRD train/validation/test folders). We respect that ordering. For the demo bundle a
timestamp-cutoff split is provided (never random).
"""
import datetime as dt

import numpy as np
import pandas as pd

MIND_NEWS = ["news_id", "category", "subcategory", "title", "abstract", "url", "te", "ae"]
MIND_BEH = ["impression_id", "user_id", "time", "history", "impressions"]


# ---- MIND ------------------------------------------------------------------------------
def parse_mind(split_dir, prefix="mind:"):
    news = pd.read_csv(f"{split_dir}/news.tsv", sep="\t", header=None, names=MIND_NEWS,
                       quoting=3, usecols=["news_id", "category", "title", "abstract"])
    news["title"] = news["title"].fillna("")
    news["abstract"] = news["abstract"].fillna("")
    articles = pd.DataFrame({
        "article_id": prefix + news["news_id"].astype(str),
        "title": news["title"],
        "abstract": news["abstract"],
        "body": "",                       # MIND-small has no body
        "category": news["category"].fillna(""),
    }).drop_duplicates("article_id").reset_index(drop=True)

    beh = pd.read_csv(f"{split_dir}/behaviors.tsv", sep="\t", header=None, names=MIND_BEH,
                      quoting=3)
    beh["t"] = pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

    impressions, history = [], {}
    for r in beh.itertuples():
        cands, labels = [], []
        if isinstance(r.impressions, str):
            for tk in r.impressions.split():
                p = tk.split("-")
                if len(p) == 2:
                    cands.append(prefix + p[0])
                    labels.append(int(p[1]))
        impressions.append({
            "impression_id": r.impression_id,
            "user_id": prefix + str(r.user_id),
            "time": r.t,
            "candidates": cands,
            "labels": labels,
        })
        if isinstance(r.history, str) and r.history:
            history[prefix + str(r.user_id)] = [prefix + x for x in r.history.split()]

    return articles, pd.DataFrame(impressions), history


# ---- EB-NeRD ---------------------------------------------------------------------------
def parse_ebnerd(base_dir, split, prefix="eb:"):
    import polars as pl

    a = pl.read_parquet(f"{base_dir}/articles.parquet")
    cols = a.columns
    def col(name):
        return a[name].to_list() if name in cols else [None] * a.height
    articles = pd.DataFrame({
        "article_id": [prefix + str(x) for x in a["article_id"].to_list()],
        "title": [t or "" for t in col("title")],
        "abstract": [s or "" for s in col("subtitle")],
        "body": [b or "" for b in col("body")],
        "category": [c or "" for c in col("category_str")],
        "published_time": col("published_time"),
        "total_pageviews": col("total_pageviews"),
        "total_inviews": col("total_inviews"),
        "total_read_time": col("total_read_time"),
        "sentiment_score": col("sentiment_score"),
    })

    b = pl.read_parquet(f"{base_dir}/{split}/behaviors.parquet")
    imp = pd.DataFrame({
        "impression_id": b["impression_id"].to_list() if "impression_id" in b.columns else range(b.height),
        "user_id": [prefix + str(u) for u in b["user_id"].to_list()],
        "time": b["impression_time"].to_list(),
        "candidates": [[prefix + str(x) for x in inv] for inv in b["article_ids_inview"].to_list()],
        "labels": ([[1 if x in set(clk or []) else 0 for x in inv]
                    for inv, clk in zip(b["article_ids_inview"].to_list(),
                                        b["article_ids_clicked"].to_list())]
                   if "article_ids_clicked" in b.columns else
                   [[0] * len(inv) for inv in b["article_ids_inview"].to_list()]),
    })

    h = pl.read_parquet(f"{base_dir}/{split}/history.parquet")
    history = {prefix + str(u): [prefix + str(x) for x in (arts or [])]
               for u, arts in zip(h["user_id"].to_list(), h["article_id_fixed"].to_list())}

    return articles, imp, history


def temporal_split_by_time(impressions_df, val_days=1, test_days=1):
    """Timestamp-cutoff split (for a single-file source like the demo bundle).
    Never random: sort by time, take the last test_days as test, preceding val_days as val.
    """
    df = impressions_df.dropna(subset=["time"]).sort_values("time")
    tmax = df["time"].max()
    test_cut = tmax - dt.timedelta(days=test_days)
    val_cut = test_cut - dt.timedelta(days=val_days)
    train = df[df["time"] < val_cut]
    val = df[(df["time"] >= val_cut) & (df["time"] < test_cut)]
    test = df[df["time"] >= test_cut]
    return train, val, test
