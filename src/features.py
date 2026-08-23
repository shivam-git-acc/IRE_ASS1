"""
Leakage-safe feature engineering — the core of the pipeline.

Everything here obeys the point-in-time rule: for an impression at timestamp T, a
feature may only use events strictly before T. Behavioural counts route through
`count_before` (binary search over per-article sorted event times), recomputed per
impression (rolling). This is the single guarantee tested by tests/test_no_leakage.py.
"""
import datetime as dt
import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def tok(t):
    return _WORD.findall(t.lower()) if isinstance(t, str) else []


# --------------------------------------------------------------------------------------
# Point-in-time counting primitive (the leakage guard)
# --------------------------------------------------------------------------------------
def count_before(sorted_times, T, window_hours=None):
    """Count events strictly before T (optionally within a trailing window)."""
    if not sorted_times:
        return 0
    hi = bisect_left(sorted_times, T)
    if window_hours is None:
        return hi
    lo = bisect_left(sorted_times, T - dt.timedelta(hours=window_hours))
    return hi - lo


class EventLog:
    """Per-article sorted click/impression event times, for rolling popularity/CTR."""

    def __init__(self):
        self.clicks = defaultdict(list)
        self.impr = defaultdict(list)

    def add(self, article_id, when, clicked):
        self.impr[article_id].append(when)
        if clicked:
            self.clicks[article_id].append(when)

    def finalize(self):
        for d in (self.clicks, self.impr):
            for k in d:
                d[k].sort()

    def popularity(self, article_id, T, window_hours=None):
        return count_before(self.clicks.get(article_id), T, window_hours)

    def ctr(self, article_id, T, window_hours=None):
        c = count_before(self.clicks.get(article_id), T, window_hours)
        s = count_before(self.impr.get(article_id), T, window_hours)
        return c / s if s > 0 else 0.0


# --------------------------------------------------------------------------------------
# BM25 (lexical)
# --------------------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.tf = [Counter(d) for d in corpus_tokens]
        self.dl = np.array([len(d) for d in corpus_tokens], dtype=float)
        self.avg = self.dl.mean() if self.N else 0.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.idf = {w: math.log((self.N - d + 0.5) / (d + 0.5) + 1) for w, d in df.items()}

    def score(self, query_tokens, doc_row):
        if not query_tokens:
            return 0.0
        tf = self.tf[doc_row]
        denom = self.k1 * (1 - self.b + self.b * self.dl[doc_row] / self.avg)
        v = 0.0
        for w in set(query_tokens):
            f = tf.get(w, 0)
            if f:
                v += self.idf.get(w, 0) * (f * (self.k1 + 1)) / (f + denom)
        return v

    def scores_all(self, query_tokens):
        """Score every document (for full-pool retrieval / recall@K)."""
        return np.array([self.score(query_tokens, r) for r in range(self.N)])


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def minmax(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def recency(published_time, T, tau_hours):
    if published_time is None or T is None:
        return 0.0
    dh = (T - published_time).total_seconds() / 3600.0
    return float(np.exp(-dh / tau_hours)) if dh >= 0 else 0.0


def mean_pool(vectors):
    if not vectors:
        return None
    m = np.mean(vectors, axis=0)
    n = np.linalg.norm(m)
    return m / (n + 1e-9)


def best_match(cand_vec, hist_vectors):
    if cand_vec is None or not hist_vectors:
        return 0.0
    return float(max(v @ cand_vec for v in hist_vectors))
