# Results & Ablations

All numbers below are from the committed notebooks in `notebooks/`. Each table notes its
source notebook and cell so results are traceable and reproducible.

---

## Leaderboard (Q5)

| Dataset  | Metric | Score  | Model / notebook |
|----------|--------|--------|------------------|
| MIND     | AUC    | 0.6435 | LightGBM + all-MiniLM-L6-v2, trained large / `mind-minilm.ipynb` |
| MIND     | MRR    | 0.3141 | " |
| MIND     | nDCG@5 | 0.3411 | " |
| MIND     | nDCG@10| 0.3976 | " |
| EB-NeRD  | AUC    | 0.6976 | LightGBM trained on `ebnerd_large` + multilingual MiniLM / `ebnerd-minilm.ipynb` |
| EB-NeRD  | MRR    | 0.4769 | " |
| EB-NeRD  | nDCG@5 | 0.5402 | " |
| EB-NeRD  | nDCG@10| 0.5875 | " |

MIND leaderboard progression: 0.588 (popularity/CTR only) → 0.6145 (+content E5/BM25)
→ 0.6292 (corrected E5 query/passage prefixes) → **0.6435** (swapped E5 → English MiniLM).

---

## Q2/Q3 — Retrieval recall@K (from full article pool)

Since ~99% of impressions have a single clicked article, recall@K = hit@K.

### EB-NeRD demo (`dataset-handling.ipynb`, cells 13–21, 27)

| signal / method | recall@50 | recall@100 | recall@200 |
|-----------------|-----------|------------|------------|
| **recency**     | 0.9142    | 0.9436     | **0.9597** |
| **popularity**  | 0.1572    | 0.2644     | 0.4363     |
| BM25 (lexical)  | 0.0145    | 0.0268     | 0.0383     |
| word2vec        | 0.0090    | 0.0180     | 0.0374     |
| E5              | 0.0101    | 0.0190     | 0.0368     |
| mBERT           | 0.0070    | 0.0139     | 0.0301     |

Finding: **recency dominates retrieval** (clicked-article age median 2.8 h; 94.8 % clicked
within 24 h of publish — cell 28). Content methods are near-zero. Semantic ≈ lexical.

### MIND small (retrieval check, MiniLM)

| method (agg)         | recall@50 | recall@100 | recall@200 |
|----------------------|-----------|------------|------------|
| BM25 (lexical)       | 0.0075    | 0.0135     | 0.0210     |
| MiniLM (mean-pool)   | 0.0125    | 0.0190     | 0.0285     |
| MiniLM (best-match)  | 0.0080    | 0.0180     | 0.0290     |

Finding: content retrieval is weak on **both** datasets (~0.02–0.04), robust to aggregation
(mean-pool ≈ best-match). MiniLM ≥ BM25. The clicked article is rarely the content-nearest
neighbour of history — news clicks are driven by freshness/popularity, not similarity.

---

## Q4 — Reranking (given in-view slate), EB-NeRD demo

`dataset-handling.ipynb`, cell 31 (full harness with bootstrap 95% CIs + head/tail).

| method            | AUC (all) | AUC head | AUC tail |
|-------------------|-----------|----------|----------|
| **popularity**    | **0.693** | 0.825    | 0.568    |
| fusion .4/.3/.3   | 0.632     | 0.697    | —        |
| content (BM25)    | 0.519     | 0.531    | 0.508    |
| recency           | 0.503     | 0.451    | 0.552    |

**Signal inversion (headline finding):** recency *dominates retrieval* (0.96 recall@200)
but *collapses to chance on reranking* (AUC 0.503) — because the production system already
filters the in-view slate to fresh articles, so recency can't discriminate among them.
Popularity does the opposite: mediocre in retrieval (0.44) but the strongest single
reranking signal (0.693). **A signal's value is task-dependent.**

Learned rankers (cell 32, temporal holdout):

| model      | AUC all | head  | tail  |
|------------|---------|-------|-------|
| LightGBM   | 0.717   | 0.827 | 0.649 |
| logistic   | 0.688   | 0.840 | 0.595 |
| popularity | 0.679   | 0.842 | 0.579 |

Logistic standardized coefficients: popularity **+0.692** ≫ recency +0.378 > cat_match
+0.142 > content +0.130 — quantifies popularity dominance.

---

## Ablation 1 — Frozen vs Rolling popularity (temporal drift)

`dataset-handling.ipynb`, cell 39. EB-NeRD small, temporal split.

| popularity mode | AUC (all) | head  | tail  |
|-----------------|-----------|-------|-------|
| **frozen** (precomputed once) | **0.568** | — | — |
| **rolling** (strictly-before-T, per impression) | **0.773** | 0.664 | 0.775 |

Frozen popularity both **leaks** (an early impression "sees" later clicks) and goes
**stale** across the train→eval time gap → 0.568. Rolling point-in-time popularity is
leakage-safe *and* +0.205 AUC. Rolling feature importances: pop_24h (1.20M) ≫ ctr_24h
(0.84M) > pop_1h > slate_size > recency > e5_bestmatch > cat_affinity. This is both the
**Q9 with/without-serving-time-features** ablation and the core temporal-drift finding.

---

## Ablation 2 — Session features (null result)

`dataset-handling.ipynb`, cell 35. EB-NeRD small.

| model            | AUC all | head  | tail  |
|------------------|---------|-------|-------|
| LightGBM         | 0.756   | 0.826 | 0.714 |
| + session feats  | 0.757   | 0.827 | 0.715 |

Sessions are short (mean 1.91 impressions, median 1; 44.4 % have >1). Adding
session_cat_match / session_depth gives +0.001 — negligible. Documented null result:
session context doesn't help when sessions are this short.

---

## Ablation 3 — E5 prefix + aggregation (MIND)

| variant                              | reranking AUC |
|--------------------------------------|---------------|
| E5, wrong (all-passage), mean-pool   | 0.588         |
| E5, correct query/passage, best-match| 0.602         |
| **MiniLM, mean-pool (no prefix)**    | **0.6714**    |
| MiniLM, best-match                   | 0.6630        |

Findings: correct E5 query/passage prefixes help (+0.014). But English-specialised
**MiniLM beats multilingual E5 by ~0.07** — model specialisation > model size on English
news. MiniLM prefers mean-pool; E5 preferred best-match (opposite optima).

---

## Ablation 3b — Lexical vs Semantic retrieval, sliced (answers "on which slices?")

Retrieval recall@K on the full pool, sliced to test *where* semantic beats lexical.
User slices use data-driven history-length percentiles (p20/p80), computed from the eval
set; article slices use median popularity. MIND uses dev as test (MiniLM as semantic);
EB-NeRD uses validation as test (E5 as semantic). Each EB-NeRD cell also carries a
bootstrap 95% CI on the (semantic − lexical) difference.

### MIND — recall@K by user-history slice (dev as test, n=4000)

History-length distribution: min=1, p20=7, median=20, p80=52, max=434.
cold := ≤ p20 (7 clicks), warm := ≥ p80 (52 clicks).

| slice | n | K | BM25 | MiniLM | ratio | winner |
|-------|-----|-----|--------|--------|-------|--------|
| all   | 4000 | 200 | 0.0185 | 0.0250 | 1.35× | MiniLM |
| cold  | 875  | 200 | 0.0183 | 0.0240 | 1.31× | MiniLM |
| mid   | 2311 | 200 | 0.0212 | 0.0234 | 1.10× | MiniLM |
| **warm** | 814 | 200 | 0.0111 | **0.0307** | **2.78×** | MiniLM |

At recall@100 the ratio trend is even sharper: cold 1.27× → mid 1.16× → warm **3.20×**.

**Finding:** semantic (MiniLM) beats lexical (BM25) on MIND across **every** slice, but the
margin is governed by **history length**, not article popularity. For cold-start users the
two are near-parity (1.31×); for warm users (long history) MiniLM's recall@200 is 2.78×
BM25's. Mechanism: as history grows, BM25 *degrades* (0.0111, its worst — the concatenated-
title query becomes a diluted bag of hundreds of tokens) while MiniLM *improves* (0.0307, its
best — the mean-pooled embedding becomes a sharper user vector). Slicing by article
popularity (head vs tail) showed **no** comparable widening — so history length, not
article niche-ness, is the axis on which semantic's advantage concentrates.

### MIND — recall@200 by article-popularity slice (head vs tail)

| slice | n | BM25 | MiniLM | winner |
|-------|-----|--------|--------|--------|
| head (pop > 856) | 1995 | 0.0105 | 0.0165 | MiniLM |
| tail (pop ≤ 856) | 2005 | 0.0264 | 0.0334 | MiniLM |

MiniLM wins both; the gap does **not** widen for tail — confirms popularity is not the axis.

### EB-NeRD — recall@K by slice (validation as test, n=4000)

History-length distribution: min=5, p20=70, median=221, p80=478, max=1000.
(Note: EB-NeRD histories are long fixed arrays, so "cold" means "less long", not truly new
users — a history-length effect is not expected.) Article-popularity median = 82,472 pageviews.
Every cell reports a bootstrap 95% CI on (E5 − BM25).

| slice | n | K | BM25 | E5 | diff (E5−BM25) | 95% CI | verdict |
|-------|-----|-----|--------|--------|--------|--------|---------|
| all | 4000 | 200 | 0.0315 | 0.0272 | −0.0043 | [−0.0113, +0.0033] | **tie** |
| cold (≤70) | 801 | 200 | 0.0300 | 0.0262 | −0.0037 | [−0.0200, +0.0112] | **tie** |
| warm (≥478) | 800 | 200 | 0.0262 | 0.0262 | +0.0000 | [−0.0162, +0.0163] | **tie** |
| head (pv>82k) | 1966 | 200 | 0.0300 | 0.0219 | −0.0081 | [−0.0183, +0.0015] | **tie** |
| tail (pv≤82k) | 2034 | 200 | 0.0329 | 0.0324 | −0.0005 | [−0.0099, +0.0093] | **tie** |

**Finding:** on EB-NeRD, BM25 and E5 are statistically **indistinguishable** — the 95% CI on
the difference spans zero in **every** slice at every K, and the nominal "winner" flips with K.
The MIND history-length mechanism does **not** replicate. Interpretation: on EB-NeRD content
barely predicts clicks (popularity/recency dominate), so the *type* of content model barely
matters — lexical vs semantic is a coin-flip.

**Cross-dataset conclusion:** where content carries signal (English MIND), *how* it is modelled
matters (semantic > lexical, widening with history); where clicks are popularity/recency-driven
(Danish EB-NeRD), the choice of content model is a coin-flip. This is the retrieval-side mirror
of the cross-dataset reranking contrast below.

---

## Ablation 4 — NRMS vs LightGBM vs Hybrid (MIND dev)

| model                                   | dev AUC |
|-----------------------------------------|---------|
| **Feature LightGBM**                    | **0.695–0.697** |
| Normalized hybrid (LightGBM + rank-normed NRMS) | 0.6755 |
| Raw hybrid (LightGBM + raw NRMS logits) | 0.665   |
| NRMS (neural, title-only)               | 0.662   |
| Frozen E5 (best-match)                  | 0.602   |

Findings: feature LightGBM beats vanilla NRMS — behavioural signals (CTR/popularity) that
NRMS lacks matter on MIND. Adding NRMS as a LightGBM feature **hurt**: raw logits hijacked
tree splits (importance 3.0M, 28× everything else → collapsed to NRMS-alone 0.665);
rank-normalization per slate + feature_fraction=0.7 recovered to 0.6755 but still below
plain LightGBM. Neural content signal is redundant with E5/BM25 content features while
displacing behavioural signal. Feature engineering beat the neural model and their fusion.

---

## Ablation 5 — Data scale (MIND, GBDT saturation)

| training data            | dev/holdout AUC |
|--------------------------|-----------------|
| MINDsmall (~157k imps)   | 0.695           |
| MINDlarge (400k sample)  | 0.697           |

Training on 400k of large ≈ full 2.2M ≈ small: **GBDT saturates**; data quantity is not the
bottleneck. The dev→leaderboard drop (0.697 → 0.6435) is the concept-drift tax, not a data
deficiency.

---

## Cross-dataset comparison (Q6)

| dimension            | EB-NeRD               | MIND                     |
|----------------------|-----------------------|--------------------------|
| language             | Danish                | English                  |
| best AUC (LightGBM)  | 0.773 dev / 0.6976 LB | 0.695 dev / 0.6435 LB    |
| dominant signal      | raw popularity (pop_24h) | click-through-rate (ctr_total) |
| content reranking    | ≈ random (0.52)       | useful (E5 0.588, MiniLM helps) |
| content retrieval    | weak (~0.037)         | weak (~0.029)            |
| retrieval hero       | recency (0.96)        | (no reliable published_time) |
| semantic vs lexical  | tie (all CIs span 0)  | MiniLM > BM25 (2.78× for warm users) |
| head/tail            | tail ≥ head (drift)   | head > tail (normal)     |
| coverage (LightGBM)  | 0.155                 | 0.037                    |

Both reach ~0.70 with a trained LightGBM but for different reasons: EB-NeRD is
popularity/recency-driven, MIND is CTR + content-driven.
