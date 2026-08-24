# MIND — Individual Kaggle Notebooks

Each notebook is fully self-contained (hardcoded paths, no cross-imports). Run any one
independently on Kaggle. Recommended run order:

| # | Notebook | What it does | Data |
|---|----------|--------------|------|
| 01 | data_pipeline | Parse -> unified schema -> temporal split -> leakage test (Q9) | small |
| 02 | bm25_retrieval | Lexical BM25 recall@K {50,100,200} | small |
| 03 | embedding_retrieval | MiniLM semantic recall@K; lexical vs semantic (semantic wins) | small |
| 04 | baseline_rankers | Single-signal reranking AUC (pop/ctr/recency/content) | small |
| 05 | lightgbm_reranker | Full LightGBM + Q4 harness (AUC/MRR/nDCG + CIs + head/tail) | small |
| 06 | ablation_frozen_vs_rolling | Q9 anti-gaming: frozen vs rolling popularity | small |
| 07 | ablation_embedding_model | E5 vs MiniLM, prefixes, mean vs best pooling | small |
| 08 | ablation_nrms_hybrid | NRMS vs LightGBM vs hybrid (feature model wins) | small/dev |
| 10 | submission | Train on large, predict on test, write prediction.zip | large |

## Notes
- Notebooks 01-08 use MINDsmall (train -> dev) for fast offline metrics.
- Notebook 10 uses MINDlarge for the actual leaderboard submission.
- `!pip install` is the first line of each (valid Jupyter magic).
- Paths are hardcoded to your Kaggle mounts.
- Faithful to your real mind-minilm.ipynb code.

Key MIND findings reproduced:
- Semantic (MiniLM) > lexical (BM25) retrieval
- English MiniLM > multilingual E5 (specialisation > size)
- CTR + content dominate features; LightGBM ~0.695 dev
- NRMS (0.662) < LightGBM; hybrid didn't help
