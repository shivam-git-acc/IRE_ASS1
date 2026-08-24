# EB-NeRD — Individual Kaggle Notebooks

Each notebook is fully self-contained (hardcoded paths, no cross-imports). Run any one
independently on Kaggle. Recommended run order:

| # | Notebook | What it does | Data |
|---|----------|--------------|------|
| 01 | data_pipeline | Parse (polars) -> unified schema -> temporal split -> leakage test (Q9) | demo |
| 02 | bm25_retrieval | Lexical BM25 recall@K {50,100,200} | demo |
| 03 | embedding_retrieval | E5 semantic + recency/popularity retrieval recall@K | demo |
| 04 | baseline_rankers | Single-signal reranking AUC (popularity/recency/content) | demo |
| 05 | lightgbm_reranker | Full LightGBM (rolling) + Q4 harness (AUC/MRR/nDCG + CIs + head/tail) | demo |
| 06 | ablation_frozen_vs_rolling | Headline: frozen 0.568 -> rolling 0.773 | demo |
| 07 | ablation_embedding_model | multilingual MiniLM vs E5, mean vs best pooling | demo |
| 09 | ablation_session_features | Session features null result (+0.001) | demo |
| 10 | submission | Load trained model, predict on 13.5M testset -> zip | large/test |

## Notes
- Notebooks 01-09 use the EB-NeRD demo/small bundle for fast offline metrics.
- Notebook 10 loads the pre-trained ebnerd_large_lgbm.txt and predicts on the full
  13.5M-impression testset (double-nested path handled). This is the real submission code.
- Multilingual MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`) is the correct encoder for Danish.
- polars is used for parquet parsing (faithful to your real notebooks).
- `!pip install` is the first line of each (valid Jupyter magic).

Key EB-NeRD findings reproduced:
- Recency dominates RETRIEVAL (~0.96) but collapses on RERANKING (signal inversion)
- Popularity is the strongest reranking signal
- Frozen popularity 0.568 -> rolling 0.773 (concept drift, the +0.205 headline)
- Content (BM25/E5) weak on Danish (~0.037 retrieval, ~0.52 reranking)
- Session features add nothing (short sessions)
