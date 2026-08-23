# Ablations

Every ablation below was **actually run** in the committed notebooks in `../notebooks/`.
This file maps each ablation to its source (notebook + cell) and its result, so the
ablations are traceable and reproducible rather than reconstructed. Full numbers are in
`../RESULTS.md`.

The `common.py` here provides shared loaders (MIND + EB-NeRD) and a configurable feature
builder used to re-run the ablations outside the notebooks if desired.

| # | Ablation | Varies | Source | Headline result |
|---|----------|--------|--------|-----------------|
| 1 | Frozen vs rolling popularity | popularity computed once vs strictly-before-T per impression | `dataset-handling.ipynb` cell 39 | 0.568 → **0.773** (leakage-safe + higher) |
| 2 | Session features | add session_cat_match / depth | `dataset-handling.ipynb` cell 35 | 0.756 → 0.757 (null; sessions too short) |
| 3 | Retrieval signals | content vs popularity vs recency, recall@K | `dataset-handling.ipynb` cells 27–30 | recency 0.96 ≫ popularity 0.44 ≫ content 0.04 |
| 4 | Retrieval fusion weights | content/recency/popularity mix | `dataset-handling.ipynb` cell 30 | recency+popularity 0.967; content-heavy drops to 0.87 |
| 5 | Reranking ladder | single → fusion → logistic → LightGBM | `dataset-handling.ipynb` cells 31–32 | popularity 0.693; LightGBM 0.717 |
| 6 | E5 prefix + aggregation | wrong vs correct prefix; mean vs best | MIND runs | E5 0.588 → 0.602; MiniLM 0.671 |
| 7 | E5 vs MiniLM (model choice) | multilingual E5 vs English MiniLM | `mind-minilm.ipynb` | 0.60 → 0.67 (specialisation > size) |
| 8 | NRMS vs LightGBM vs hybrid | neural vs features vs fusion | MIND NRMS runs | LightGBM 0.697 > hybrid 0.676 > NRMS 0.662 |
| 9 | Data scale | small vs 400k-large sample | `mind-minilm.ipynb` | 0.695 ≈ 0.697 (GBDT saturates) |

## Reproducing

Ablations 1–5 live in `dataset-handling.ipynb` (EB-NeRD demo/small) and reproduce by
running that notebook top-to-bottom on the EB-NeRD demo + small bundles. Ablations 6–9
span the MIND runs (`mind-minilm.ipynb` for the MiniLM swap and data-scale check; the
NRMS/hybrid comparison was a separate MIND run summarised in `../RESULTS.md`).

Because the full loops need the datasets mounted on GPU compute (Kaggle), the notebooks
are the primary reproducible artifact; `common.py` supports re-running the core ablations
against the shared `src/` modules.
