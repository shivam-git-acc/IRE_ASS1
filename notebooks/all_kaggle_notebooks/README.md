# Individual Kaggle Notebooks — MIND + EB-NeRD

Self-contained, reproducible Kaggle notebooks — one per component per dataset.
Run any notebook independently (hardcoded paths, no cross-imports).

- `mind/`   — 9 notebooks (01 pipeline, 02 bm25, 03 embedding, 04 baseline,
              05 lightgbm+Q4, 06 frozen/rolling, 07 embedding-model, 08 nrms/hybrid, 10 submission)
- `ebnerd/` — 9 notebooks (01 pipeline, 02 bm25, 03 embedding, 04 baseline,
              05 lightgbm+Q4, 06 frozen/rolling, 07 embedding-model, 09 session, 10 submission)

See each folder's README.md for the run order and what each notebook produces.

All notebooks are faithful to the working submission code (MIND leaderboard 0.6435,
EB-NeRD 0.6976) and reproduce the assignment's metrics and ablations.
