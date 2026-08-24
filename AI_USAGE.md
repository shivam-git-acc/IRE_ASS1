# AI Usage Log — CS4.406 IRE Assignment 1

**Student:** Shivam Patel (2025202030)

Per the assignment's academic-integrity requirements (Q7.4), this log documents all AI
assistance used, links the full chat exports, and marks AI-assisted vs. human-written code.

## AI tools used

| Tool | Role | Used for |
|------|------|----------|
| Anthropic Claude | Primary assistant | Pipeline design, code drafting, debugging, ablation design, evaluation-harness logic, LaTeX design note, repository structuring |
| OpenAI ChatGPT | Secondary assistant | Concept clarification (understanding retrieval/reranking metrics, BM25 and embedding-based retrieval, and general recommender-system concepts) |

## Chat exports

- **Claude conversation:** https://claude.ai/share/6c4c2c56-88c3-49d6-8eb1-767a7b563caa

- **ChatGPT conversation 2 (concept clarification):**
  https://chatgpt.com/share/6a8c0a90-9000-83e9-942e-2de2e7bd7e33

The full prompt-and-response history is contained in the linked/attached exports.

## How AI was used

AI was used as a pair-programming and design assistant, not as an autonomous code
generator. The overall approach, dataset choices, experiment decisions, and the
interpretation of every result were directed by me. Concretely, AI assistance took the
following forms:

- **Concept clarification (ChatGPT):** understanding the distinction between retrieval
  (recall@K over the full pool) and reranking (AUC/MRR/nDCG over the in-view slate),
  how BM25 scoring works, and how embedding-based semantic similarity is computed.
- **Design discussion (Claude):** the unified-schema mapping across MIND and EB-NeRD,
  the rationale for temporal (never random) splitting, leakage-safe point-in-time
  features, and the "where it breaks at 10×" scale analysis.
- **Code drafting (Claude):** initial implementations of the BM25 class, the point-in-time
  feature builders, the LightGBM reranking loop, the evaluation harness (AUC/MRR/nDCG +
  bootstrap CIs + slices), and the Codabench submission writers — which I then adapted to
  the actual dataset paths and executed on Kaggle.
- **Debugging (Claude):** the `run_fast_eval` news-vector caching bug, out-of-memory
  crashes when building feature matrices for millions of impressions, Kaggle interactive-
  session loss recovery (moving long jobs to committed runs), and the E5
  `query:`/`passage:` prefix correction.
- **Ablation & analysis (Claude):** framing the frozen-vs-rolling popularity ablation and
  interpreting the findings — the signal-inversion result, the cross-dataset contrast
  (EB-NeRD popularity/recency-driven vs. MIND CTR/content-driven), and the MiniLM-vs-E5
  model-choice result.
- **Write-up (Claude):** structuring and typesetting the LaTeX design note.

Every AI-suggested code cell was executed and validated by me. All numerical results
reported in the design note and in `RESULTS.md` come from my own runs on Kaggle
(dual Tesla T4), not from AI output.

## AI-generated vs. human-written code

**AI-assisted** — drafted with AI, then edited, path-configured, executed and verified
by me:
- `src/features.py` — BM25 + rolling point-in-time popularity/CTR primitives
- `src/rerank.py` — LightGBM feature matrix + LambdaMART training
- `src/evaluate.py` — AUC/MRR/nDCG + diversity/novelty/coverage + bootstrap CIs + slices
- `src/embeddings.py`, `src/bm25.py` — retrieval + recall@K
- `src/data_pipeline.py`, `scripts/build_pipeline.py`, `scripts/run_pipeline.py`
- `tests/test_no_leakage.py` — Q9 no-future-leakage assertion
- Notebook cells (`notebooks/`) for the MIND/EB-NeRD MiniLM submissions and the NRMS /
  hybrid experiments
- The LaTeX source of the design note

**Human-written / human-directed:**
- All dataset-path configuration and Kaggle environment / secret setup
- All experiment decisions — which datasets, embedding models, features, and ablations
  to run, and in what order
- All execution on Kaggle and all reported results
- All interpretation of results and the findings presented in the design note
- Final review, editing, and verification of every code file
- The GitHub repository organization and commit history

## Declaration

I understand the material in this submission and can explain every design choice and code
path. AI was used as an assistive tool for concept clarification, code drafting, and
debugging; the engineering decisions, execution, verification, and analysis are my own.