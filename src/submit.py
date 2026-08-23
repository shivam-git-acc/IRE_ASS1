"""
Q5 Codabench submission writer.

Both competitions use the MIND-standard format: one line per impression,
    <impression_id> [rank1,rank2,...]
where rank_j is the rank assigned to the j-th in-view candidate (rank 1 = highest
score). Ranks are a permutation of 1..len(candidates). Line count must equal the number
of test impressions. The file is zipped (prediction.txt for MIND, predictions.txt for
EB-NeRD — note the plural).

Scores the UNLABELED test split with the reranker trained by `src.rerank`:

  python -m src.rerank --dataset mind
  python -m src.submit --dataset mind --out prediction.zip
"""
import argparse
import os
import sys
import zipfile

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ablations"))
import config as C  # noqa
import rerank as RK  # noqa


def write_submission(rows, out_txt, out_zip, inner_name):
    """rows: iterable of (impression_id, ranks_list)."""
    with open(out_txt, "w") as f:
        for iid, ranks in rows:
            f.write(f"{iid} [{','.join(map(str, ranks))}]\n")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_txt, inner_name)
    return out_zip


# inner filenames per competition
INNER_NAME = {
    "mind": "prediction.txt",
    "ebnerd": "predictions.txt",   # plural for EB-NeRD
}


def _ranks(scores):
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def run_mind_submission(cfg, split, model_path):
    import common as CL  # ablations/common.py
    import pandas as pd

    train_dir = C.resolve(cfg["paths"]["train"])
    dev_dir = C.resolve(cfg["paths"]["dev"])
    test_dir = C.resolve(cfg["paths"][split])
    b = CL.load_mind(train_dir, dev_dir, extra_news_dirs=(test_dir,))
    emb_by_id, _ = CL.encode_texts(cfg["embedding_model"], b["ids"], b["texts"],
                                    prefix=cfg.get("embedding_prefix", "none"))
    bm25 = RK.F.BM25(b["corpus"])
    ctx = dict(bm25=bm25, id_to_row=b["id_to_row"], title_tok=b["title_tok"],
               emb_by_id=emb_by_id, cat=b["cat"], log=b["log"],
               first_seen=b["first_seen"], hist=b["hist"])

    beh = pd.read_csv(f"{test_dir}/behaviors.tsv", sep="\t", header=None, names=CL.MIND_BEH,
                       quoting=3)
    beh["t"] = pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    for u, h in zip(beh["user_id"], beh["history"]):
        if isinstance(h, str) and h:
            ctx["hist"].setdefault(CL.mind_pfx(u), [CL.mind_pfx(x) for x in h.split()])

    model = lgb.Booster(model_file=model_path)
    rows = []
    for iid, uid, T, cand in CL.mind_test_impressions(beh):
        X = RK.build_mind_features(uid, T, cand, ctx)
        rows.append((iid, _ranks(model.predict(X))))
    return rows


def run_ebnerd_submission(cfg, split, model_path):
    import common as CL  # ablations/common.py

    base = C.resolve(cfg["paths"][split])
    b = CL.load_ebnerd(base)
    ids = list(b["texts"].keys())
    emb_by_id, _ = CL.encode_texts(cfg["embedding_model"], ids, b["texts"],
                                    prefix=cfg.get("embedding_prefix", "none"))
    tau = cfg.get("features", {}).get("recency_tau_hours", 24.0)
    ctx = dict(emb_by_id=emb_by_id, cat=b["cat"], hist=b["load_hist"]("test"),
               pub=b["pub"], pv=b["pv"], iv=b["iv"], rt=b["rt"], sent=b["sent"],
               recency_tau_hours=tau)

    model = lgb.Booster(model_file=model_path)
    rows = []
    for iid, uid, T, cand in CL.ebnerd_test_impressions(base, "test"):
        X = RK.build_ebnerd_features(uid, T, cand, ctx)
        rows.append((iid, _ranks(model.predict(X))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None,
                     help="mind: large_test (default); ebnerd: testset (default)")
    ap.add_argument("--model", default=None, help="default: artifacts/<dataset>/reranker.txt")
    ap.add_argument("--out", default="prediction.zip")
    args = ap.parse_args()

    cfg = C.load_config(args.config or f"configs/{args.dataset}.yaml")
    model_path = args.model or os.path.join("artifacts", args.dataset, "reranker.txt")
    if not os.path.exists(model_path):
        raise SystemExit(
            f"No trained model at {model_path}. Run `python -m src.rerank --dataset "
            f"{args.dataset}` first."
        )

    if args.dataset == "mind":
        rows = run_mind_submission(cfg, args.split or "large_test", model_path)
    else:
        rows = run_ebnerd_submission(cfg, args.split or "testset", model_path)

    out_txt = os.path.splitext(args.out)[0] + ".txt"
    out_zip = write_submission(rows, out_txt, args.out, INNER_NAME[args.dataset])
    print(f"[submit] dataset={args.dataset} impressions={len(rows)}")
    print(f"[submit] wrote {out_zip}")


if __name__ == "__main__":
    main()
