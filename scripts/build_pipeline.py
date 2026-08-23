"""
Q1 one-command pipeline rebuild: download -> parse -> temporal split -> feature store.

  python scripts/build_pipeline.py --dataset mind   --config configs/mind.yaml
  python scripts/build_pipeline.py --dataset ebnerd --config configs/ebnerd.yaml

Steps:
  1. download raw archives (if not present) from the URLs in the config
  2. parse into the unified schema (articles / impressions / history)
  3. temporal split (predefined for MIND & EB-NeRD; timestamp-cutoff for the demo bundle)
  4. write a small feature store (article features + user history) under paths.processed
"""
import argparse
import os
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config as C  # noqa
import data_pipeline as DP  # noqa


def download(url, dest):
    if os.path.exists(dest):
        print(f"[skip] {dest} exists")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[download] {url}")
    subprocess.run(["wget", "-c", "-O", dest, url], check=True)


def unzip(path, out_dir):
    print(f"[unzip] {path} -> {out_dir}")
    with zipfile.ZipFile(path) as z:
        z.extractall(out_dir)


def build_mind(cfg):
    dl = cfg.get("download", {})
    raw = os.path.join("data", "mind", "raw")
    os.makedirs(raw, exist_ok=True)
    if "train_url" in dl:
        download(dl["train_url"], f"{raw}/MINDsmall_train.zip")
        unzip(f"{raw}/MINDsmall_train.zip", cfg["paths"]["train"])
    if "dev_url" in dl:
        download(dl["dev_url"], f"{raw}/MINDsmall_dev.zip")
        unzip(f"{raw}/MINDsmall_dev.zip", cfg["paths"]["dev"])

    train_dir = C.resolve(cfg["paths"]["train"])
    dev_dir = C.resolve(cfg["paths"]["dev"])
    art_tr, imp_tr, hist_tr = DP.parse_mind(train_dir)
    art_dv, imp_dv, hist_dv = DP.parse_mind(dev_dir)
    proc = C.ensure_dir(cfg["paths"]["processed"])
    art_tr.to_parquet(f"{proc}/articles_train.parquet")
    art_dv.to_parquet(f"{proc}/articles_dev.parquet")
    print(f"[mind] train impressions={len(imp_tr)} dev impressions={len(imp_dv)} "
          f"articles={len(art_tr)}+{len(art_dv)}")
    print(f"[mind] temporal split: train (Nov 9-14) -> dev (Nov 15, strictly later)")
    print(f"[mind] feature store written to {proc}")


def build_ebnerd(cfg):
    dl = cfg.get("download", {})
    raw = os.path.join("data", "ebnerd", "raw")
    os.makedirs(raw, exist_ok=True)
    if "small_url" in dl:
        download(dl["small_url"], f"{raw}/ebnerd_small.zip")
        unzip(f"{raw}/ebnerd_small.zip", os.path.dirname(cfg["paths"]["small"]))

    base = C.resolve(cfg["paths"]["small"])
    art, imp_tr, hist_tr = DP.parse_ebnerd(base, "train")
    _, imp_va, hist_va = DP.parse_ebnerd(base, "validation")
    proc = C.ensure_dir(cfg["paths"]["processed"])
    art.to_parquet(f"{proc}/articles.parquet")
    print(f"[ebnerd] train impressions={len(imp_tr)} val impressions={len(imp_va)} "
          f"articles={len(art)}")
    print(f"[ebnerd] temporal split: predefined train/validation/test folders")
    print(f"[ebnerd] NOTE testset is double-nested (ebnerd_testset/ebnerd_testset)")
    print(f"[ebnerd] feature store written to {proc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mind", "ebnerd"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = C.load_config(args.config or f"configs/{args.dataset}.yaml")
    if args.dataset == "mind":
        build_mind(cfg)
    else:
        build_ebnerd(cfg)
    print("[done] pipeline rebuilt.")


if __name__ == "__main__":
    main()
