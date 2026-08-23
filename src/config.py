"""Config loading + path resolution."""
import glob
import os

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(pattern_or_path):
    """Resolve a path that may be a glob (useful on Kaggle where mount paths vary).

    Returns the first match if a glob, else the path unchanged.
    """
    if any(ch in pattern_or_path for ch in "*?[]"):
        hits = glob.glob(pattern_or_path, recursive=True)
        if not hits:
            raise FileNotFoundError(f"No path matched: {pattern_or_path}")
        return hits[0]
    return pattern_or_path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
