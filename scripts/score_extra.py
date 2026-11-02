"""CPU half of Samer's extra metrics for every pair and floor pair: chrF, word edit rate, exact equality
(`lexical`, ~4 min, fine locally) and entity / number / quantity recall (`information`, needs
scripts/info_features.py first; runs on Colab in notebooks/colab_extra_scoring.ipynb). Chunked and resumable.
Usage: python scripts/score_extra.py [lexical|information|all] [--derived DIR] [--sot DIR]
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from pilot.config import load_config
from pilot.metrics import (PAIR_KEYS, build_pairs, extra_lexical_scores, load_corpus, score_information_chunked,
                           score_lexical_chunked)

VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]
FLOOR_KEYS = ["dataset", "floor_id"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["lexical", "information", "all"])
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--sot", type=Path)
    a = ap.parse_args()
    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]
    workers = max(1, (os.cpu_count() or 4) - 2)
    pairs = build_pairs(load_corpus(SOT), pd.read_parquet(D / "sot_sample.parquet"), VARIANTS)
    floors = pd.read_parquet(D / "sot_floors.parquet")
    print(f"{len(pairs):,} pairs, {len(floors):,} floor pairs, {workers} workers", flush=True)

    if a.what in ("lexical", "all"):
        floor_frame = floors.assign(source="", key="", variant="", round=0, kind="floor")
        score_lexical_chunked(pairs, D / "sot_pairs_extra_lexical_full.parquet", workers, fn=extra_lexical_scores)
        score_lexical_chunked(floor_frame, D / "sot_floors_extra_lexical.parquet", workers, fn=extra_lexical_scores)
        print("extra lexical done", flush=True)
    if a.what in ("information", "all"):
        info = pd.read_parquet(D / "sot_info_features.parquet")
        score_information_chunked(pairs, info, D / "sot_pairs_information_full.parquet", PAIR_KEYS, workers)
        score_information_chunked(floors, info, D / "sot_floors_information.parquet", FLOOR_KEYS, workers)
        print("information done", flush=True)
