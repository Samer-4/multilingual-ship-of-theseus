"""CPU half of Samer's extra metrics for every pair and floor pair: chrF, word edit rate, exact equality
(no dependencies) and entity / number / quantity recall (needs scripts/info_features.py first).
Chunked and resumable. Usage: python scripts/score_extra.py [lexical|information|all]   (default: all)
"""
import os
import sys

import pandas as pd

from pilot.config import load_config
from pilot.metrics import (PAIR_KEYS, build_pairs, extra_lexical_scores, load_corpus, score_information_chunked,
                           score_lexical_chunked)

VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]
FLOOR_KEYS = ["dataset", "floor_id"]

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    D = cfg["paths"]["derived"]
    workers = max(1, (os.cpu_count() or 4) - 2)
    pairs = build_pairs(load_corpus(cfg["paths"]["sot"]), pd.read_parquet(D / "sot_sample.parquet"), VARIANTS)
    floors = pd.read_parquet(D / "sot_floors.parquet")
    floor_frame = floors.assign(source="", key="", variant="", round=0, kind="floor")
    print(f"{len(pairs):,} pairs, {len(floors):,} floor pairs, {workers} workers", flush=True)

    if what in ("lexical", "all"):
        score_lexical_chunked(pairs, D / "sot_pairs_extra_lexical_full.parquet", workers, fn=extra_lexical_scores)
        score_lexical_chunked(floor_frame, D / "sot_floors_extra_lexical.parquet", workers, fn=extra_lexical_scores)
        print("extra lexical done", flush=True)
    if what in ("information", "all"):
        info = pd.read_parquet(D / "sot_info_features.parquet")
        score_information_chunked(pairs, info, D / "sot_pairs_information_full.parquet", PAIR_KEYS, workers)
        score_information_chunked(floors, info, D / "sot_floors_information.parquet", FLOOR_KEYS, workers)
        print("information done", flush=True)
