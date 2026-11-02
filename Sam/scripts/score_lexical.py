"""Background runner for the notebook's lexical scores at full scale (bounded memory, resumable).
Usage: python scripts/score_lexical.py [scope]   (scope defaults to "full"; reads data/derived/sot_sample.parquet)
"""
import sys

import pandas as pd

from pilot.config import load_config
from pilot.metrics import build_pairs, load_corpus, score_lexical_chunked

VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]

if __name__ == "__main__":
    scope = sys.argv[1] if len(sys.argv) > 1 else "full"
    cfg = load_config()
    D = cfg["paths"]["derived"]
    pairs = build_pairs(load_corpus(cfg["paths"]["sot"]), pd.read_parquet(D / "sot_sample.parquet"), VARIANTS)
    print(f"{len(pairs):,} pairs", flush=True)
    score_lexical_chunked(pairs, D / f"sot_pairs_lexical_{scope}.parquet")
    print("lexical scoring done", flush=True)
