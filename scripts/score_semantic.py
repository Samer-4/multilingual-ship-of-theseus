"""Background runner for the notebook's semantic scoring (rescaled BERTScore + length-weighted MPNet cosine).

Same functions as notebooks/sot_analysis.ipynb §2; writes the same resumable caches, so the notebook just
loads them. Reads data/derived/sot_sample.parquet and sot_floors.parquet (written by the notebook's §2).

Usage:
  python scripts/score_semantic.py                                   # Mac / defaults (fp32, small batches)
  python scripts/score_semantic.py --jobs-per-batch 32 --batch-size 64 [--dtype fp16]   # large CUDA GPU
  python scripts/score_semantic.py --derived /content/drive/MyDrive/sot/derived ...     # caches elsewhere
"""
import argparse
from pathlib import Path

import pandas as pd

from pilot.config import load_config
from pilot.metrics import chain_jobs, load_corpus, score_semantic

ID = ["dataset", "source", "key"]
VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32", help="fp16 applies on CUDA only")
    ap.add_argument("--jobs-per-batch", type=int, default=1, help="originals whose texts are embedded together")
    ap.add_argument("--batch-size", type=int, default=16, help="sequences per forward pass")
    ap.add_argument("--derived", type=Path, help="derived-data folder (default: config paths.derived)")
    ap.add_argument("--sot", type=Path, help="corpus folder (default: config paths.sot)")
    ap.add_argument("--limit", type=int, help="score only the first N originals (for timing / validation)")
    ap.add_argument("--suffix", default="", help="cache-name suffix, e.g. _fp16check")
    a = ap.parse_args()

    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]
    sample = pd.read_parquet(D / "sot_sample.parquet")
    floors = pd.read_parquet(D / "sot_floors.parquet")
    if a.limit:
        sample = sample.head(a.limit)
    texts = load_corpus(SOT).set_index(ID + ["variant", "round"]).text
    opts = dict(dtype=a.dtype, jobs_per_batch=a.jobs_per_batch, batch_size=a.batch_size)
    print(f"{len(sample):,} originals, {len(floors):,} floor pairs | {opts}", flush=True)

    score_semantic(chain_jobs(sample, texts, VARIANTS, ID), D / f"sot_pairs_semantic_v2{a.suffix}.parquet", ID, **opts)
    if not a.limit:
        floor_jobs = (({"dataset": r.dataset, "floor_id": r.floor_id}, [r.ref, r.hyp], [(0, 1, {"kind": "floor"})])
                      for r in floors.itertuples(index=False))
        score_semantic(floor_jobs, D / f"sot_floors_semantic_v2{a.suffix}.parquet", ["dataset", "floor_id"], **opts)
