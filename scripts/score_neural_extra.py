"""GPU half of Samer's extra metrics: E5 document cosine and MPNet sentence coverage for every pair and floor
pair. Needs data/derived/sot_sentences.parquet (from scripts/info_features.py). Resumable; same options as
scripts/score_semantic.py.

Usage:
  python scripts/score_neural_extra.py --jobs-per-batch 32 --batch-size 64 --derived /content/drive/MyDrive/sot_colab/derived
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pilot.config import load_config
from pilot.metrics import chain_jobs, load_corpus, score_neural_extra

ID = ["dataset", "source", "key"]
VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32", help="fp16 applies on CUDA only")
    ap.add_argument("--jobs-per-batch", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--sot", type=Path)
    ap.add_argument("--limit", type=int, help="score only the first N originals (for timing / validation)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]
    sample = pd.read_parquet(D / "sot_sample.parquet")
    floors = pd.read_parquet(D / "sot_floors.parquet")
    if a.limit:
        sample = sample.head(a.limit)
    sents = pd.read_parquet(D / "sot_sentences.parquet")
    sentences_of = dict(zip(sents.text_hash, sents.sentences_json.map(json.loads)))
    texts = load_corpus(SOT).set_index(ID + ["variant", "round"]).text
    opts = dict(dtype=a.dtype, jobs_per_batch=a.jobs_per_batch, batch_size=a.batch_size)
    print(f"{len(sample):,} originals, {len(floors):,} floor pairs, {len(sentences_of):,} texts with sentences | {opts}",
          flush=True)

    score_neural_extra(chain_jobs(sample, texts, VARIANTS, ID), sentences_of,
                       D / f"sot_pairs_neural_extra{a.suffix}.parquet", ID, **opts)
    if not a.limit:
        floor_jobs = (({"dataset": r.dataset, "floor_id": r.floor_id}, [r.ref, r.hyp], [(0, 1, {"kind": "floor"})])
                      for r in floors.itertuples(index=False))
        score_neural_extra(floor_jobs, sentences_of, D / f"sot_floors_neural_extra{a.suffix}.parquet",
                           ["dataset", "floor_id"], **opts)
