"""Background runner for the notebook's semantic scoring (BERTScore + sentence-embedding cosine).

Same functions as notebooks/sot_analysis.ipynb §3; writes the same resumable caches, so the notebook
just loads them. Usage: python scripts/score_semantic.py
"""
import pandas as pd

from pilot.config import load_config
from pilot.metrics import chain_jobs, load_corpus, score_semantic

ID = ["dataset", "source", "key"]
VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]

cfg = load_config()
D = cfg["paths"]["derived"]
sample = pd.read_parquet(D / "sot_sample.parquet")
floors = pd.read_parquet(D / "sot_floors.parquet")
texts = load_corpus(cfg["paths"]["sot"]).set_index(ID + ["variant", "round"]).text

score_semantic(chain_jobs(sample, texts, VARIANTS, ID), D / "sot_pairs_semantic.parquet", ID)
floor_jobs = ((({"dataset": r.dataset, "floor_id": r.floor_id}), [r.ref, r.hyp], [(0, 1, {"kind": "floor"})])
              for r in floors.itertuples(index=False))
score_semantic(floor_jobs, D / "sot_floors_semantic.parquet", ["dataset", "floor_id"])
