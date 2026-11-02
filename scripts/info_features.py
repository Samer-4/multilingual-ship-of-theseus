"""Named entities, numbers, quantities and spaCy sentences for every text the analysis scores (Samer's
information-preservation inputs). spaCy multiprocessing deadlocks inside Jupyter on macOS, hence a script.

Writes data/derived/sot_info_features.parquet (text_hash + JSON columns, resumable) and the slim
sot_sentences.parquet (text_hash, sentences_json) that the GPU script needs for sentence coverage.
Usage: python scripts/info_features.py
"""
import os
import time

import pandas as pd

from pilot.config import load_config
from pilot.metrics import build_pairs, extract_info, load_corpus, text_hash

VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]

if __name__ == "__main__":
    cfg = load_config()
    D = cfg["paths"]["derived"]
    pairs = build_pairs(load_corpus(cfg["paths"]["sot"]), pd.read_parquet(D / "sot_sample.parquet"), VARIANTS)
    floors = pd.read_parquet(D / "sot_floors.parquet")
    texts = list(pd.unique(pd.concat([pairs.ref, pairs.hyp, floors.ref, floors.hyp])))

    cache = D / "sot_info_features.parquet"
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["text_hash"])
    known = set(done.text_hash)
    todo = [t for t in texts if text_hash(t) not in known]
    print(f"{len(texts):,} texts needed, {len(texts) - len(todo):,} cached, {len(todo):,} to parse", flush=True)
    started = time.time()
    if todo:
        done = pd.concat([done, extract_info(todo, n_process=max(1, (os.cpu_count() or 4) - 2))], ignore_index=True)
        done.to_parquet(cache, index=False)
    done[["text_hash", "sentences_json"]].to_parquet(D / "sot_sentences.parquet", index=False)
    print(f"info features for {len(todo):,} new texts in {time.time() - started:.0f}s", flush=True)
