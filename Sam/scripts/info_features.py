"""Named entities, numbers, quantities and spaCy sentences for every text the analysis scores (Samer's
information-preservation inputs). spaCy multiprocessing deadlocks inside Jupyter on macOS, hence a script.
~40 min for the full corpus, so it runs on Colab (notebooks/colab_extra_scoring.ipynb).

Writes <derived>/sot_info_features.parquet (text_hash + JSON columns) and the slim sot_sentences.parquet
(text_hash, sentences_json) that score_neural_extra.py needs. Saves every chunk to a parts folder, so an
interrupted run resumes.
Usage: python scripts/info_features.py [--derived DIR] [--sot DIR]
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd

from pilot.config import load_config
from pilot.metrics import build_pairs, extract_info, load_corpus, text_hash

VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--sot", type=Path)
    ap.add_argument("--chunk", type=int, default=20_000, help="texts per saved part")
    a = ap.parse_args()
    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]

    pairs = build_pairs(load_corpus(SOT), pd.read_parquet(D / "sot_sample.parquet"), VARIANTS)
    floors = pd.read_parquet(D / "sot_floors.parquet")
    texts = list(pd.unique(pd.concat([pairs.ref, pairs.hyp, floors.ref, floors.hyp])))

    parts = D / "sot_info_features.parts"
    parts.mkdir(exist_ok=True)
    known = set()
    for p in parts.glob("part_*.parquet"):
        known |= set(pd.read_parquet(p, columns=["text_hash"]).text_hash)
    todo = [t for t in texts if text_hash(t) not in known]
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(texts):,} texts needed, {len(texts) - len(todo):,} cached, {len(todo):,} to parse, {workers} workers",
          flush=True)
    started, n0 = time.time(), len(list(parts.glob("part_*.parquet")))
    for i, s in enumerate(range(0, len(todo), a.chunk)):
        out = extract_info(todo[s:s + a.chunk], n_process=workers)
        tmp = parts / f"part_{n0 + i:04d}.tmp.parquet"
        out.to_parquet(tmp, index=False)
        tmp.replace(parts / f"part_{n0 + i:04d}.parquet")
        print(f"info chunk {i + 1}/{-(-len(todo) // a.chunk)} done, {time.time() - started:.0f}s", flush=True)

    info = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part_*.parquet"))], ignore_index=True)
    info = info.drop_duplicates("text_hash")
    assert {text_hash(t) for t in texts} <= set(info.text_hash), "info parts don't cover every text"
    info.to_parquet(D / "sot_info_features.parquet", index=False)
    info[["text_hash", "sentences_json"]].to_parquet(D / "sot_sentences.parquet", index=False)
    print(f"info features done: {len(info):,} texts, {time.time() - started:.0f}s this session", flush=True)
