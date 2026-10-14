"""Background runner for the notebook's style features (spaCy is slow and its multiprocessing
deadlocks inside Jupyter on macOS). Writes data/derived/sot_style_features.parquet keyed by text hash.
Usage: python scripts/style_features.py
"""
import json
import time

import pandas as pd

from pilot.config import load_config
from pilot.metrics import extract_style, load_corpus, load_reference, reference_subset, text_hash, top_function_words

ID = ["dataset", "source", "key"]
VARIANTS = ["chatgpt", "palm", "dipper", "pegasus(full)", "dipper(low)", "dipper(high)", "pegasus(slight)"]

if __name__ == "__main__":
    cfg = load_config()
    D = cfg["paths"]["derived"]
    corpus = load_corpus(cfg["paths"]["sot"])
    sample = pd.read_parquet(D / "sot_sample.parquet")
    floors = pd.read_parquet(D / "sot_floors.parquet")
    ref_sub = reference_subset(load_reference(cfg["paths"]["sot"]), 300, cfg["seed"])
    fw = top_function_words(ref_sub.text)

    in_sample = corpus.merge(sample, on=ID)
    in_sample = in_sample[in_sample.variant.isin(VARIANTS + ["original"])]
    texts = list(pd.unique(pd.concat([in_sample.text, floors.ref, floors.hyp, ref_sub.text])))
    # Resumable: texts already in the cache (keyed by hash) are skipped, so a larger scope reuses earlier work.
    cache = D / "sot_style_features.parquet"
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["text_hash"])
    known = set(done.text_hash)
    todo = [t for t in texts if text_hash(t) not in known]
    print(f"{len(texts):,} texts needed, {len(texts) - len(todo):,} cached, {len(todo):,} to parse", flush=True)
    started = time.time()
    if todo:
        feats = pd.concat([done, extract_style(todo, fw, n_process=6)], ignore_index=True)
        feats.to_parquet(cache, index=False)
    (D / "sot_function_words.json").write_text(json.dumps(fw))
    print(f"style features for {len(todo):,} new texts in {time.time() - started:.0f}s", flush=True)
