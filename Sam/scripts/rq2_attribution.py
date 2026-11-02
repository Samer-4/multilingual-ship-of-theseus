"""RQ2 source attribution on paraphrases, with Samer's protocol (pilot.attribution), on his 200 XSum T0 documents.

Train on T0 texts of the training articles (5 article-grouped folds x 5 shufflings, seed 6120), predict the held-out
articles' T0 texts and each paraphraser's T1-T3 rewrites. Three text variants:
  full_text          the texts as they are
  opening_removed    the measured shared opening (identical across sources of an article) removed from every text
  truncated_50       opening removed, then every text cut to its first 50 words (length control)
Writes <derived>/rq2_paraphrase_predictions.parquet, rq2_texts.parquet (ids, opening length, word counts; no text)
and rq2_features.parquet (Samer's stylometric features per text variant, keyed by text_id + variant).
~5 min on a laptop. Needs scripts/import_samer_rq1.py --files-only first.
Usage: python scripts/rq2_attribution.py [--derived DIR] [--sot DIR] [--words 50]
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from pilot.attribution import (LENGTH_COLS, MODELS, add_length_cols, grouped_cv, shared_opening_lengths,
                               strip_opening, style_columns, stylometric_features)
from pilot.config import load_config
from pilot.metrics import load_corpus

PARAPHRASERS = ["chatgpt", "palm", "dipper", "pegasus(full)"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--sot", type=Path)
    ap.add_argument("--words", type=int, default=50, help="truncation length for the length control")
    a = ap.parse_args()
    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]
    t0 = time.time()

    items = pd.read_csv(D / "samer_items.csv")[["item_id", "article_id", "source_label"]]
    c = load_corpus(SOT)
    c = c[(c.dataset == "xsum") & c.variant.isin(["original"] + PARAPHRASERS)].copy()
    c["item_id"] = "xsum|" + c.source + "|" + c.key
    c = c[c.item_id.isin(items.item_id)]
    c = c[(c.variant == "original") & (c["round"] == 0) | (c.variant != "original") & (c["round"] > 0)]
    c = c.drop_duplicates(["item_id", "variant", "round"])
    texts = c.merge(items, on="item_id", validate="m:1").rename(columns={"variant": "rewriter"})
    texts["stage"] = "T" + texts["round"].astype(str)
    texts["text_id"] = texts.item_id + "|" + texts.rewriter + "|" + texts.stage
    texts = texts[["text_id", "item_id", "article_id", "source_label", "rewriter", "stage", "text"]].reset_index(drop=True)
    texts = texts[texts.text.str.strip().str.len() > 0].reset_index(drop=True)
    print(f"{len(texts):,} texts: {texts.groupby(['rewriter', 'stage']).size().to_dict()}", flush=True)

    texts["opening_words"] = shared_opening_lengths(texts)
    stripped = strip_opening(texts, texts.opening_words)
    trunc = stripped.assign(text=stripped.text.map(lambda t: " ".join(t.split()[:a.words])))
    VARIANTS = {"full_text": texts, "opening_removed": stripped, f"truncated_{a.words}": trunc}

    ref = json.loads((D / "samer_reference_lists.json").read_text())
    workers = max(1, (os.cpu_count() or 4) - 2)
    feats = []
    for name, df in VARIANTS.items():
        f = stylometric_features(df.text.tolist(), ref, n_process=workers)
        feats.append(f.assign(text_id=df.text_id.to_numpy(), variant=name))
        print(f"features {name}: {len(f):,} texts, {time.time() - t0:.0f}s", flush=True)
    feats = pd.concat(feats, ignore_index=True)
    STYLE = style_columns(feats)
    feats.to_parquet(D / "rq2_features.parquet", index=False)

    preds = []
    for name, df in VARIANTS.items():
        df = add_length_cols(df.merge(feats[feats.variant == name].drop(columns="variant"), on="text_id",
                                      validate="1:1"))
        evals = {"original|T0": df[df.stage == "T0"]}
        for p in PARAPHRASERS:
            for k in (1, 2, 3):
                evals[f"{p}|T{k}"] = df[(df.rewriter == p) & (df.stage == f"T{k}")]
        for m in MODELS:
            cols = {"stylometric": STYLE, "length_only": LENGTH_COLS}.get(m)
            preds.append(grouped_cv(df[df.stage == "T0"], evals, "source_label", m, cols=cols).assign(ablation=name))
        print(f"cross-validation {name}: done, {time.time() - t0:.0f}s", flush=True)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(D / "rq2_paraphrase_predictions.parquet", index=False)
    texts.assign(n_words=texts.text.str.split().str.len()).drop(columns="text").to_parquet(D / "rq2_texts.parquet",
                                                                                          index=False)
    print(f"done: {len(preds):,} predictions, {len(STYLE)} stylometric features, {time.time() - t0:.0f}s", flush=True)
