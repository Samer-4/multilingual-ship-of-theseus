"""RQ3 rewriter fingerprint on paraphrases: can a classifier tell which paraphraser (ChatGPT, PaLM, Dipper, Pegasus)
produced a text? Samer's translator-identification protocol (his RQ3 notebook, pilot.attribution), on his XSum
documents restricted to those every paraphraser completed and every translator completed (so his translator
predictions can be re-scored on the same documents).

Experiments (5 article-grouped folds x 5 shufflings, seed 6120, as in his RQ3):
  main          train on pooled T1-T3 texts, evaluate each round; 5 models (+ stylometric without sentence features)
  cross_round   train on one round, evaluate every round (char n-grams, stylometric)
  loso          leave one original source out of training, test on it (char n-grams, stylometric; 2 shufflings)
  typography    curly quotes/apostrophes/long dashes normalised to ASCII (char n-grams) or their rates removed
                (stylometric)
Needs scripts/import_samer_rq1.py --files-only and scripts/rq2_attribution.py (its stylometric features) first.
Writes <derived>/rq3_paraphrase_predictions.parquet. ~5 min on a laptop.
Usage: python scripts/rq3_fingerprint.py [--derived DIR] [--sot DIR]
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from pilot.attribution import LENGTH_COLS, add_length_cols, grouped_cv, style_columns
from pilot.config import load_config
from pilot.metrics import load_corpus

PARAPHRASERS = ["chatgpt", "palm", "dipper", "pegasus(full)"]
STAGES = ["T1", "T2", "T3"]
SENT = ["sent_len_mean", "sent_len_sd", "punct_00", "dep_ROOT", "cap_rate"]     # Samer's segmentation control
TYPO_COLS = ["punct_11", "punct_12", "punct_13", "punct_14", "punct_15", "punct_16"]
TYPO = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--sot", type=Path)
    a = ap.parse_args()
    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]
    t0 = time.time()

    items = pd.read_csv(D / "samer_items.csv")[["item_id", "article_id", "source_label"]]
    his = pd.read_parquet(D / "samer_rq3_predictions.parquet")
    c = load_corpus(SOT)
    c = c[(c.dataset == "xsum") & c.variant.isin(PARAPHRASERS) & (c["round"] > 0)].copy()
    c["item_id"] = "xsum|" + c.source + "|" + c.key
    c = c.drop_duplicates(["item_id", "variant", "round"])
    c = c[c.text.str.strip().str.len() > 0]
    complete = set.intersection(*(set(c[(c.variant == p) & (c["round"] == 3)].item_id) for p in PARAPHRASERS))
    complete &= set.intersection(*(set(c[(c.variant == p) & (c["round"] == k)].item_id)
                                   for p in PARAPHRASERS for k in (1, 2)))
    cohort = complete & set(his.item_id)
    pool = c[c.item_id.isin(cohort)].merge(items, on="item_id", validate="m:1").rename(columns={"variant": "rewriter"})
    pool["stage"] = "T" + pool["round"].astype(str)
    pool["text_id"] = pool.item_id + "|" + pool.rewriter + "|" + pool.stage
    feats = pd.read_parquet(D / "rq2_features.parquet")
    feats = feats[feats.variant == "full_text"].drop(columns="variant")
    pool = add_length_cols(pool[["text_id", "item_id", "article_id", "source_label", "rewriter", "stage", "text"]]
                           .merge(feats, on="text_id", validate="1:1"))
    STYLE = style_columns(pool)
    bal = pool.groupby(["stage", "rewriter"]).size().unstack()
    assert bal.nunique(axis=1).eq(1).all(), "classes must be balanced"
    print(f"cohort: {len(cohort)} documents, {pool.article_id.nunique()} articles, {len(pool):,} texts", flush=True)

    MODELS = {"tfidf_word": None, "tfidf_char": None, "stylometric": STYLE,
              "stylometric_no_sentence": [c for c in STYLE if c not in SENT], "length_only": LENGTH_COLS}
    kind = lambda m: "stylometric" if m.startswith("stylometric") else m   # noqa: E731
    EVALS = {s: pool[pool.stage == s] for s in STAGES}
    out = []
    for m, cols in MODELS.items():
        out.append(grouped_cv(pool, EVALS, "rewriter", kind(m), cols=cols).assign(model=m, experiment="main"))
    print(f"main done, {time.time() - t0:.0f}s", flush=True)
    for m in ("tfidf_char", "stylometric"):
        for s in STAGES:
            out.append(grouped_cv(pool[pool.stage == s], EVALS, "rewriter", m, cols=MODELS[m])
                       .assign(model=m, experiment="cross_round", train_stage=s))
    print(f"cross-round done, {time.time() - t0:.0f}s", flush=True)
    for m in ("tfidf_char", "stylometric"):
        for src in sorted(pool.source_label.unique()):
            out.append(grouped_cv(pool, {src: pool[pool.source_label == src]}, "rewriter", m, cols=MODELS[m],
                                  repeats=2, train_filter=lambda tr, s=src: tr[tr.source_label != s])
                       .assign(model=m, experiment="loso", held_out_source=src))
    print(f"leave-one-source-out done, {time.time() - t0:.0f}s", flush=True)
    pool_n = pool.assign(text=pool.text.str.translate(TYPO))
    out.append(grouped_cv(pool_n, {s: pool_n[pool_n.stage == s] for s in STAGES}, "rewriter", "tfidf_char")
               .assign(model="tfidf_char_typography_normalised", experiment="typography"))
    out.append(grouped_cv(pool, EVALS, "rewriter", "stylometric", cols=[c for c in STYLE if c not in TYPO_COLS])
               .assign(model="stylometric_no_typography", experiment="typography"))
    preds = pd.concat(out, ignore_index=True)
    preds.to_parquet(D / "rq3_paraphrase_predictions.parquet", index=False)
    # text-free descriptives for the confound checks (word counts, typography share)
    TYPO_CHARS = "[‘’“”–—]"
    pool.assign(has_typography=pool.text.str.contains(TYPO_CHARS)).drop(columns="text") \
        .to_parquet(D / "rq3_pool_features.parquet", index=False)
    print(f"done: {len(preds):,} predictions, {time.time() - t0:.0f}s", flush=True)
