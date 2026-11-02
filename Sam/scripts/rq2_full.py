"""RQ2 source attribution on the FULL corpus: 7 datasets, every completer original (data/derived/sot_sample.parquet)
and its ChatGPT / PaLM / Dipper / Pegasus rewrites T1-T3. Samer's protocol (pilot.attribution), run per dataset:
train on T0 texts of the training articles (5 article-grouped folds x 5 shufflings), predict the held-out articles'
T0 and T1-T3 texts. Three text variants: full_text, opening_removed (primary), truncated_50 (length control).

Stylometric features: Samer's definitions, but the function-word list is the team standard's, fitted on held-out
reference texts (data/derived/sot_style_vocab.json), since his list was fitted on XSum T0 texts only.

~1.5-2 h on a Colab CPU runtime (notebooks/colab_rq2_attribution.ipynb); every stage resumes from its saved parts:
  texts     -> rq2_full_texts.parquet (with text) and rq2_full_texts_meta.parquet (ids, opening words, word counts)
  features  -> rq2_full_features.parts/<variant>_<NNNN>.parquet
  cv        -> rq2_full_preds.parts/<dataset>__<variant>__<model>.parquet, then rq2_full_predictions.parquet
Usage: python scripts/rq2_full.py [texts|features|cv|all] [--derived DIR] [--sot DIR] [--limit N] [--workers N]
  --limit N keeps N articles (with all their sources) per dataset (smoke test)
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pandas as pd

from pilot.attribution import (LENGTH_COLS, MODELS, add_length_cols, grouped_cv, shared_opening_lengths,
                               strip_opening, style_columns, stylometric_features)
from pilot.config import load_config
from pilot.metrics import load_corpus

PARAPHRASERS = ["chatgpt", "palm", "dipper", "pegasus(full)"]
VARIANTS = ["full_text", "opening_removed", "truncated_50"]
CHUNK = 20_000
KEEP = ["text_id", "article_id", "rewriter", "stage", "y_true", "y_pred", "repeat", "model", "dataset", "ablation"]
CATS = [c for c in KEEP if c != "repeat"]
FRAMES = {}   # (variant) -> texts + features; filled before forking the cross-validation workers


def build_texts(D, SOT, limit):
    sample = pd.read_parquet(D / "sot_sample.parquet")
    if limit:   # N whole articles (all their sources) per dataset
        keys = sample.drop_duplicates(["dataset", "key"]).groupby("dataset", group_keys=False).head(limit)
        sample = sample.merge(keys[["dataset", "key"]], on=["dataset", "key"])
    c = load_corpus(SOT)
    c = c[c.variant.isin(["original"] + PARAPHRASERS)]
    c = c[(c.variant == "original") & (c["round"] == 0) | (c.variant != "original") & (c["round"] > 0)]
    c = c.merge(sample, on=["dataset", "source", "key"]).drop_duplicates(["dataset", "source", "key", "variant", "round"])
    t = pd.DataFrame({
        "dataset": c.dataset.to_numpy(), "item_id": (c.dataset + "|" + c.source + "|" + c.key).to_numpy(),
        "article_id": (c.dataset + "|" + c.key).to_numpy(), "source_label": c.source.to_numpy(),
        "rewriter": c.variant.to_numpy(), "stage": ("T" + c["round"].astype(str)).to_numpy(), "text": c.text.to_numpy()})
    t["text_id"] = t.item_id + "|" + t.rewriter + "|" + t.stage
    t = t[t.text.str.strip().str.len() > 0].reset_index(drop=True)
    t["opening_words"] = shared_opening_lengths(t)
    t["n_words"] = t.text.str.split().str.len()
    return t


def variant_texts(t, name):
    if name == "full_text":
        return t
    s = strip_opening(t, t.opening_words)
    return s if name == "opening_removed" else s.assign(text=s.text.map(lambda x: " ".join(x.split()[:50])))


def cv_job(job):
    dataset, variant, model, out = job
    df = FRAMES[variant]
    df = df[df.dataset == dataset]
    evals = {"original|T0": df[df.stage == "T0"]}
    for p in PARAPHRASERS:
        for k in (1, 2, 3):
            evals[f"{p}|T{k}"] = df[(df.rewriter == p) & (df.stage == f"T{k}")]
    cols = {"stylometric": FRAMES["style_cols"], "length_only": LENGTH_COLS}.get(model)
    t = time.time()
    pr = grouped_cv(df[df.stage == "T0"], evals, "source_label", model, cols=cols)
    pr = pr.assign(dataset=dataset, ablation=variant)[KEEP]   # slim: ~13M rows over the full corpus
    for c in CATS:
        pr[c] = pr[c].astype("category")
    pr["repeat"] = pr["repeat"].astype("int8")
    tmp = out.with_suffix(".tmp.parquet")
    pr.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"{dataset} {variant} {model}: {len(pr):,} predictions, {time.time() - t:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["texts", "features", "cv", "all"])
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--sot", type=Path)
    ap.add_argument("--limit", type=int, help="articles per dataset (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()
    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]
    sfx = f"_limit{a.limit}" if a.limit else ""
    T, FP, PP = D / f"rq2_full_texts{sfx}.parquet", D / f"rq2_full_features{sfx}.parts", D / f"rq2_full_preds{sfx}.parts"
    t0 = time.time()

    if a.what in ("texts", "all") and not T.exists():
        t = build_texts(D, SOT, a.limit)
        t.to_parquet(T, index=False)
        t.drop(columns="text").to_parquet(D / f"rq2_full_texts_meta{sfx}.parquet", index=False)
        print(f"texts: {len(t):,} ({t.groupby('dataset').size().to_dict()}), {time.time() - t0:.0f}s", flush=True)
    t = pd.read_parquet(T)

    if a.what in ("features", "all"):
        ref = {"function_words": json.loads((D / "sot_style_vocab.json").read_text())["function_words"]}
        FP.mkdir(exist_ok=True)
        for v in VARIANTS:
            vt = variant_texts(t, v)
            n = -(-len(vt) // CHUNK)
            for i in range(n):
                out = FP / f"{v}_{i:04d}.parquet"
                if out.exists():
                    continue
                part = vt.iloc[i * CHUNK:(i + 1) * CHUNK]
                f = stylometric_features(part.text.tolist(), ref, n_process=a.workers)
                f.assign(text_id=part.text_id.to_numpy()).to_parquet(out.with_suffix(".tmp.parquet"), index=False)
                out.with_suffix(".tmp.parquet").replace(out)
                print(f"features {v} chunk {i + 1}/{n} done, {time.time() - t0:.0f}s", flush=True)

    if a.what in ("cv", "all"):
        PP.mkdir(exist_ok=True)
        for v in VARIANTS:
            f = pd.concat([pd.read_parquet(p) for p in sorted(FP.glob(f"{v}_*.parquet"))], ignore_index=True)
            assert len(f) == len(t), f"features for {v} incomplete: run the features stage"
            # the features' own n_words (regex words, Samer's definition) feeds the length columns
            FRAMES[v] = add_length_cols(variant_texts(t, v).drop(columns="n_words").merge(f, on="text_id", validate="1:1"))
        FRAMES["style_cols"] = style_columns(FRAMES["full_text"])
        jobs = [(d, v, m, PP / f"{d}__{v}__{m}.parquet") for d in sorted(t.dataset.unique()) for v in VARIANTS
                for m in MODELS]
        todo = [j for j in jobs if not j[3].exists()]
        print(f"cross-validation: {len(jobs)} jobs, {len(todo)} to run, {a.workers} workers, "
              f"{len(FRAMES['style_cols'])} stylometric features", flush=True)
        # char/word TF-IDF jobs are the slow ones: start them first
        todo.sort(key=lambda j: j[2] not in ("tfidf_char", "tfidf_word"))
        with mp.get_context("fork").Pool(a.workers) as pool:
            for k, msg in enumerate(pool.imap_unordered(cv_job, todo), 1):
                print(f"cv {k}/{len(todo)} {msg} | {time.time() - t0:.0f}s", flush=True)
        # merge as dictionary-encoded Arrow columns (plain strings would need >10 GB of RAM)
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.concat_tables([pq.read_table(j[3], read_dictionary=CATS) for j in jobs]).unify_dictionaries()
        pq.write_table(table, D / f"rq2_full_predictions{sfx}.parquet")
        print(f"done: {table.num_rows:,} predictions -> rq2_full_predictions{sfx}.parquet, {time.time() - t0:.0f}s",
              flush=True)
