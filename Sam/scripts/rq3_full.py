"""RQ3 rewriter fingerprint on the FULL corpus: can a classifier tell which paraphraser (ChatGPT, PaLM, Dipper,
Pegasus) rewrote a text? Samer's RQ3 protocol (pilot.attribution, as in scripts/rq3_fingerprint.py), run per dataset
on all 7 datasets.

Reuses the RQ2 full-corpus run: its texts (rq2_full_texts.parquet) and its full_text stylometric features
(rq2_full_features.parts/full_text_*.parquet), so there is no new spaCy pass. Run scripts/rq2_full.py first.

Cohort, per dataset: originals that all 4 paraphrasers completed at T1, T2 and T3 (non-empty), so every class has the
same documents at every round (asserted). Experiments, per dataset, article-grouped 5 folds x `--repeats` shufflings
(seed 6120):
  main          train on pooled T1-T3, evaluate each round; 5 models
  cross_round   train on one round, evaluate every round (char n-grams, stylometric)
  loso          leave one original source out of training, test on it (char n-grams, stylometric; min(2, repeats))
  typography    curly quotes/apostrophes/long dashes mapped to ASCII (char n-grams) or their rates removed
                (stylometric)
--repeats defaults to 1 (protocol_rq3.md §8: ~1 h on a 12-vCPU Colab runtime); --repeats 5 is Samer's full protocol
(~2.5-3 h).

Outputs (resumable: finished jobs are skipped):
  rq3_full_preds.parts/<dataset>__<experiment>__<model>[__<extra>].parquet
  rq3_full_predictions.parquet      every prediction, dictionary-encoded
  rq3_full_pool_features.parquet    text-free descriptives per text (T0 and the pool): lengths, sentence length,
                                    MATTR, punctuation rates, typography flag
Usage: python scripts/rq3_full.py [--derived DIR] [--limit N] [--repeats R] [--workers N]
  --limit N reads the RQ2 smoke-test files (rq2_full_*_limitN) and writes rq3_full_*_limitN
"""
import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import pandas as pd

from pilot.attribution import LENGTH_COLS, add_length_cols, grouped_cv, style_columns
from pilot.config import load_config

PARAPHRASERS = ["chatgpt", "palm", "dipper", "pegasus(full)"]
STAGES = ["T1", "T2", "T3"]
SENT = ["sent_len_mean", "sent_len_sd", "punct_00", "dep_ROOT", "cap_rate"]     # Samer's segmentation control
TYPO_COLS = ["punct_11", "punct_12", "punct_13", "punct_14", "punct_15", "punct_16"]
TYPO = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
TYPO_CHARS = "[‘’“”–—]"
KEEP = ["text_id", "article_id", "source_label", "stage", "y_true", "y_pred", "repeat", "model", "experiment",
        "train_stage", "held_out_source", "dataset"]
CATS = [c for c in KEEP if c != "repeat"]
DESCRIPTIVE = ["n_words", "n_sentences", "sent_len_mean", "word_len_mean", "mattr"]
F = {}   # pool, typography-normalised pool, column lists, repeats; filled before forking the workers


def build_pool(t, feats):
    p = t[t.rewriter.isin(PARAPHRASERS) & t.stage.isin(STAGES)]
    n = p.groupby("item_id").size()                       # one text per (paraphraser, round) after rq2_full's dedup
    p = p[p.item_id.isin(n[n == len(PARAPHRASERS) * len(STAGES)].index)]
    # the features' own n_words (regex words, Samer's definition) feeds the length columns, as in rq2_full.py
    p = add_length_cols(p.drop(columns=["n_words", "opening_words"]).merge(feats, on="text_id", validate="1:1"))
    bal = p.groupby(["dataset", "stage", "rewriter"]).size().unstack()
    assert bal.nunique(axis=1).eq(1).all(), "classes must be balanced within each dataset and round"
    return p.reset_index(drop=True)


def jobs_for(dataset, sources):
    j = [(dataset, "main", m, None) for m in ("tfidf_char", "tfidf_word", "stylometric", "stylometric_no_sentence",
                                              "length_only")]
    j += [(dataset, "cross_round", m, s) for m in ("tfidf_char", "stylometric") for s in STAGES]
    j += [(dataset, "loso", m, src) for m in ("tfidf_char", "stylometric") for src in sources]
    j += [(dataset, "typography", m, None) for m in ("tfidf_char_typography_normalised", "stylometric_no_typography")]
    return j


def part_path(PP, job):
    d, e, m, x = job
    return PP / (f"{d}__{e}__{m}" + (f"__{x}" if x else "") + ".parquet")


def cv_job(arg):
    (dataset, exp, model, extra), out = arg
    pool = F["pool"][F["pool"].dataset == dataset]
    evals = {s: pool[pool.stage == s] for s in STAGES}
    cols = {"stylometric": F["style"], "stylometric_no_sentence": F["no_sentence"], "length_only": LENGTH_COLS,
            "stylometric_no_typography": F["no_typography"]}.get(model)
    kind = "stylometric" if model.startswith("stylometric") else model
    t = time.time()
    if exp == "main":
        pr = grouped_cv(pool, evals, "rewriter", kind, cols=cols, repeats=F["repeats"])
    elif exp == "cross_round":
        pr = grouped_cv(pool[pool.stage == extra], evals, "rewriter", kind, cols=cols, repeats=F["repeats"])
    elif exp == "loso":
        pr = grouped_cv(pool, {extra: pool[pool.source_label == extra]}, "rewriter", kind, cols=cols,
                        repeats=min(2, F["repeats"]), train_filter=lambda tr: tr[tr.source_label != extra])
    else:   # typography
        if model == "tfidf_char_typography_normalised":
            pool = F["pool_typo"][F["pool_typo"].dataset == dataset]
            evals, kind = {s: pool[pool.stage == s] for s in STAGES}, "tfidf_char"
        pr = grouped_cv(pool, evals, "rewriter", kind, cols=cols, repeats=F["repeats"])
    pr = pr.assign(model=model, experiment=exp, dataset=dataset,
                   train_stage=extra if exp == "cross_round" else None,
                   held_out_source=extra if exp == "loso" else None)[KEEP]
    for c in CATS:
        pr[c] = pr[c].astype("string").astype("category")
    pr["repeat"] = pr["repeat"].astype("int8")
    tmp = out.with_suffix(".tmp.parquet")
    pr.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"{dataset} {exp} {model}{' ' + extra if extra else ''}: {len(pr):,} predictions, {time.time() - t:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--limit", type=int, help="use the RQ2 smoke-test files for N articles per dataset")
    ap.add_argument("--repeats", type=int, default=1, help="fold shufflings (1 recommended, 5 = Samer's protocol)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()
    D = a.derived or load_config()["paths"]["derived"]
    sfx = f"_limit{a.limit}" if a.limit else ""
    PP = D / f"rq3_full_preds{sfx}_r{a.repeats}.parts"
    t0 = time.time()

    t = pd.read_parquet(D / f"rq2_full_texts{sfx}.parquet")
    parts = sorted((D / f"rq2_full_features{sfx}.parts").glob("full_text_*.parquet"))
    feats = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert len(feats) == len(t), "RQ2 full_text features incomplete: run scripts/rq2_full.py features first"

    pool = build_pool(t, feats)
    F["pool"], F["repeats"] = pool, a.repeats
    F["pool_typo"] = pool.assign(text=pool.text.str.translate(TYPO))
    F["style"] = style_columns(pool)
    F["no_sentence"] = [c for c in F["style"] if c not in SENT]
    F["no_typography"] = [c for c in F["style"] if c not in TYPO_COLS]
    per = pool.groupby("dataset").agg(documents=("item_id", "nunique"), articles=("article_id", "nunique"),
                                      texts=("text_id", "size"))
    print(f"cohort: {per.documents.sum():,} of {t[t.stage == 'T0'].item_id.nunique():,} originals "
          f"({per.articles.sum():,} articles), {len(pool):,} texts, {len(F['style'])} stylometric features, "
          f"{a.repeats} repeat(s)\n{per.to_string()}", flush=True)

    # text-free descriptives for the confound checks (T0 rows give the length ratios)
    punct = [c for c in feats.columns if c.startswith("punct_")]
    desc = t[t.stage.eq("T0") | t.text_id.isin(pool.text_id)]
    desc = desc.assign(has_typography=desc.text.str.contains(TYPO_CHARS))[
        ["text_id", "dataset", "item_id", "article_id", "source_label", "rewriter", "stage", "has_typography"]]
    desc.merge(feats[["text_id", *DESCRIPTIVE, *punct]], on="text_id", validate="1:1") \
        .to_parquet(D / f"rq3_full_pool_features{sfx}.parquet", index=False)

    PP.mkdir(exist_ok=True)
    jobs = [(j, part_path(PP, j)) for d in sorted(pool.dataset.unique())
            for j in jobs_for(d, sorted(pool[pool.dataset == d].source_label.unique()))]
    todo = [j for j in jobs if not j[1].exists()]
    # slowest first: char then word n-grams, on the largest datasets
    size = pool.groupby("dataset").size().to_dict()
    cost = {"tfidf_char": 3, "tfidf_char_typography_normalised": 3, "tfidf_word": 2}
    todo.sort(key=lambda j: -cost.get(j[0][2], 1) * size[j[0][0]] * (1 if j[0][1] in ("main", "typography") else 0.6))
    print(f"cross-validation: {len(jobs)} jobs, {len(todo)} to run, {a.workers} workers", flush=True)
    with mp.get_context("fork").Pool(a.workers) as p:
        for k, msg in enumerate(p.imap_unordered(cv_job, todo), 1):
            print(f"cv {k}/{len(todo)} {msg} | {time.time() - t0:.0f}s", flush=True)

    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.concat_tables([pq.read_table(j[1], read_dictionary=CATS) for j in jobs],
                             promote_options="permissive").unify_dictionaries()
    pq.write_table(table, D / f"rq3_full_predictions{sfx}.parquet")
    print(f"done: {table.num_rows:,} predictions -> rq3_full_predictions{sfx}.parquet, {time.time() - t0:.0f}s",
          flush=True)
