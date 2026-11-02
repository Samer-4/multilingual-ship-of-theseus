"""Bring Samer's round-trip translation RQ1 scores into the SoT analysis (notebook §5), and check that they use the
same metric definitions as pilot.metrics.

Samer's translations are not in the repo (corpus licence), but his branch publishes text-free per-pair scores
(`Sam/results/research/metrics/pair_metrics.parquet`, keyed by text SHA-256) and the item manifest mapping his 200
XSum T0 documents to (dataset, source, key). This script:
  1. extracts both from the branch into <derived>/samer_pair_metrics.parquet and samer_items.csv;
  2. rescores his 368 baseline pairs (unrelated and cross-source T0 pairs, i.e. texts we have) with pilot.metrics
     and saves them to <derived>/samer_baselines_rescored.parquet, so the notebook can show per-pair agreement.
~3 min on a laptop (MPS).
Usage: python scripts/import_samer_rq1.py [--branch origin/feature/sam-experiments] [--derived DIR] [--sot DIR]
"""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from pilot.config import ROOT, load_config
from pilot.metrics import (extra_lexical_scores, extract_info, information_scores, lexical_scores, load_corpus,
                           score_neural_extra, score_semantic, text_hash)

FILES = {"samer_pair_metrics.parquet": "Sam/results/research/metrics/pair_metrics.parquet",
         "samer_items.csv": "Sam/data/manifests/xsum_madlad200_ids.csv",
         # RQ2 (source attribution): his per-text predictions, fold assignments, stylometric reference lists and
         # text-free per-text features, used to validate the port in pilot.attribution
         "samer_rq2_predictions.parquet": "Sam/results/research/rq2/predictions.parquet",
         "samer_rq2_splits.csv": "Sam/results/research/rq2/splits.csv",
         "samer_reference_lists.json": "Sam/results/research/shared_features/reference_lists.json",
         "samer_text_features.parquet": "Sam/results/research/shared_features/text_features.parquet",
         # RQ3 (rewriter fingerprint): his translator-identification predictions and folds
         "samer_rq3_predictions.parquet": "Sam/results/research/rq3/predictions.parquet",
         "samer_rq3_splits.csv": "Sam/results/research/rq3/splits.csv"}
ID = ["dataset", "floor_id"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="origin/feature/sam-experiments")
    ap.add_argument("--derived", type=Path)
    ap.add_argument("--sot", type=Path)
    ap.add_argument("--files-only", action="store_true", help="only extract the files; skip the baseline rescoring")
    a = ap.parse_args()
    cfg = load_config()
    D, SOT = a.derived or cfg["paths"]["derived"], a.sot or cfg["paths"]["sot"]

    for name, path in FILES.items():
        blob = subprocess.run(["git", "show", f"{a.branch}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
        (D / name).write_bytes(blob)
        print(f"{name}: {len(blob) / 1e3:.0f} kB from {a.branch}:{path}", flush=True)
    if a.files_only:
        raise SystemExit(0)

    c = load_corpus(SOT)
    t0 = c[(c.dataset == "xsum") & (c["round"] == 0) & (c.variant == "original")].drop_duplicates("text")
    by_sha = {hashlib.sha256(t.encode()).hexdigest(): t for t in t0.text}
    items = pd.read_csv(D / "samer_items.csv")
    assert items.t0_sha256.isin(by_sha).all(), "Samer's T0 texts differ from our corpus copy"

    pm = pd.read_parquet(D / "samer_pair_metrics.parquet")
    b = pm[pm.kind.isin(["unrelated", "cross_source"])].reset_index(drop=True)
    b = b.assign(ref=b.ref_sha256.map(by_sha), hyp=b.hyp_sha256.map(by_sha), floor_id=range(len(b)), dataset="xsum")
    assert b.ref.notna().all() and b.hyp.notna().all(), "baseline texts not found in our corpus"

    lex = [{**lexical_scores(r.ref, r.hyp, ENGLISH_STOP_WORDS), **extra_lexical_scores(r.ref, r.hyp)}
           for r in b.itertuples()]
    info = extract_info(list(pd.unique(pd.concat([b.ref, b.hyp])))).set_index("text_hash")
    inf = [information_scores(info.loc[text_hash(r.ref)], r.hyp, info.loc[text_hash(r.hyp)]) for r in b.itertuples()]

    def jobs():
        return (({"dataset": "xsum", "floor_id": r.floor_id}, [r.ref, r.hyp], [(0, 1, {"kind": r.kind})])
                for r in b.itertuples())

    tmp = {k: D / f"samer_baselines_{k}.tmp.parquet" for k in ("sem", "neu")}
    for p in tmp.values():
        p.unlink(missing_ok=True)
    sem = score_semantic(jobs(), tmp["sem"], ID, jobs_per_batch=8)
    neu = score_neural_extra(jobs(), {h: json.loads(s) for h, s in info.sentences_json.items()}, tmp["neu"], ID,
                             jobs_per_batch=8)
    out = (pd.concat([b[["floor_id", "kind", "item_id", "ref_sha256", "hyp_sha256"]], pd.DataFrame(lex),
                      pd.DataFrame(inf)], axis=1)
           .merge(sem.drop(columns=["dataset", "kind"]), on="floor_id", validate="1:1")
           .merge(neu.drop(columns=["dataset", "kind"]), on="floor_id", validate="1:1"))
    out.to_parquet(D / "samer_baselines_rescored.parquet", index=False)
    for p in tmp.values():
        p.unlink(missing_ok=True)
    print(f"rescored {len(out)} baseline pairs -> samer_baselines_rescored.parquet", flush=True)
