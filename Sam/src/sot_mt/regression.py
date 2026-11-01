"""Metric regression: do the imported MADLAD texts reproduce the recorded metrics?

Targets, in order of strength:
  1. per item: results/madlad_translation/matched_item_metrics.csv (197 rows), recomputed with the
     worker's own metric code (legacy_madlad.worker_metrics);
  2. medians: the README tables (3 decimals for the lexical/style metrics);
  3. E5 cosine and BERTScore F1 (README only): PENDING, since no code, per-item values or model weights
     are available and nothing is downloaded.
The pilot (paraphrase baseline) definitions are computed on the same pairs for information only. They
are different metrics with the same names, never a regression target.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import paths

METRICS = ["bleu", "rougeL", "chrf", "word_retention", "function_word_cosine", "punctuation_f1", "exact_match"]
TOLERANCE = 1e-12


def per_item_comparison(recomputed: pd.DataFrame, recorded: pd.DataFrame) -> pd.DataFrame:
    j = recorded.merge(recomputed, on="item_id", how="outer", suffixes=("_recorded", "_recomputed"), indicator=True)
    out = j[["item_id", "_merge"]].rename(columns={"_merge": "presence"})
    for m in METRICS + ["words"]:
        a, b = j[f"{m}_recorded"].astype(float), j[f"{m}_recomputed"].astype(float)
        out[f"{m}_recorded"], out[f"{m}_recomputed"] = a, b
        out[f"{m}_abs_diff"] = (a - b).abs()
    return out


def summary(per_item: pd.DataFrame, readme: dict) -> pd.DataFrame:
    both = per_item[per_item.presence == "both"]
    rows = []
    for m in METRICS:
        d = both[f"{m}_abs_diff"]
        rec, new = both[f"{m}_recorded"], both[f"{m}_recomputed"]
        r = readme.get(m)
        med_new = float(new.median())
        rows.append({
            "metric_id": f"madlad_v1.{m}", "target": "matched_item_metrics.csv per item",
            "n_recorded": int(per_item[f"{m}_recorded"].notna().sum()), "n_compared": len(both),
            "n_exact": int((d == 0).sum()), "max_abs_diff": float(d.max()),
            "recorded_median": float(rec.median()), "recomputed_median": med_new,
            "readme_median": r, "readme_match_3dp": None if r is None else bool(round(med_new, 3) == r),
            "status": "pass" if len(both) and d.max() <= TOLERANCE and per_item.presence.eq("both").all()
                      and (r is None or round(med_new, 3) == r) else "fail",
        })
    for m, note in [("e5_cosine_multilingual", "E5 cosine similarity (not the Qwen e5 distance)"),
                    ("bertscore_f1", "BERTScore F1 (configuration unknown)")]:
        rows.append({"metric_id": f"madlad_readme_only.{m}", "target": "README median",
                     "n_recorded": 0, "n_compared": 0, "n_exact": None, "max_abs_diff": None,
                     "recorded_median": None, "recomputed_median": None, "readme_median": readme.get(m),
                     "readme_match_3dp": None,
                     "status": f"pending: {note}; no code or per-item values in the repo, model weights not downloaded"})
    return pd.DataFrame(rows)


def pilot_reference(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pilot (baseline) lexical definitions on the same T0/T1 pairs, imported read-only from the team's
    src/pilot. Informational: shows how the same names give different numbers."""
    src = str(paths.REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pilot.metrics import lexical_scores
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = frozenset(ENGLISH_STOP_WORDS)
    rows = [lexical_scores(r.S, r.B_F, stop) for r in pairs.itertuples(index=False)]
    return pd.DataFrame(rows).assign(item_id=pairs.item_id.values)


def pilot_summary(pilot_items: pd.DataFrame) -> pd.DataFrame:
    p = pilot_items.drop(columns="item_id")
    return pd.DataFrame({"metric_id": [f"pilot.{c}" for c in p.columns], "median": p.median().values,
                         "mean": p.mean().values, "n": len(p),
                         "note": "pilot (baseline) definition; not a regression target for madlad_v1"})


def compare_name_clashes(per_item: pd.DataFrame, pilot_items: pd.DataFrame) -> pd.DataFrame:
    """Per-item agreement between the two definitions sharing a short name."""
    j = per_item[per_item.presence == "both"].merge(pilot_items, on="item_id")
    rows = []
    for short in ["bleu", "rougeL", "word_retention"]:
        a, b = j[f"{short}_recomputed"], j[short]
        d = (a - b).abs()
        rows.append({"short_name": short, "n": len(j), "madlad_v1_median": float(np.median(a)),
                     "pilot_median": float(np.median(b)), "items_differing_gt_1e-9": int((d > 1e-9).sum()),
                     "max_abs_diff": float(d.max())})
    return pd.DataFrame(rows)
