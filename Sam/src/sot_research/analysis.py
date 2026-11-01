"""Loading and aggregation helpers shared by the three RQ notebooks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import corpus, paths, plots

# metric -> (label, family, higher_is_more_similar)
METRICS = {
    "bleu": ("BLEU", "lexical", True), "rougeL": ("ROUGE-L F", "lexical", True), "chrf": ("chrF", "lexical", True),
    "word_retention": ("Word retention", "lexical", True), "word_edit_rate": ("Word edit rate", "lexical", False),
    "length_ratio": ("Length ratio (words)", "lexical", None), "exact_equal": ("Exact equality", "lexical", True),
    "e5_cosine": ("E5 cosine", "semantic", True), "mpnet_cosine": ("MPNet cosine", "semantic", True),
    "bertscore_f1": ("BERTScore F1 (raw)", "semantic", True),
    "bertscore_f1_rescaled": ("BERTScore F1 (rescaled)", "semantic", True),
    "fw_cosine": ("Function-word cosine", "style", True), "punct_l1": ("Punctuation L1", "style", False),
    "pos_jsd": ("POS JSD", "style", False), "dep_jsd": ("Dependency JSD", "style", False),
    "char3_cosine": ("Char 3-gram cosine", "style", True), "burrows_delta": ("Burrows' Delta", "style", False),
    "sent_len_ratio": ("Sentence-length ratio", "style", None),
    "sentence_count_ratio": ("Sentence-count ratio", "style", None),
    "entity_recall": ("Entity recall", "information", True), "number_recall": ("Number recall", "information", True),
    "quantity_recall": ("Quantity recall", "information", True),
    "sent_coverage": ("Sentence coverage", "information", True),
}


def load(require_text: bool = False, pairs: bool = True) -> dict:
    """Committed text-free tables (+ the local text table when require_text). Translators are discovered."""
    cov = paths.results("data_coverage")
    d = {"pairs": pd.read_parquet(paths.results("metrics") / "pair_metrics.parquet") if pairs else None,
         "features": pd.read_parquet(paths.results("shared_features") / "text_features.parquet"),
         "chains": pd.read_csv(cov / "chains.csv"), "cohort": pd.read_csv(cov / "cohort.csv"),
         "translators_table": pd.read_csv(cov / "translators.csv"),
         "lineage": json.loads((cov / "lineage_report.json").read_text())}
    d["translators"] = corpus.translator_order(d["translators_table"].translator)
    d["names"] = {t: corpus.display_name(t) for t in d["translators"]}
    d["colors"] = plots.translator_colors(d["translators"])
    d["matched_items"] = set(d["cohort"].loc[d["cohort"].matched, "item_id"])
    if require_text:
        d["texts"] = corpus.load_texts()
    return d


def cluster_ci(df: pd.DataFrame, by: list, value: str, cluster: str = "article_id", n_boot: int = 2000,
               seed: int = 6120) -> pd.DataFrame:
    """Mean of `value` per group with a 95% percentile CI from resampling clusters (articles) with replacement."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, part in df.dropna(subset=[value]).groupby(by, observed=True):
        g = g if isinstance(g, tuple) else (g,)
        pc = part.groupby(cluster)[value].agg(["sum", "count"])
        s, c = pc["sum"].to_numpy(float), pc["count"].to_numpy(float)
        idx = rng.integers(0, len(s), size=(n_boot, len(s)))
        boots = s[idx].sum(1) / c[idx].sum(1)
        rows.append({**dict(zip(by, g)), "metric": value, "mean": s.sum() / c.sum(),
                     "median": float(part[value].median()), "ci_low": np.percentile(boots, 2.5),
                     "ci_high": np.percentile(boots, 97.5), "n_pairs": int(c.sum()), "n_articles": len(s)})
    return pd.DataFrame(rows)


def paired_differences(pairs: pd.DataFrame, value: str, translators: list, kind="cumulative", stage="T3",
                       n_boot: int = 2000, seed: int = 6120) -> pd.DataFrame:
    """For every translator pair (a, b): mean over items of value_a - value_b on the SAME items (matched cohort),
    95% article-cluster bootstrap CI. Positive = a scores higher."""
    rng = np.random.default_rng(seed)
    w = (pairs[(pairs.kind == kind) & (pairs.hyp_stage == stage) & pairs.matched]
         .pivot_table(index=["item_id", "article_id"], columns="translator", values=value).reset_index())
    rows = []
    for i, a in enumerate(translators):
        for b in translators[i + 1:]:
            x = w.dropna(subset=[a, b]).assign(diff=lambda d: d[a] - d[b])
            pc = x.groupby("article_id")["diff"].agg(["sum", "count"])
            s, c = pc["sum"].to_numpy(), pc["count"].to_numpy()
            idx = rng.integers(0, len(s), size=(n_boot, len(s)))
            boots = s[idx].sum(1) / c[idx].sum(1)
            rows.append({"metric": value, "stage": stage, "a": a, "b": b, "mean_diff": s.sum() / c.sum(),
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "a_better_share": float((x["diff"] > 0).mean()), "n_items": len(x), "n_articles": len(s)})
    return pd.DataFrame(rows)


def fmt_ci(r, digits=3) -> str:
    return f"{r['mean']:.{digits}f} [{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]"


def excerpt(text: str, n: int = 60) -> str:
    w = str(text).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")
