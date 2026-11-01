"""The immutable T0 manifest: the exact 200 XSum texts of the MADLAD pilot, one row per text.

`madlad_round_robin` replicates the selection in the MADLAD worker script
(experiments/madlad_translation/madlad_explorer.ipynb, cell 5) step for step. It never resamples,
rebalances or replaces records: if the result disagrees with the legacy chain CSV, building fails.
"""
from __future__ import annotations

import pandas as pd

from .files import sha256_file, sha256_text

MANIFEST_COLUMNS = [
    "experiment_id", "sample_id", "sample_index", "dataset", "item_id", "article_id", "group_id",
    "source_label", "is_human", "t0_text", "t0_sha256", "t0_words", "t0_chars",
    "shared_prefix_words_with_human", "corpus_repo", "corpus_commit", "source_file", "source_file_sha256",
    "source_row_index", "selection_procedure",
]


def make_item_id(dataset: str, source: str, key: str) -> str:
    return f"{dataset}|{source}|{key}"


def madlad_round_robin(raw: pd.DataFrame, dataset: str, n_items, expected_originals=None) -> pd.DataFrame:
    """Worker logic: originals, drop missing text, de-duplicate (key, source), sort by (source, item_id),
    number within source, re-sort by (position, source, item_id), take the first n."""
    items = (raw.loc[raw.version_name.eq("original"), ["key", "source", "text", "source_row_index"]]
             .dropna(subset=["text"]).drop_duplicates(["key", "source"]).copy())
    items["dataset"] = dataset
    items["item_id"] = [make_item_id(dataset, s, k) for s, k in zip(items.source.astype(str), items.key.astype(str))]
    if expected_originals is not None and len(items) != expected_originals:
        raise RuntimeError(f"expected {expected_originals} originals, found {len(items)}; corpus may have changed")
    items = items.sort_values(["source", "item_id"]).reset_index(drop=True)
    items["_pos"] = items.groupby("source").cumcount()
    items = items.sort_values(["_pos", "source", "item_id"]).drop(columns="_pos").reset_index(drop=True)
    if items.item_id.duplicated().any() or items.text.astype(str).str.strip().eq("").any():
        raise ValueError("invalid source IDs or blank text")
    return items.iloc[:n_items].copy() if n_items is not None else items


def shared_prefix_words(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def build_manifest(raw: pd.DataFrame, cfg: dict, source_file_path, corpus_commit: str) -> pd.DataFrame:
    s = cfg["sample"]
    items = madlad_round_robin(raw, s["dataset"], s["n_items"], s.get("expected_originals"))
    # Observation only (nothing is removed): how many leading words each text shares with the Human
    # original of the same article in the full corpus. NaN for Human texts and articles with no Human original.
    human = (raw[(raw.version_name == "original") & (raw.source == "Human")].dropna(subset=["text"])
             .drop_duplicates("key").set_index("key").text)
    prefix = [shared_prefix_words(t, human[k]) if src != "Human" and k in human.index else None
              for t, k, src in zip(items.text, items.key, items.source)]
    m = pd.DataFrame({
        "experiment_id": cfg["experiment_id"],
        "sample_id": s["sample_id"],
        "sample_index": range(len(items)),
        "dataset": items.dataset,
        "item_id": items.item_id,
        "article_id": items.key,
        "group_id": items.dataset + "|" + items.key,
        "source_label": items.source,
        "is_human": items.source.eq("Human"),
        "t0_text": items.text,
        "t0_sha256": [sha256_text(t) for t in items.text],
        "t0_words": items.text.str.split().str.len(),
        "t0_chars": items.text.str.len(),
        "shared_prefix_words_with_human": pd.array(prefix, dtype="Int64"),
        "corpus_repo": cfg["corpus"]["repo"],
        "corpus_commit": corpus_commit,
        "source_file": f"paraphrased_datasets/{s['dataset']}_paraphrased.csv",
        "source_file_sha256": sha256_file(source_file_path),
        "source_row_index": items.source_row_index,
        "selection_procedure": s["procedure"],
    })
    return m[MANIFEST_COLUMNS].reset_index(drop=True)


def check_against_legacy(manifest: pd.DataFrame, legacy: pd.DataFrame) -> None:
    """Fail unless the manifest has the legacy chain CSV's item_ids in the same order with identical T0 text."""
    if manifest.item_id.tolist() != legacy.item_id.tolist():
        raise AssertionError("manifest item_ids/order differ from the legacy MADLAD chain CSV")
    if manifest.t0_text.tolist() != legacy.S.tolist():
        raise AssertionError("manifest T0 text differs from legacy column S")
    for col, leg in [("source_label", "source"), ("article_id", "key"), ("dataset", "domain")]:
        if manifest[col].tolist() != legacy[leg].tolist():
            raise AssertionError(f"manifest {col} differs from legacy {leg}")


def profile(manifest: pd.DataFrame) -> dict:
    """Small inspection tables for the report."""
    m = manifest
    per_article = m.groupby("article_id").size()
    prefix = m.shared_prefix_words_with_human.dropna().astype(int)
    return {
        "counts": pd.DataFrame([{
            "texts": len(m), "distinct_item_ids": m.item_id.nunique(), "distinct_articles": m.article_id.nunique(),
            "distinct_groups": m.group_id.nunique(), "sources": m.source_label.nunique(),
            "duplicate_t0_texts": int(m.t0_text.duplicated().sum()),
            "articles_with_7_sources": int((per_article == 7).sum()),
            "articles_with_lt_7_sources": int((per_article < 7).sum()),
        }]),
        "sources": m.groupby("source_label").agg(texts=("item_id", "size"), articles=("article_id", "nunique"),
                                                 median_words=("t0_words", "median"), min_words=("t0_words", "min"),
                                                 max_words=("t0_words", "max")).reset_index(),
        "length": m.t0_words.describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_frame("t0_words").T,
        "texts_per_article": per_article.value_counts().sort_index().rename_axis("texts_per_article")
                            .reset_index(name="articles"),
        "shared_prefix": pd.DataFrame([{
            "llm_texts_with_human_in_corpus": len(prefix),
            "median_shared_prefix_words": float(prefix.median()) if len(prefix) else None,
            "share_with_ge_10_words": round(float((prefix >= 10).mean()), 3) if len(prefix) else None,
            "share_with_exactly_30_words": round(float((prefix == 30).mean()), 3) if len(prefix) else None,
            "llm_texts_lt_40_words_beyond_prefix": int(((m.t0_words - m.shared_prefix_words_with_human.fillna(0))
                                                       .lt(40) & ~m.is_human).sum()),
        }]),
    }
