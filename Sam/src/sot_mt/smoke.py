"""Stage 3B smoke test helpers: choose the test items, store their IDs, and check MADLAD tokenizer parity.

Item selection (deterministic). Candidates are items whose S1 and T1 records in the pathway are both valid
and whose S2 does not exist yet, so each one continues T1 -> S2 -> T2 -> S3 -> T3. For each target
quantile of the T1 word count, in order, take the nearest candidate (ties broken by item_id), using
each source label and each article at most once. Quantiles 0.05/0.3/0.5/0.75/0.95 give short, medium
and long texts.

Tokenizer parity (MADLAD only, no GPU). Round 1 stored how many segments it cut each text into
(legacy_fields chunks_en for T0 -> S1, chunks_es for S1 -> T1). Re-segmenting the same texts with the
tokenizer of this session must give the same counts. A mismatch means the tokenizer differs from round
1, and rounds 2-3 would not use the same input handling. Equal counts are necessary, not sufficient:
two tokenizers can differ in ways that never move a segment boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import segmentation

SMOKE_QUANTILES = (0.05, 0.3, 0.5, 0.75, 0.95)
ID_COLUMNS = ["item_id", "article_id", "source_label", "t1_words", "t1_quantile_target", "length_band"]


def _pathway(table, experiment_id, translator, prompt_variant):
    return table[(table.experiment_id == experiment_id) & (table.translator == translator) &
                 (table.prompt_variant == prompt_variant)]


def candidates(table: pd.DataFrame, experiment_id: str, translator: str, prompt_variant: str) -> pd.DataFrame:
    """Items with valid S1 and T1 and no S2 yet, with their T1 word counts."""
    p = _pathway(table, experiment_id, translator, prompt_variant)
    st = p.pivot_table(index="item_id", columns="output_stage", values="validation_status", aggfunc="first")
    ok = st.index[(st.get("S1") == "valid") & (st.get("T1") == "valid") &
                  (st["S2"].isna() if "S2" in st else True)]
    t1 = p[(p.output_stage == "T1") & p.item_id.isin(ok)]
    return (t1.assign(t1_words=t1.output_text.str.split().str.len())
            [["item_id", "article_id", "source_label", "t1_words"]].sort_values("item_id").reset_index(drop=True))


def select_smoke_items(table: pd.DataFrame, experiment_id: str, translator: str = "madlad-3b",
                       prompt_variant: str = "mt-default", quantiles=SMOKE_QUANTILES) -> pd.DataFrame:
    c = candidates(table, experiment_id, translator, prompt_variant)
    if len(c) < len(quantiles):
        raise ValueError(f"only {len(c)} candidate items")
    used_items, used_src, used_art, rows = set(), set(), set(), []
    for q in quantiles:
        target = float(c.t1_words.quantile(q))
        pool = c[~c.item_id.isin(used_items) & ~c.source_label.isin(used_src) & ~c.article_id.isin(used_art)]
        if not len(pool):
            raise ValueError("ran out of distinct sources/articles")
        pick = pool.assign(_d=(pool.t1_words - target).abs()).sort_values(["_d", "item_id"]).iloc[0]
        used_items.add(pick.item_id), used_src.add(pick.source_label), used_art.add(pick.article_id)
        band = "short" if q < 0.25 else "long" if q > 0.7 else "medium"
        rows.append({**pick[["item_id", "article_id", "source_label"]].to_dict(), "t1_words": int(pick.t1_words),
                     "t1_quantile_target": q, "length_band": band})
    return pd.DataFrame(rows, columns=ID_COLUMNS)


def read_item_ids(path) -> list:
    ids = pd.read_csv(path).item_id.astype(str).tolist()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: empty or duplicated item_id list")
    return ids


def madlad_segment_parity(table: pd.DataFrame, count_tokens, budget: int, experiment_id: str,
                          translator: str = "madlad-3b", prompt_variant: str = "mt-default",
                          item_ids=None) -> pd.DataFrame:
    """One row per (item, direction): round-1 segment count vs this session's. count_tokens(text, tgt)."""
    p = _pathway(table, experiment_id, translator, prompt_variant)
    s1 = p[(p.output_stage == "S1") & p.legacy_fields.notna()]
    if item_ids is not None:
        s1 = s1[s1.item_id.isin(item_ids)]
    rows = []
    for r in s1.itertuples(index=False):
        legacy = json.loads(r.legacy_fields)
        for direction, text, tgt, key in [("en->es", r.input_text, "es", "chunks_en"),
                                          ("es->en", r.output_text, "en", "chunks_es")]:
            if not text or key not in legacy:
                continue
            try:
                n = len(segmentation.madlad_pack(text, lambda s: count_tokens(s, tgt), budget))
            except segmentation.SegmentationError as e:
                n = f"error:{e}"
            rows.append({"item_id": r.item_id, "direction": direction, "round1_segments": int(legacy[key]),
                         "session_segments": n, "match": n == int(legacy[key])})
    return pd.DataFrame(rows, columns=["item_id", "direction", "round1_segments", "session_segments", "match"])


def write_ids(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def word_similarity(a, b) -> float:
    """difflib ratio over word sequences: 1.0 identical, near 0 for unrelated or degenerate text."""
    import difflib
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return round(difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio(), 4)


def probe_verdict(t0_text: str, round1_s1: str, probe_text, finish: str, validation_cfg: dict,
                  min_similarity: float = 0.5) -> dict:
    """Judge one T0 -> S1 re-translation against round 1. ok = not degenerate (length cap, blocking check)
    and at least `min_similarity` word similarity to the stored S1. Exact identity is reported, not required:
    small library/GPU drift can change wording without breaking the translation."""
    from .checks import check_step
    blocking = ["engine:generation_cap"] if finish == "length" else []
    if probe_text is None:
        blocking.append("engine:no_output")
    else:
        blocking += check_step(t0_text, probe_text, "es", validation_cfg)["blocking"]
    sim = word_similarity(round1_s1, probe_text)
    return {"identical": probe_text == round1_s1, "similarity": sim, "blocking": ";".join(blocking) or None,
            "ok": not blocking and sim >= min_similarity}


# -- real-model preflights of the other translators (NLLB, Qwen, Llama): the same documents for every model ----
PREFLIGHT_QUANTILES = (0.1, 0.5, 1.0)       # short, median and the longest T0 (worst case for memory and tokens)


def select_preflight_items(manifest: pd.DataFrame, quantiles=PREFLIGHT_QUANTILES) -> pd.DataFrame:
    """Original T0 documents by T0 word quantile (nearest, ties by item_id), each source label and article once.
    Chosen from T0 only, never from another model's outputs."""
    m = manifest.sort_values("item_id")
    used_src, used_art, rows = set(), set(), []
    for q in quantiles:
        target = float(m.t0_words.quantile(q))
        pool = m[~m.source_label.isin(used_src) & ~m.article_id.isin(used_art)]
        pick = pool.assign(_d=(pool.t0_words - target).abs()).sort_values(["_d", "item_id"]).iloc[0]
        used_src.add(pick.source_label), used_art.add(pick.article_id)
        rows.append({"item_id": pick.item_id, "dataset": pick.dataset, "article_id": pick.article_id,
                     "group_id": pick.group_id, "source_label": pick.source_label, "t0_words": int(pick.t0_words),
                     "t0_quantile_target": q, "length_band": "short" if q < 0.25 else "long" if q > 0.7 else "medium"})
    return pd.DataFrame(rows)
