"""Validation counts for generation tables."""
from __future__ import annotations

import pandas as pd

from .schema import PATHWAY


def _has(series: pd.Series, code: str) -> pd.Series:
    return series.fillna("").str.split(";").apply(lambda xs: any(x == code or x.startswith(code + "=") for x in xs))


def validation_summary(gen: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in gen.groupby(PATHWAY + ["output_stage"], sort=False):
        reason, warn = g.validation_reason, g.validation_warnings
        rows.append({
            **dict(zip(PATHWAY + ["output_stage"], k)),
            "records": len(g),
            "manifest_items_without_record": len(set(manifest.item_id) - set(g.item_id)),
            "records_not_in_manifest": len(set(g.item_id) - set(manifest.item_id)),
            "missing_ids": int(g[["item_id", "article_id"]].isna().any(axis=1).sum()),
            "valid": int((g.validation_status == "valid").sum()),
            "invalid": int((g.validation_status == "invalid").sum()),
            "legacy_flagged": int(reason.fillna("").str.contains("legacy:").sum()),
            "legacy_generation_cap": int(_has(reason, "legacy:generation_cap").sum()),
            "legacy_severe_expansion": int(_has(reason, "legacy:severe_expansion").sum()),
            "legacy_severe_compression": int(_has(reason, "legacy:severe_compression").sum()),
            "missing_output": int(_has(reason, "check:missing_output").sum()),
            "blank": int(_has(reason, "check:blank").sum()),
            "unchanged": int(_has(reason, "check:unchanged").sum()),
            "wrong_language_untranslated": int(_has(reason, "check:wrong_language").sum()),
            "repetition_loop": int(_has(reason, "check:repetition").sum()),
            "warn_possible_truncation": int(_has(warn, "check:possible_truncation").sum()),
            "warn_length_ratio": int(_has(warn, "check:length_ratio").sum()),
        })
    return pd.DataFrame(rows)


def item_flags(gen: pd.DataFrame) -> pd.DataFrame:
    """Text-free per-record flag listing for every record that is invalid or has a warning."""
    g = gen[(gen.validation_status == "invalid") | gen.validation_warnings.notna()]
    return g[PATHWAY + ["item_id", "output_stage", "validation_status", "validation_reason",
                        "validation_warnings"]].reset_index(drop=True)
