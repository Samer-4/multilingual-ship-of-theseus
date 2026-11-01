"""Optional import of the legacy Qwen round 1 (chain_es_Qwen3-4B-Instruct-2507.csv from the Colab run).

The file lives only on a teammate's Google Drive and may never arrive, so nothing depends on it.
Imported records go to the pathway `qwen3-4b-vllm-legacy/<free|literal>`, kept apart from `qwen3-4b`:
they were produced with vLLM, and the one-config-per-pathway rule forbids continuing a vLLM chain with
transformers. Whether to use them (as a reference, or as round 1 with a vLLM continuation) is a decision
for the team, not something this importer assumes.

Legacy per-condition flags look like "F:truncated B_F:spanish" (space-separated, prefixed by column).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .checks import check_step, stage2_rules
from .files import sha256_file, sha256_text
from .schema import GENERATION_COLUMNS, record_id, stage_info
from .translators import generation_config

TRANSLATOR = "qwen3-4b-vllm-legacy"
COLUMNS = {"free": ("F", "B_F"), "literal": ("L", "B_L")}
RUN_ID = "legacy-qwen-colab-r1"


class ArtifactNotFoundError(FileNotFoundError):
    pass


def _text(v):
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


def import_qwen_chain(path, manifest: pd.DataFrame, cfg: dict, tcfg: dict, prompt_variant: str = "free"):
    """Return (rows, coverage). Raises ArtifactNotFoundError if the file is absent."""
    path = Path(path) if path else None
    if path is None or not path.exists():
        raise ArtifactNotFoundError(f"Qwen legacy chain file not found: {path}. Run qwen3-4b from T0 instead.")
    fwd, back = COLUMNS[prompt_variant]
    d = pd.read_csv(path, keep_default_na=False, na_values=[""])
    need = {"item_id", "S", fwd, back}
    if need - set(d.columns):
        raise ValueError(f"{path.name} lacks columns {sorted(need - set(d.columns))}")
    if d.item_id.duplicated().any():
        raise ValueError(f"{path.name} has duplicate item_ids")
    m = manifest.set_index("item_id")
    d = d[d.item_id.isin(m.index)]
    coverage = {"manifest_items": len(m), "found": len(d), "missing": sorted(set(m.index) - set(d.item_id))}
    gcid, _ = generation_config(tcfg, TRANSLATOR, prompt_variant)
    entry = tcfg["translators"][TRANSLATOR]
    sha = sha256_file(path)
    rows = []
    for i, r in d.iterrows():
        t0 = m.loc[r.item_id]
        if t0.t0_text != r.S:
            raise ValueError(f"{r.item_id}: legacy S differs from the manifest T0 text")
        legacy = str(r.get("bad_reason", "") or "").split()
        s1_text, t1_text = _text(r[fwd]), _text(r[back])
        prev_status, prev_out = "valid", t0.t0_text
        for stage, col, out in [("S1", fwd, s1_text), ("T1", back, t1_text)]:
            if stage == "T1" and s1_text is None:
                break                                   # back translation never ran: no T1 record
            info = stage_info(stage)
            reasons = [f"legacy:{x.split(':', 1)[1]}" for x in legacy if x.startswith(col + ":")]
            if _text(r.get(f"finish_{col}")) == "length" and "legacy:truncated" not in reasons:
                reasons.append("legacy:truncated")
            chk = check_step(prev_out, out, info["output_language"], stage2_rules(cfg["validation"]))
            reasons += chk["blocking"]
            if prev_status != "valid":
                reasons.append("legacy:upstream_invalid")
            row = {"experiment_id": cfg["experiment_id"], "dataset": t0.dataset, "item_id": r.item_id,
                   "article_id": t0.article_id, "source_label": t0.source_label, "translator": TRANSLATOR,
                   "prompt_variant": prompt_variant, "model_checkpoint": entry["model_checkpoint"],
                   "model_revision": entry.get("model_revision"), **info, "output_stage": stage,
                   "input_text": prev_out, "output_text": out, "input_sha256": sha256_text(prev_out),
                   "output_sha256": None if out is None else sha256_text(out),
                   "validation_status": "invalid" if reasons else "valid",
                   "validation_reason": ";".join(reasons) or None,
                   "validation_warnings": ";".join(chk["warnings"]) or None,
                   "generation_config_id": gcid, "run_id": RUN_ID,
                   "provenance": f"legacy:{path.name}#row={i};column={col}",
                   "legacy_fields": json.dumps({"bad_reason": " ".join(legacy), "finish": _text(r.get(f"finish_{col}")),
                                                "source_file_sha256": sha}, sort_keys=True),
                   "generation_meta": None}
            row["record_id"] = record_id(row)
            rows.append(row)
            prev_status, prev_out = row["validation_status"], out
    return pd.DataFrame(rows, columns=GENERATION_COLUMNS), coverage
