"""Import the MADLAD round-1 pilot (results/madlad_translation/chain_es_madlad4003b.csv) into the
generation schema, and recompute its metrics with the worker's own code for regression testing.

Legacy flags (`bad`, `bad_reason`) were assigned per ROUND TRIP: `generation_cap` means some chunk in
either direction ended without EOS, and the length-ratio flags compare T1 with T0. The direction that
failed is not recorded, so a flagged item's S1 and T1 rows are both marked invalid, with
"legacy:direction_unattributed". This is why the import runs with allow_invalid_predecessor=True: the
legacy T1 rows were generated from S1 rows that are invalid under this stricter rule. No new
generation may do the same.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from .checks import BLOCKING, check_step, stage2_rules
from .files import sha256_text
from .schema import GENERATION_COLUMNS, record_id, stage_info

LEGACY_COLUMNS = ["key", "source", "S", "domain", "item_id", "words", "F", "B_F", "bad", "bad_reason",
                  "chunks_en", "chunks_es", "seconds"]
TRANSLATOR, PROMPT_VARIANT = "madlad-3b", "mt-default"


def read_legacy_chain(path: Path) -> pd.DataFrame:
    """Read exactly as the worker did (keep_default_na=False keeps literal text such as "NA")."""
    d = pd.read_csv(path, keep_default_na=False)
    missing = set(LEGACY_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError(f"legacy chain CSV lacks columns {sorted(missing)}")
    if d.item_id.duplicated().any() or d.item_id.eq("").any():
        raise ValueError("legacy chain CSV has duplicate or blank item_ids")
    if not (d.item_id == d.domain + "|" + d.source + "|" + d.key).all():
        raise ValueError("legacy item_id is not domain|source|key")
    d["bad"] = d.bad.astype(str).str.lower().eq("true")
    return d


def generation_config(translators_cfg: dict) -> tuple:
    from .translators import generation_config as _gc
    return _gc(translators_cfg, TRANSLATOR, PROMPT_VARIANT)


def import_rows(legacy: pd.DataFrame, manifest: pd.DataFrame, cfg: dict, translators_cfg: dict,
                legacy_rel_path: str, legacy_sha256: str) -> pd.DataFrame:
    """Two generation rows (S1, T1) per legacy item. T0 comes from the manifest and must equal legacy S."""
    m = manifest.set_index("item_id")
    if set(legacy.item_id) != set(m.index):
        raise ValueError("legacy item_ids differ from the manifest")
    gcid, _ = generation_config(translators_cfg)
    t = translators_cfg["translators"][TRANSLATOR]
    rows = []
    for i, r in legacy.iterrows():
        t0 = m.loc[r.item_id]
        if t0.t0_text != r.S:
            raise ValueError(f"{r.item_id}: legacy S differs from the manifest T0 text")
        legacy_reasons = [x for x in r.bad_reason.split(";") if x] if r.bad else []
        if r.bad and not legacy_reasons:
            legacy_reasons = ["unspecified"]
        common = {"bad": bool(r.bad), "bad_reason": r.bad_reason, "chunks_en": int(r.chunks_en),
                  "chunks_es": int(r.chunks_es), "seconds_round_trip": float(r.seconds), "words": int(r.words),
                  "source_file": legacy_rel_path, "source_file_sha256": legacy_sha256, "source_row_index": int(i)}
        for stage, inp, out, col in [("S1", r.S, r.F, "F"), ("T1", r.F, r.B_F, "B_F")]:
            info = stage_info(stage)
            chk = check_step(inp, out, info["output_language"], stage2_rules(cfg["validation"]))
            reasons = [f"legacy:{x}" for x in legacy_reasons]
            if reasons:
                reasons.append("legacy:direction_unattributed")
            reasons += chk["blocking"]
            row = {
                "experiment_id": cfg["experiment_id"], "dataset": t0.dataset, "item_id": r.item_id,
                "article_id": t0.article_id, "source_label": t0.source_label,
                "translator": TRANSLATOR, "prompt_variant": PROMPT_VARIANT,
                "model_checkpoint": t["model_checkpoint"], "model_revision": t["model_revision"],
                **info, "output_stage": stage,
                "input_text": inp, "output_text": out,
                "input_sha256": sha256_text(inp), "output_sha256": sha256_text(out),
                "validation_status": "invalid" if reasons else "valid",
                "validation_reason": ";".join(reasons) or None,
                "validation_warnings": ";".join(chk["warnings"]) or None,
                "generation_config_id": gcid, "run_id": cfg["legacy"]["madlad_run_id"],
                "provenance": f"legacy:{legacy_rel_path}#row={i};column={col}",
                "legacy_fields": json.dumps({**common, "column": col}, sort_keys=True),
                "generation_meta": None,
            }
            row["record_id"] = record_id(row)
            rows.append(row)
    return pd.DataFrame(rows)[GENERATION_COLUMNS]


def blocking_codes(reason) -> list:
    return [x for x in str(reason or "").split(";") if x and x.split(":", 1)[-1] in BLOCKING]


# -- worker metric code, reused verbatim ---------------------------------------------------------

WORKER_METRIC_NAMES = {"func", "words", "word_retention", "function_cos", "punctuation_f1", "bleu", "chrf", "rouge"}


def worker_source(notebook: Path) -> str:
    """The worker script the MADLAD notebook writes with WORKER.write_text('...')."""
    nb = json.loads(Path(notebook).read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or "WORKER.write_text" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text" and node.args and isinstance(node.args[0], ast.Constant)):
                return node.args[0].value
    raise ValueError("worker script not found in the MADLAD notebook")


def worker_metrics(notebook: Path) -> dict:
    """Execute only the worker's metric definitions (imports + func/words/... + bleu/chrf/rouge objects)."""
    tree = ast.parse(worker_source(notebook))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {a.name for a in node.names}
            if names & {"re", "Counter", "BLEU", "CHRF", "rouge_scorer"}:
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in WORKER_METRIC_NAMES:
            keep.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in WORKER_METRIC_NAMES):
            keep.append(node)
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), f"{notebook}::worker_metrics", "exec"), ns)
    missing = WORKER_METRIC_NAMES - set(ns)
    if missing:
        raise ValueError(f"worker metric definitions not found: {sorted(missing)}")
    return ns


def recompute_legacy_metrics(pairs: pd.DataFrame, notebook: Path) -> pd.DataFrame:
    """Same row construction as the worker's metric loop; `pairs` has item_id, source, words, seconds, S, B_F."""
    w = worker_metrics(notebook)
    rows = []
    for r in pairs.itertuples(index=False):
        s, h = str(r.S), str(r.B_F)
        rows.append(dict(item_id=r.item_id, source=r.source, words=r.words, seconds=r.seconds,
                         bleu=w["bleu"].sentence_score(h, [s]).score / 100,
                         rougeL=w["rouge"].score(s, h)["rougeL"].fmeasure,
                         chrf=w["chrf"].sentence_score(h, [s]).score / 100,
                         word_retention=w["word_retention"](s, h),
                         function_word_cosine=w["function_cos"](s, h),
                         punctuation_f1=w["punctuation_f1"](s, h),
                         exact_match=float(s.strip() == h.strip())))
    return pd.DataFrame(rows)
