"""Consolidate the finished MADLAD XSum pilot into one final, verified folder (never overwrites).

Source: the remainder export (data/generated/madlad_pilot/remainder/xsum/madlad-3b), which already holds Stage 2
+ smoke + batch 1 + remainder. It is checked with verify_madlad_pilot.verify (hashes, Stage 2 unchanged, lineage,
unique ids, earlier exports byte-identical); invalid records are KEPT (attrition), never repaired or dropped.

Writes data/generated/madlad_final/xsum/madlad-3b/:
  generations_merged.parquet   byte copy of the verified source
  chains.csv                   one row per item: ids, source label, status per stage, complete (S1-T3 all valid)
  FINAL.json                   sha256s, sources, counts, attrition, warning counts, verification result
and data/generated/madlad_final/audit/invalid_<stage>_<item>.json for every invalid new record (full text + meta).
Other translators (--translator nllb | qwen3-4b | phi4-mini): the source is the model's LATEST downloaded full-run
attempt (data/generated/<model>/full[_rK]/...; its merged table holds the preflight and every earlier attempt),
checked to contain the preflight and every earlier attempt byte-identical; all six stages are that model's own;
written to data/generated/<model>_final/.
Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/finalize_madlad.py
  Sam/.venv/bin/python Sam/scripts/finalize_madlad.py --translator qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sot_mt import paths, runner, schema, session  # noqa: E402
from sot_mt.files import sha256_file  # noqa: E402
import verify_madlad_pilot as vp  # noqa: E402

SOURCE = "data/generated/madlad_pilot/remainder/xsum/madlad-3b"
CONTAINS = ["data/generated/madlad_pilot/b01/xsum/madlad-3b", "data/generated/smoke5_v2/xsum/madlad-3b"]
OUT = "data/generated/madlad_final"
STAGES = ["S1", "T1", "S2", "T2", "S3", "T3"]
WARNINGS = ["sentence_drop", "number_change", "roundtrip_length_loss", "entity_loss"]


def loop_summary(text: str, n: int = 8) -> dict:
    """The most repeated word n-gram: how a decoding loop shows up in the text (audit only)."""
    w = re.findall(r"\w+", text.lower())
    grams = Counter(zip(*[w[i:] for i in range(n)]))
    g, c = grams.most_common(1)[0] if grams else ((), 0)
    return {"ngram": " ".join(g), "n": n, "count": c, "words": len(w)}


def model_sources(model: str, max_attempts: int = 20) -> tuple:
    """(latest full attempt export, [preflight + earlier attempts], final folder) for a non-MADLAD model."""
    import model_runs as mr
    done = [k for k in range(1, max_attempts + 1)
            if (paths.SAM_ROOT / mr.export_dir(model, "full", k) / runner.RESUME).exists()]
    if not done:
        raise SystemExit(f"{model}: no downloaded full run")
    earlier = [mr.export_dir(model, "preflight")] + [mr.export_dir(model, "full", k) for k in done[:-1]]
    return mr.export_dir(model, "full", done[-1]), earlier, f"data/generated/{mr.MODELS[model][0]}_final"


def finalize(source=SOURCE, contains=CONTAINS, out=OUT, translator="madlad-3b") -> dict:
    final = paths.SAM_ROOT / out
    if final.exists():
        raise SystemExit(f"{final} already exists; the final dataset is never overwritten")
    out = final.with_name(final.name + ".partial")       # built here, renamed only when complete
    if out.exists():
        shutil.rmtree(out)
    gen = out / "xsum" / translator
    v = vp.verify(source, contains, translator)       # raises on any integrity problem
    src = paths.SAM_ROOT / source
    t = pd.read_parquet(src / runner.MERGED)
    pv = "mt-default" if translator in {"madlad-3b", "nllb"} else "free"
    pw = t[(t.translator == translator) & (t.prompt_variant == pv)]
    new_stages = vp.NEW_STAGES if translator == "madlad-3b" else vp.ALL_STAGES
    st = pw.pivot(index="item_id", columns="output_stage", values="validation_status").reindex(columns=STAGES)
    manifest = session.load_inputs()[0]
    ids = manifest.set_index("item_id")[["dataset", "article_id", "group_id", "source_label"]]
    chains = ids.join(st).reset_index()
    chains["complete"] = (chains[STAGES] == "valid").all(axis=1)
    chains = chains.sort_values("item_id")
    new = pw[pw.output_stage.isin(new_stages)]
    w = new.validation_warnings.fillna("")
    gen.mkdir(parents=True)
    shutil.copy2(src / runner.MERGED, gen / runner.MERGED)
    chains.to_csv(gen / "chains.csv", index=False)
    audit = out / "audit"
    audit.mkdir(exist_ok=True)
    audits = []
    for r in new[new.validation_status != "valid"].itertuples(index=False):
        name = f"invalid_{r.output_stage}_{r.item_id.replace('|', '_')}.json"
        rec = {"record_id": r.record_id, "item_id": r.item_id, "output_stage": r.output_stage,
               "validation_reason": r.validation_reason, "validation_warnings": r.validation_warnings,
               "run_id": r.run_id, "input_text": r.input_text, "output_text": r.output_text,
               "generation_meta": json.loads(r.generation_meta), "loop": loop_summary(r.output_text),
               "note": "kept in generations_merged.parquet as an invalid record; never fed to a later stage"}
        (audit / name).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        audits.append(name)
    by_label = chains.groupby("source_label").agg(items=("item_id", "size"), complete=("complete", "sum"))
    info = {"finalized_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_export": source, "source_resume_sha256": sha256_file(src / runner.RESUME),
            "generations_merged_sha256": sha256_file(gen / runner.MERGED),
            "chains_csv_sha256": sha256_file(gen / "chains.csv"),
            "records": len(t), "stage2_records": v["stage2_records"], "new_records": v["new_records"],
            "items": len(chains), "eligible_items": v["eligible_items"], "complete_items": int(chains.complete.sum()),
            "new_by_stage": v["new_by_stage"], "invalid_new": v["invalid_new"], "audit_files": audits,
            "attrition": schema.attrition(pw, manifest).drop(columns=["experiment_id"],
                                                                              errors="ignore").to_dict("records"),
            "warnings_by_stage": {c: {s: int(w[new.output_stage == s].str.contains("check:" + c).sum())
                                      for s in new_stages} for c in WARNINGS},
            "records_with_any_warning": int((w != "").sum()),
            "source_labels": {k: {"items": int(r["items"]), "complete": int(r["complete"])} for k, r in by_label.iterrows()},
            "earlier_exports_kept": v["earlier_exports_kept"],
            "verification": "scripts/verify_madlad_pilot.py: integrity OK"}
    info["translator"], info["prompt_variant"] = translator, pv
    (gen / "FINAL.json").write_text(json.dumps(info, indent=1, default=int) + "\n")
    out.rename(final)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--translator", default="madlad-3b", choices=["madlad-3b", "nllb", "qwen3-4b", "phi4-mini"])
    a = ap.parse_args()
    info = finalize() if a.translator == "madlad-3b" else finalize(*model_sources(a.translator), translator=a.translator)
    print(json.dumps({k: info[k] for k in ["records", "stage2_records", "new_records", "complete_items",
                                           "eligible_items", "invalid_new", "source_labels", "audit_files"]},
                     indent=1, default=int))


if __name__ == "__main__":
    main()
