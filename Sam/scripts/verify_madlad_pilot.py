"""Integrity check of a downloaded MADLAD pilot export (read-only; nothing is written unless --report is given).

Checks, all hard failures except completeness:
  - RESUME.json hashes of the placed generation files, staged-data hashes, the 400 Stage 2 records unchanged;
  - duplicate / conflict / chain checks on the merged table (runner.load_tables) and unique record ids;
  - lineage of every new record: input_stage is the previous stage, input_text is that stage's output;
  - every record of each earlier export (--contains) is present byte-identical (output, run id, metadata),
    so nothing was regenerated;
  - new records use the frozen config, the MADLAD checkpoint and the verified decoder wiring / runtime patch;
    no quarantined or MOCK run.
Completeness (reported, and required with --expect-complete): 4 new records (S2, T2, S3, T3) for every item
with a valid T1, all valid -> 400 + 4 x 197 = 1,188 records.

Other translators (--translator nllb | qwen3-4b | llama): every stage S1-T3 is new; S1 must read the original
T0 of the manifest (never another model's text); config and checkpoint come from translators.yaml; complete =
all 200 documents valid in all six stages -> 400 (MADLAD Stage 2, unchanged) + 1,200 records.

Usage (from the repo root), after the remainder run is downloaded:
  Sam/.venv/bin/python Sam/scripts/verify_madlad_pilot.py data/generated/madlad_pilot/remainder/xsum/madlad-3b \\
      --contains data/generated/madlad_pilot/b01/xsum/madlad-3b data/generated/smoke5_v2/xsum/madlad-3b --expect-complete
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import paths, runner, schema, session  # noqa: E402
from sot_mt.files import load_yaml, sha256_file  # noqa: E402

NEW_STAGES = ["S2", "T2", "S3", "T3"]
ALL_STAGES = ["S1", "T1"] + NEW_STAGES
PREVIOUS = {"S1": "T0", "T1": "S1", "S2": "T1", "T2": "S2", "S3": "T2", "T3": "S3"}
KEEP = ["record_id", "output_sha256", "input_sha256", "validation_status", "run_id", "generation_meta"]


class IntegrityError(RuntimeError):
    pass


def _abs(d) -> Path:
    return Path(d) if Path(d).is_absolute() else paths.SAM_ROOT / d


def verify(export_dir, contains=(), translator="madlad-3b", prompt_variant=None) -> dict:
    from sot_mt.translators import generation_config
    prompt_variant = prompt_variant or ("free" if translator in {"qwen3-4b", "llama", "phi4-mini"} else "mt-default")
    stages = NEW_STAGES if translator == "madlad-3b" else ALL_STAGES
    exp = _abs(export_dir)
    resume = json.loads((exp / runner.RESUME).read_text())
    errs = [f"{n} does not match RESUME.json" for n, sha in resume["files"].items()      # the placed generation
            if (exp / n).exists() and sha256_file(exp / n) != sha]                        # files (reports live in
    errs += [f"{n} missing" for n in (runner.MERGED, runner.DELTA) if not (exp / n).exists()]   # results/)
    if errs:
        raise IntegrityError(errs)
    manifest, staged = session.load_inputs()                     # staged hashes verified here
    runner.load_tables([exp / runner.MERGED], manifest)          # duplicate / conflict / chain checks
    merged = pd.read_parquet(exp / runner.MERGED)
    cols = ["record_id", "output_sha256", "validation_status", "generation_config_id", "run_id"]
    s2 = staged[cols].sort_values("record_id").reset_index(drop=True)
    m2 = merged[merged.record_id.isin(s2.record_id)][cols].sort_values("record_id").reset_index(drop=True)
    if len(m2) != len(s2) or not m2.fillna("").equals(s2.fillna("")):
        errs.append("the Stage 2 records are not all present unchanged")
    if merged.record_id.duplicated().any():
        errs.append(f"{int(merged.record_id.duplicated().sum())} duplicate record ids")
    if merged.duplicated(schema.KEY).any():
        errs.append("duplicate composite keys")
    tcfg, cfg = load_yaml("translators.yaml"), load_yaml("experiment.yaml")
    pw = merged[(merged.translator == translator) & (merged.prompt_variant == prompt_variant)]
    by = pw.set_index(["item_id", "output_stage"])
    new = pw[pw.output_stage.isin(stages)]
    t0 = manifest.set_index("item_id").t0_text
    config_id = generation_config(tcfg, translator, prompt_variant)[0]
    checkpoint = tcfg["translators"][translator]["model_checkpoint"]
    patch = (tcfg.get("runtime_patches") or {}).get(translator)
    for r in new.itertuples(index=False):
        if r.output_stage == "S1":            # from the original document only
            if r.input_stage != "T0" or r.input_text != t0.get(r.item_id):
                errs.append(f"{r.item_id} S1: input is not the manifest T0")
            prev = False
        else:
            prev = by.loc[(r.item_id, PREVIOUS[r.output_stage])] if (r.item_id, PREVIOUS[r.output_stage]) in by.index else None
        if prev is False:
            pass
        elif r.input_stage != PREVIOUS[r.output_stage] or prev is None:
            errs.append(f"{r.item_id} {r.output_stage}: input stage {r.input_stage}, predecessor missing={prev is None}")
        elif prev.validation_status != "valid" or prev.output_text != r.input_text:
            errs.append(f"{r.item_id} {r.output_stage}: input is not the valid {PREVIOUS[r.output_stage]} output")
        meta = json.loads(r.generation_meta)
        if r.generation_config_id != config_id or r.model_checkpoint != checkpoint:
            errs.append(f"{r.item_id} {r.output_stage}: config/model {r.generation_config_id}/{r.model_checkpoint}")
        if patch and (meta.get("embedding_wiring") != patch["embedding_tie"] or meta.get("runtime_patch") != patch["patch_id"]):
            errs.append(f"{r.item_id} {r.output_stage}: wiring {meta.get('embedding_wiring')}/{meta.get('runtime_patch')}")
    runs = set(merged.run_id.astype(str))
    if runs & set(cfg.get("quarantined_runs") or []):
        errs.append(f"quarantined runs present: {sorted(runs & set(cfg['quarantined_runs']))}")
    if any(x.startswith(runner.MOCK_PREFIX) for x in runs):
        errs.append("MOCK records present")
    kept = {}
    for d in contains:
        old = pd.read_parquet(_abs(d) / runner.MERGED)[KEEP].set_index("record_id").sort_index()
        missing = old.index.difference(merged.record_id)
        same = merged.set_index("record_id")[KEEP[1:]].reindex(old.index)
        changed = int((same.fillna("") != old.fillna("")).any(axis=1).sum())
        kept[str(d)] = {"records": len(old), "missing": len(missing), "changed": changed}
        if len(missing) or changed:
            errs.append(f"{d}: {len(missing)} records missing, {changed} changed")
    if errs:
        raise IntegrityError(f"{len(errs)} problems, first: {errs[:5]}")
    st = pw.pivot(index="item_id", columns="output_stage", values="validation_status")
    eligible = (st.index[(st.S1 == "valid") & (st.T1 == "valid")] if translator == "madlad-3b"
                else pd.Index(manifest.item_id))
    complete = [i for i in eligible if all(st.get(s, pd.Series(dtype=object)).get(i) == "valid" for s in stages)]
    counts = new.groupby("output_stage").validation_status.value_counts().unstack(fill_value=0)
    return {"export": str(export_dir), "records": len(merged), "stage2_records": len(s2),
            "new_records": len(new), "eligible_items": len(eligible), "complete_items": len(complete),
            "expected_new": len(stages) * len(eligible), "expected_total": len(s2) + len(stages) * len(eligible),
            "new_by_stage": {s: counts.loc[s].to_dict() if s in counts.index else {} for s in stages},
            "translator": translator, "source_labels": int(pw.source_label.nunique()) if len(pw) else 0,
            "invalid_new": new[new.validation_status != "valid"][["item_id", "output_stage", "validation_reason"]]
            .to_dict("records"), "earlier_exports_kept": kept, "resume_records_total": resume["records_total"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", help="downloaded export (relative to Sam/)")
    ap.add_argument("--contains", nargs="*", default=[], help="earlier exports whose records must be kept unchanged")
    ap.add_argument("--expect-complete", action="store_true", help="fail unless every eligible item has valid S2-T3")
    ap.add_argument("--report", help="also write the result as JSON here")
    ap.add_argument("--translator", default="madlad-3b")
    a = ap.parse_args()
    r = verify(a.export_dir, a.contains, a.translator)
    print(json.dumps(r, indent=1, default=str))
    if a.report:
        Path(a.report).write_text(json.dumps(r, indent=1, default=str) + "\n")
    target = (1188, 197) if a.translator == "madlad-3b" else (1600, 200)
    ok = r["new_records"] == r["expected_new"] and (r["records"], r["eligible_items"]) == target \
        and r["complete_items"] == r["eligible_items"]
    print(f"\nintegrity OK | records {r['records']} (expected {r['expected_total']}) | new {r['new_records']}/"
          f"{r['expected_new']} | complete items {r['complete_items']}/{r['eligible_items']}")
    if a.expect_complete and not ok:
        raise SystemExit("integrity OK, but the pilot is NOT complete (see invalid_new / new_by_stage)")


if __name__ == "__main__":
    main()
