"""Download a finished Kaggle run and file it locally, keeping the raw download untouched.

  1. checks `kaggle kernels status` is COMPLETE (override with --allow-incomplete);
  2. `kaggle kernels output K -p Sam/results/<run>/raw/<slug>_v<version>_<UTC stamp>/` (raw, never modified),
     plus DOWNLOAD.json there (kernel, version, time, sha256 of every file);
  3. finds the export folder (RESUME.json of the pathway), verifies every file against RESUME.json, loads the
     merged table with the chain checks and confirms all Stage 2 records are present and unchanged;
  4. copies the generation tables to Sam/data/generated/<dataset>/<translator>/ and the reports, logs and the
     export zip to Sam/results/<run>/, each with PROVENANCE.json, plus progress_events.jsonl and summary.md.
An existing file with different content is never overwritten: the run stops (or, with --replace, the old
file is moved to superseded/<stamp>/ first). Everything with generated text is gitignored (Sam/.gitignore).

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/download_kaggle_outputs.py                 # the smoke deployment
  Sam/.venv/bin/python Sam/scripts/download_kaggle_outputs.py --raw-dir <existing raw folder>   # re-file only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import kaggle_cli, paths, runner, schema, session  # noqa: E402
from sot_mt.files import sha256_file  # noqa: E402

DEFAULT_DEPLOY = paths.SAM_ROOT / "kaggle" / "madlad_smoke5"
GENERATION_FILES = [runner.MERGED, runner.DELTA, runner.FAILURES, runner.RUNS, runner.RESUME]
REPORT_FILES = [runner.REPORT, runner.ATTRITION, runner.NEW_TEXTS, "tokenizer_parity.csv", "drift_probe.csv",
                "sanity_probe.csv", "gpu_memory.json"]


class ProvenanceError(RuntimeError):
    pass


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def listing(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def download(cli_run, kernel: str, raw_dir: Path, version=None, allow_incomplete=False) -> Path:
    rc, text = cli_run("kernels", "status", kernel)
    status, failure = kaggle_cli.parse_status(text)
    if status != "COMPLETE" and not allow_incomplete:
        raise SystemExit(f"{kernel} status is {status or text.strip()[:200]}" +
                         (f' ("{failure}")' if failure else "") + "; nothing downloaded")
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise SystemExit(f"{raw_dir} is not empty; raw downloads are never overwritten")
    raw_dir.mkdir(parents=True, exist_ok=True)
    rc, text = cli_run("kernels", "output", kernel, "-p", str(raw_dir), "-o")
    if rc != 0:
        raise SystemExit(f"download failed: {text.strip()[:500]}")
    info = {"kernel": kernel, "version": version, "status_at_download": status, "downloaded_at": stamp(),
            "files": listing(raw_dir)}
    (raw_dir / "DOWNLOAD.json").write_text(json.dumps(info, indent=1) + "\n")
    return raw_dir


def find_export(raw: Path, translator: str, prompt_variant: str) -> Path:
    hits = [p.parent for p in sorted(raw.rglob(runner.RESUME))
            if json.loads(p.read_text()).get("pathway") == [translator, prompt_variant]
            and (p.parent / runner.MERGED).exists()]
    if len(hits) != 1:
        raise ProvenanceError(f"expected one {translator}/{prompt_variant} export in {raw}, found {len(hits)}")
    return hits[0]


def verify_export(exp: Path, staged_dir: Path, allow_mock=False) -> dict:
    resume = json.loads((exp / runner.RESUME).read_text())
    for name, sha in resume["files"].items():
        if not (exp / name).exists() or sha256_file(exp / name) != sha:
            raise ProvenanceError(f"{name} does not match RESUME.json")
    manifest, staged = session.load_inputs(staged_dir)        # staged hashes verified here
    merged = runner.load_tables([exp / runner.MERGED], manifest)  # duplicate / conflict / chain checks
    cols = ["record_id", "output_sha256", "validation_status", "generation_config_id"]
    s = staged[cols].sort_values("record_id").reset_index(drop=True)
    m = merged[merged.record_id.isin(s.record_id)][cols].sort_values("record_id").reset_index(drop=True)
    if len(m) != len(s) or not m.fillna("").equals(s.fillna("")):
        raise ProvenanceError("the merged table does not contain the Stage 2 records unchanged")
    delta = pd.read_parquet(exp / runner.DELTA) if (exp / runner.DELTA).exists() else schema.empty_table()
    if not allow_mock and len(delta) and delta.run_id.astype(str).str.startswith(runner.MOCK_PREFIX).any():
        raise ProvenanceError("export contains MOCK records")
    return {"resume": resume, "merged_records": len(merged), "stage2_records": len(s), "new_records": len(delta),
            "dataset": ",".join(sorted(set(delta.dataset))) if len(delta) else None, "delta": delta}


def place(src: Path, dest_dir: Path, replace: bool, moved: list) -> str:
    dest = dest_dir / src.name
    if dest.exists():
        if sha256_file(dest) == sha256_file(src):
            return "unchanged"
        if not replace:
            raise ProvenanceError(f"{dest} exists with different content (use --replace to supersede it)")
        old = dest_dir / "superseded" / stamp()
        old.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dest), old / dest.name)
        moved.append(str(old / dest.name))
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return "copied"


def summarize(v: dict, report: pd.DataFrame, events: list, exp: Path) -> str:
    ev = {e["event"]: e for e in events}
    lines = ["# Kaggle run summary", "", f"- export: `{exp.name}`; merged records {v['merged_records']} "
             f"(Stage 2: {v['stage2_records']}, new: {v['new_records']})"]
    if "run_end" in ev:
        e = ev["run_end"]
        lines.append(f"- run status: {e.get('status')}; operations {e.get('operations')}, valid {e.get('valid')}, "
                     f"invalid {e.get('invalid')}, failed attempts {e.get('failed_attempts')}")
        lines.append(f"- model load {e.get('load_s')} s; translation time {e.get('translate_elapsed_s')} s; "
                     f"{e.get('ops_per_min')} ops/min; {e.get('words_per_s')} words/s; segments {e.get('segments_done')}")
    if "probe_model_loaded" in ev:
        lines.append(f"- model load before the first probe: {ev['probe_model_loaded'].get('probe_load_s')} s")
    if "parity_done" in ev:
        lines.append(f"- tokenizer parity: {ev['parity_done']['matched']}/{ev['parity_done']['checked']}")
    probes = [e for e in events if e["event"] == "probe_done"]
    sanity = [e for e in probes if e.get("probe") == "sanity" or str(e.get("item_id", "")).startswith("sanity")]
    drift = [e for e in probes if e not in sanity and e.get("probe", "drift") == "drift"]
    if drift:
        lines.append(f"- drift probe: {sum(bool(e.get('identical')) for e in drift)}/{len(drift)} S1 identical to round 1")
    if sanity:
        lines.append(f"- sanity probe: {sum(bool(e.get('ok')) for e in sanity)}/{len(sanity)} fixed sentences ok")
    if len(report):
        lines += ["", "| stage | valid | invalid | median seconds | median length ratio |", "|---|---|---|---|---|"]
        for st, g in report.groupby("output_stage", sort=False):
            lines.append(f"| {st} | {(g.validation_status == 'valid').sum()} | {(g.validation_status == 'invalid').sum()}"
                         f" | {g.seconds.median()} | {g.length_ratio.median()} |")
        bad = report[report.validation_status == "invalid"]
        for r in bad.itertuples():
            lines.append(f"- invalid: {r.item_id} {r.output_stage}: {r.validation_reason}")
    return "\n".join(lines) + "\n"


def organize(raw: Path, staged_dir: Path, generated_root: Path, results_dir: Path, translator: str,
             prompt_variant: str, replace=False, allow_mock=False) -> dict:
    exp = find_export(raw, translator, prompt_variant)
    v = verify_export(exp, staged_dir, allow_mock)
    gen_dir = generated_root / (v["dataset"] or "unknown") / translator
    moved, placed = [], {}
    for name in GENERATION_FILES:
        if (exp / name).exists():
            placed[f"generated/{name}"] = place(exp / name, gen_dir, replace, moved)
    for name in REPORT_FILES:
        if (exp / name).exists():
            placed[f"results/{name}"] = place(exp / name, results_dir, replace, moved)
    for z in sorted(exp.parent.glob(exp.name + ".zip")):
        placed[f"results/{z.name}"] = place(z, results_dir, replace, moved)
    logs = sorted(p for p in raw.rglob("*.log"))
    parser = kaggle_cli.EventParser()
    for lg in logs:
        parser.feed(kaggle_cli.log_lines(lg.read_text(errors="replace")))
        placed[f"results/{lg.name}"] = place(lg, results_dir, replace, moved)
    parser.close()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "progress_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in parser.events))
    report = pd.read_csv(exp / runner.REPORT) if (exp / runner.REPORT).exists() else pd.DataFrame()
    (results_dir / "summary.md").write_text(summarize(v, report, parser.events, exp))
    prov = {"raw_download": str(raw), "export_dir": str(exp), "download": (json.loads((raw / "DOWNLOAD.json")
            .read_text()) if (raw / "DOWNLOAD.json").exists() else None), "organized_at": stamp(),
            "resume_files": v["resume"]["files"], "merged_records": v["merged_records"],
            "new_records": v["new_records"], "stage2_records_verified": v["stage2_records"],
            "superseded": moved, "progress_events": len(parser.events), "malformed_progress_lines": parser.malformed}
    for d in (gen_dir, results_dir):
        (d / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, default=str) + "\n")
    return {"generated_dir": gen_dir, "results_dir": results_dir, "placed": placed, **prov}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-dir", default=str(DEFAULT_DEPLOY))
    ap.add_argument("--kernel")
    ap.add_argument("--kaggle")
    ap.add_argument("--results-dir", help="default: deploy.json results_dir (under Sam/)")
    ap.add_argument("--generated-root", help="default: deploy.json generated_root (under Sam/)")
    ap.add_argument("--raw-dir", help="use an existing raw download (skip the Kaggle call)")
    ap.add_argument("--prompt-variant")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--replace", action="store_true", help="supersede differing local files (moved, never deleted)")
    ap.add_argument("--raw-only", action="store_true",
                    help="failed/cancelled runs: download whatever output exists (+ DOWNLOAD.json), file nothing")
    a = ap.parse_args()
    deploy = json.loads((Path(a.deploy_dir) / "deploy.json").read_text())
    kernel = a.kernel or deploy["kernel"]
    translator = deploy["translator"]
    pv = a.prompt_variant or ("free" if translator in {"qwen3-4b", "llama", "phi4-mini"} else "mt-default")
    results = Path(a.results_dir) if a.results_dir else paths.SAM_ROOT / deploy.get("results_dir", "results/madlad_smoke5")
    gen_root = Path(a.generated_root) if a.generated_root else paths.SAM_ROOT / deploy.get("generated_root", "data/generated")
    if a.raw_dir:
        raw = Path(a.raw_dir)
    else:
        push = Path(a.deploy_dir) / "last_push.json"
        version = json.loads(push.read_text())["version"] if push.exists() else None
        cli = kaggle_cli.find_cli(a.kaggle)
        raw = results / "raw" / f"{kernel.split('/')[1]}_v{version or 'latest'}_{stamp()}"
        download(lambda *x: kaggle_cli.run(cli, *x, timeout=1800), kernel, raw, version,
                 a.allow_incomplete or a.raw_only)
    if a.raw_only:
        print(f"raw download kept in {raw} (DOWNLOAD.json lists sha256 of every file); nothing filed")
        return
    out = organize(raw, paths.staged_dir(), gen_root, results, translator, pv, a.replace)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in out.items() if k != "resume_files"},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
