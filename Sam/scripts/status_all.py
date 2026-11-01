"""One status table for all translator deployments: model, run state, documents completed, valid / invalid
operations, elapsed time and ETA, download and finalization status.

Reads local files (deploy.json, last_push.json, the downloaded export and results) and, unless --offline, asks
Kaggle for each pushed kernel's status and stored log (`kernels status`, `kernels logs`: read-only).
Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/status_all.py            # all deployments under Sam/kaggle/{nllb,qwen3-4b,llama}
  Sam/.venv/bin/python Sam/scripts/status_all.py --offline  # local files only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import kaggle_cli, paths, runner  # noqa: E402
from sot_mt.progress import fmt_seconds  # noqa: E402

GROUPS = ["nllb", "qwen3-4b", "phi4-mini", "llama"]


def deployments(root: Path = paths.SAM_ROOT / "kaggle", groups=GROUPS) -> list:
    return sorted(p.parent for g in groups for p in (root / g).glob("*/deploy.json"))


def local_state(d: Path) -> dict:
    dep = json.loads((d / "deploy.json").read_text())
    translator = dep["translator"]
    gen = paths.SAM_ROOT / dep["generated_root"] / "xsum" / translator
    res = paths.SAM_ROOT / dep["results_dir"]
    row = {"deployment": str(d.relative_to(paths.SAM_ROOT / "kaggle")), "model": translator,
           "kernel": dep["kernel"], "pushed": (d / "last_push.json").exists(),
           "downloaded": (gen / runner.RESUME).exists(), "finalized": (gen / "FINAL.json").exists()}
    if row["downloaded"]:
        delta = pd.read_parquet(gen / runner.DELTA)
        merged = pd.read_parquet(gen / runner.MERGED)
        pw = merged[merged.translator == translator]
        st = pw.pivot(index="item_id", columns="output_stage", values="validation_status")
        stages = [s for s in ["S1", "T1", "S2", "T2", "S3", "T3"] if s in st]
        row.update(docs_complete=int((st.reindex(columns=["S1", "T1", "S2", "T2", "S3", "T3"]) == "valid")
                                     .all(axis=1).sum()) if len(stages) == 6 else 0,
                   valid=int((pw.validation_status == "valid").sum()),
                   invalid=int((pw.validation_status == "invalid").sum()), this_run=len(delta))
    return row


def remote_state(kernel: str, cli_run) -> dict:
    rc, text = cli_run("kernels", "status", kernel)
    status, failure = kaggle_cli.parse_status(text)
    out = {"state": status or f"unknown (rc {rc})", "failure": failure}
    try:
        rc, text = cli_run("kernels", "logs", kernel)
    except Exception as e:  # noqa: BLE001  a log problem must not hide the status
        rc, text = -1, f"{type(e).__name__}: {e}"
    if rc != 0:
        out["phase"] = "log unavailable"
    else:
        p = kaggle_cli.EventParser()
        p.feed(kaggle_cli.log_lines(text))
        p.close()
        ev = p.last                       # a property: the latest progress event, or None
        if ev is None:
            out["phase"] = "no progress events yet"
        else:
            out.update(phase=ev.get("phase"), done=ev.get("done"), total=ev.get("total"),
                       valid_ops=ev.get("valid"), invalid_ops=ev.get("invalid"), elapsed=fmt_seconds(ev.get("elapsed_s")),
                       eta=ev.get("eta_text"))
    return out


def table(rows: list) -> str:
    cols = ["model", "deployment", "state", "phase", "ops", "valid/invalid", "docs complete", "elapsed", "eta",
            "downloaded", "finalized"]
    lines = []
    for r in rows:
        ops = f"{r.get('done')}/{r.get('total')}" if r.get("total") else "-"
        vi = (f"{r['valid_ops']}/{r['invalid_ops']}" if r.get("valid_ops") is not None else
              f"{r['valid']}/{r['invalid']}" if r.get("valid") is not None else "-")
        lines.append([r["model"], r["deployment"], r.get("state") or ("pushed" if r["pushed"] else "not launched"),
                      r.get("phase") or "-", ops, vi, str(r.get("docs_complete", "-")), r.get("elapsed") or "-",
                      r.get("eta") or "-", "yes" if r["downloaded"] else "no", "yes" if r["finalized"] else "no"])
    return pd.DataFrame(lines, columns=cols).to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--kaggle")
    a = ap.parse_args()
    cli = None if a.offline else kaggle_cli.find_cli(a.kaggle)
    rows = []
    for d in deployments():
        r = local_state(d)
        if cli and r["pushed"]:
            r.update(remote_state(r["kernel"], lambda *x: kaggle_cli.run(cli, *x)))
        rows.append(r)
    print(table(rows) if rows else "no deployments")


if __name__ == "__main__":
    main()
