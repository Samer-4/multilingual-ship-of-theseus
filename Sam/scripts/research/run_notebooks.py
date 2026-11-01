"""Execute the three research notebooks top-to-bottom in fresh kernels (in place, outputs saved).

They only read the cached shared tables (CPU work), so they run in parallel by default.
Usage (from the repo root, research environment):
  Sam/.venv-research/bin/python Sam/scripts/research/run_notebooks.py            # all three, in parallel
  Sam/.venv-research/bin/python Sam/scripts/research/run_notebooks.py 01 --serial
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SAM = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["01_RQ1_Translation_Decay.ipynb", "02_RQ2_Source_Attribution.ipynb",
             "03_RQ3_Translator_Fingerprint.ipynb"]


def command(nb: Path) -> list:
    return [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=7200", "--ExecutePreprocessor.kernel_name=sot-research", str(nb)]


def check(nb: Path) -> dict:
    d = json.loads(nb.read_text())
    code = [c for c in d["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    images = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    unexecuted = sum(1 for c in code if c.get("execution_count") is None)
    return {"notebook": nb.name, "code_cells": len(code), "unexecuted": unexecuted, "errors": len(errors),
            "figures": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="prefixes, e.g. 01 03 (default: all)")
    ap.add_argument("--serial", action="store_true")
    a = ap.parse_args()
    nbs = [SAM / "notebooks" / n for n in NOTEBOOKS if not a.which or any(n.startswith(w) for w in a.which)]
    started, results = time.time(), {}
    procs = {}
    for nb in nbs:
        log = open(SAM / "data" / "research" / f"run_{nb.stem}.log", "w")
        p = subprocess.Popen(command(nb), stdout=log, stderr=subprocess.STDOUT, cwd=SAM / "notebooks")
        procs[nb] = (p, time.time())
        if a.serial:
            p.wait()
    for nb, (p, t) in procs.items():
        rc = p.wait()
        results[nb.name] = {"returncode": rc, "seconds": round(time.time() - t, 1), **check(nb)}
        print(json.dumps(results[nb.name]))
    manifest = SAM / "results" / "research" / "run_manifests" / "notebook_runs.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    old.update({k: {**v, "executed_at": datetime.now(timezone.utc).isoformat()} for k, v in results.items()})
    manifest.write_text(json.dumps(old, indent=1))
    ok = all(r["returncode"] == 0 and r["errors"] == 0 and r["unexecuted"] == 0 for r in results.values())
    print(f"{'OK' if ok else 'FAILED'} in {time.time() - started:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
