"""Prepare a Kaggle kernel deployment folder (local only: nothing is uploaded or pushed).

  1. builds the code and staged upload folders (scripts/package_kaggle_inputs.py) and records the code digest;
  2. writes <out>/00_generate_translations.ipynb: the notebook with its configuration cell overridden;
  3. writes <out>/kernel-metadata.json (private, GPU T4, Internet on, both datasets attached);
  4. validates the metadata against the rules of the installed Kaggle CLI (2.2.5 kernels_push);
  5. writes <out>/deploy.json (kernel id, code digest, timeout) for launch_kaggle.py / monitor / download.

Usage (from the repo root), the Stage 3B smoke test:
  Sam/.venv/bin/python Sam/scripts/prepare_kaggle_deploy.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import paths  # noqa: E402
from sot_mt.files import sha256_file  # noqa: E402

NOTEBOOK = paths.SAM_ROOT / "notebooks" / "00_generate_translations.ipynb"
CODE_FILE = "00_generate_translations.ipynb"
ACCELERATOR = "NvidiaTeslaT4"          # CLI help: 'NvidiaTeslaT4' = GPU T4 x2
RETIRED = {"nvidiateslap100", "tpuv38", "tpu1vmv38", "tpuv232", "tpuv2256"}   # from the CLI source

SMOKE = {"user": "samahmed17", "slug": "sam-sot-mt-madlad-smoke5", "out": paths.SAM_ROOT / "kaggle" / "madlad_smoke5",
         "timeout_s": 5400,
         "overrides": {"TRANSLATOR": '"madlad-3b"', "PROMPT_VARIANT": "None",
                       "ITEM_IDS_FILE": '"data/manifests/stage3b_madlad_smoke5.csv"', "MAX_ITEMS": "5",
                       "ROUNDS": "(1, 3)", "MOCK": "False", "ALLOW_MODEL_DOWNLOAD": "True", "USE_HF_TOKEN": "False",
                       "CHECK_TOKENIZER_PARITY": "True", "DRIFT_PROBE": "True", "TIME_LIMIT_MINUTES": "60",
                       "PREVIOUS_EXPORTS": '"auto"', "CODE_DATASET": '"sam-sot-mt-code"',
                       "STAGED_DATASET": '"sam-sot-mt-staged"'}}


SMOKE["overrides"].update({"MADLAD_DIAGNOSTIC": "False", "REFERENCE_GATE": "True"})
# Rerun after the embedding fix: reuse the verified one-document chain (xsum-4729, S2-T3 from kernel
# sam-sot-mt-madlad-verify1 v2) by attaching that kernel's output; the notebook requires its
# generations_merged.parquet to be exactly the one downloaded and checked locally. New results go to new
# folders: results/madlad_smoke5 and data/generated/xsum/madlad-3b hold the quarantined first run.
SMOKE.update(results_dir="results/madlad_smoke5_v2", generated_root="data/generated/smoke5_v2",
             resume_kernels=["samahmed17/sam-sot-mt-madlad-verify1"],
             resume_exports=["data/generated/verify1/xsum/madlad-3b"])

# One-document real-model verification after the transformers 5.x embedding failure: load with the verified
# wiring, one gated reference check (T0 -> S1 vs round 1), then, only if it passes, the item's own 4 operations
# S2 -> T3. xsum-4729 has 2 segments per direction, so padding and
# batching in fp16 are exercised too.
VERIFY1 = {"user": "samahmed17", "slug": "sam-sot-mt-madlad-verify1", "out": paths.SAM_ROOT / "kaggle" / "madlad_verify1",
           "timeout_s": 2700, "results_dir": "results/madlad_verify1", "generated_root": "data/generated/verify1",
           "overrides": {**SMOKE["overrides"], "ITEM_IDS_FILE": '"data/manifests/stage3b_madlad_verify1.csv"',
                         "MAX_ITEMS": "1", "TIME_LIMIT_MINUTES": "30",
                         # v2: the wiring is now verified (runtime_patches: decoder), so one reference check
                         # (the drift probe, gated) replaces the three-way diagnostic
                         "MADLAD_DIAGNOSTIC": "False"}}
PRESETS = {"smoke5": SMOKE, "verify1": VERIFY1}

# Full MADLAD pilot: one kernel per batch (sam-sot-mt-madlad-pilot-bNN), each on the same pinned base: the smoke
# kernel's output (Stage 2 + 20 verified records). Batches are disjoint, so nothing is ever redone; a batch that
# stops early is resumed with --resume-attempt, which attaches that batch's own previous output (pinned too).
PILOT_PLAN = paths.SAM_ROOT / "data" / "manifests" / "madlad_pilot" / "PLAN.json"


def pilot_preset(batch: int, attempt: int = 1) -> dict:
    plan = json.loads(PILOT_PLAN.read_text())
    b = plan["batches"][batch - 1]
    name = b["batch"].replace("batch_", "b")
    slug = f"sam-sot-mt-madlad-pilot-{name}" + (f"-r{attempt}" if attempt > 1 else "")
    tag = name + (f"_r{attempt}" if attempt > 1 else "")
    resume_k = ["samahmed17/sam-sot-mt-madlad-smoke5"]          # its latest output: v3, the verified smoke run
    resume_e = ["data/generated/smoke5_v2/xsum/madlad-3b"]        # pinned by sha256 in the notebook
    if attempt > 1:          # the previous attempt's downloaded export (download it first, --raw-only if it failed)
        prev = f"sam-sot-mt-madlad-pilot-{name}" + (f"-r{attempt - 1}" if attempt > 2 else "")
        resume_k.append(f"samahmed17/{prev}")
        resume_e.append(f"data/generated/madlad_pilot/{name}" + (f"_r{attempt - 1}" if attempt > 2 else "") + "/xsum/madlad-3b")
    timeout = int(min(43200, max(3600, 3 * b["est_session_s"])))
    return {"user": "samahmed17", "slug": slug, "out": paths.SAM_ROOT / "kaggle" / "madlad_pilot" / tag,
            "timeout_s": timeout, "results_dir": f"results/madlad_pilot/{tag}",
            "generated_root": f"data/generated/madlad_pilot/{tag}", "resume_kernels": resume_k, "resume_exports": resume_e,
            "overrides": {**SMOKE["overrides"],
                          "ITEM_IDS_FILE": json.dumps(f"data/manifests/madlad_pilot/{b['file']}"),
                          "MAX_ITEMS": "None", "TIME_LIMIT_MINUTES": str(int(min(660, timeout / 60 - 15))),
                          "DRIFT_PROBE_MAX_ITEMS": "2", "CHECKPOINT_EVERY": "5"}}


# Consolidated remainder: one session for all 168 items after batch 1 (data/manifests/madlad_pilot_remainder), on
# batch 1's downloaded export (Stage 2 + the 20 smoke + 96 batch-1 records), pinned by sha256 and checked by
# verify_madlad_pilot.py first. Hard session limit 4 h (kernels push -t); the runner stops cleanly at
# TIME_LIMIT_MINUTES, leaving time for the export; a rolling export every EXPORT_EVERY operations keeps a
# resumable RESUME.json in the output even if the session is cut off. A resume attempt attaches only the previous
# attempt's output (its merged table already holds everything batch 1's did).
REMAINDER_IDS = "data/manifests/madlad_pilot_remainder/remainder_01.csv"
REMAINDER_BASE = "data/generated/madlad_pilot/b01/xsum/madlad-3b"
REMAINDER_CONTAINS = [REMAINDER_BASE, "data/generated/smoke5_v2/xsum/madlad-3b"]
REMAINDER_TIMEOUT_S = 4 * 3600
REMAINDER_TIME_LIMIT_MIN = 215       # translation stops here; setup (~4 min) + export (<1 min) fit before 240


def remainder_preset(attempt: int = 1) -> dict:
    tag = "remainder" + (f"_r{attempt}" if attempt > 1 else "")
    slug = "sam-sot-mt-madlad-pilot-" + tag.replace("_", "-")
    if attempt == 1:
        resume_k, resume_e = ["samahmed17/sam-sot-mt-madlad-pilot-b01"], [REMAINDER_BASE]
    else:
        prev = "remainder" + (f"_r{attempt - 1}" if attempt > 2 else "")
        resume_k = ["samahmed17/sam-sot-mt-madlad-pilot-" + prev.replace("_", "-")]
        resume_e = [f"data/generated/madlad_pilot/{prev}/xsum/madlad-3b"]
    return {"user": "samahmed17", "slug": slug, "out": paths.SAM_ROOT / "kaggle" / "madlad_pilot" / tag,
            "timeout_s": REMAINDER_TIMEOUT_S, "results_dir": f"results/madlad_pilot/{tag}",
            "generated_root": f"data/generated/madlad_pilot/{tag}", "resume_kernels": resume_k,
            "resume_exports": resume_e, "contains": REMAINDER_CONTAINS + resume_e[:1] * (attempt > 1),
            "overrides": {**SMOKE["overrides"], "ITEM_IDS_FILE": json.dumps(REMAINDER_IDS), "MAX_ITEMS": "None",
                          "TIME_LIMIT_MINUTES": str(REMAINDER_TIME_LIMIT_MIN),
                          # one reference probe per source label (the first 7 items, round-robin over labels)
                          "DRIFT_PROBE_MAX_ITEMS": "7", "CHECKPOINT_EVERY": "5", "EXPORT_EVERY": "20"}}


def slugify(title: str) -> str:
    """ASCII equivalent of python-slugify's default for the titles used here."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def override_config(nb: dict, overrides: dict) -> dict:
    """Replace `KEY = ...` lines in the configuration cell (the first code cell). Each key must occur once."""
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
    src = "".join(cell["source"])
    for key, value in overrides.items():
        rx = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
        if len(rx.findall(src)) != 1:
            raise KeyError(f"configuration cell has no single `{key} = ...` line")
        src = rx.sub(f"{key} = {value}   # set by prepare_kaggle_deploy.py", src)
    cell["source"] = src.splitlines(True)
    return nb


def kernel_metadata(user: str, slug: str, datasets: list, kernel_sources: list, model_sources=()) -> dict:
    return {"id": f"{user}/{slug}", "title": slug, "code_file": CODE_FILE, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": ACCELERATOR, "dataset_sources": datasets,
            "competition_sources": [], "kernel_sources": kernel_sources, "model_sources": list(model_sources),
            "keywords": []}


def validate(folder: Path, meta: dict) -> list:
    """Problems that `kaggle kernels push` (2.2.5) would reject, plus this project's own requirements."""
    errs = []
    title = meta.get("title") or ""
    if title and len(title) < 5:
        errs.append("title must be at least five characters")
    if not meta.get("code_file") or not (folder / meta["code_file"]).is_file():
        errs.append("code_file missing")
    kid = meta.get("id", "")
    parts = kid.split("/")
    if len(parts) != 2 or not all(parts):
        errs.append("id must be <owner>/<kernel-slug> without a version")
    elif title and slugify(title) != parts[1].lower():
        errs.append("title does not resolve to the id slug")
    if meta.get("language") not in {"python", "r", "rmarkdown"}:
        errs.append("language")
    if meta.get("kernel_type") not in {"script", "notebook"}:
        errs.append("kernel_type")
    for d in meta.get("dataset_sources", []):
        sp = d.split("/")
        if "/" not in d or not sp[0] or not sp[1] or len(sp) > 3:
            errs.append(f"dataset source {d!r}")
    for k in meta.get("kernel_sources", []):
        if len(k.split("/")) != 2:
            errs.append(f"kernel source {k!r}")
    for m in meta.get("model_sources", []):
        if len(m.split("/")) != 5:
            errs.append(f"model source {m!r} (owner/model/framework/variation/version)")
    if (meta.get("machine_shape") or "").lower() in RETIRED:
        errs.append("machine_shape is a retired accelerator")
    # project requirements
    if meta.get("is_private") is not True:
        errs.append("notebook must be private")
    if meta.get("enable_gpu") is not True or meta.get("machine_shape") != ACCELERATOR:
        errs.append("GPU T4 required")
    if meta.get("enable_internet") is not True:
        errs.append("Internet required (model download)")
    if code := folder / meta.get("code_file", ""):
        if code.is_file():
            nb = json.loads(code.read_text())
            if not isinstance(nb.get("cells"), list):
                errs.append("code_file is not a notebook")
    return errs


def cli_slugify(title: str):
    """Cross-check with the CLI's own python-slugify, if the CLI venv is present (no Kaggle import, no network)."""
    py = Path.home() / ".venvs/kaggle-cli/bin/python"
    if not py.exists():
        return None
    p = subprocess.run([str(py), "-I", "-c", "import sys; from slugify import slugify; print(slugify(sys.argv[1]))",
                        title], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def planned_export_sha(export_dirs, translator: str = "madlad-3b") -> list:
    """sha256 of each local export's generations_merged.parquet, after checking it against its RESUME.json,
    its pathway and the quarantine list. These pin exactly which attached kernel output may be reused."""
    from sot_mt.files import load_yaml
    bad_runs = set(load_yaml("experiment.yaml").get("quarantined_runs") or [])
    out = []
    for d in export_dirs:
        d = paths.SAM_ROOT / d if not Path(d).is_absolute() else Path(d)
        resume = json.loads((d / "RESUME.json").read_text())
        if resume["pathway"][0] != translator:
            raise ValueError(f"{d}: pathway {resume['pathway']} is not {translator}")
        sha = sha256_file(d / "generations_merged.parquet")
        if sha != resume["files"]["generations_merged.parquet"]:
            raise ValueError(f"{d}: generations_merged.parquet does not match its RESUME.json")
        import pandas as pd
        runs = set(pd.read_parquet(d / "generations_merged.parquet").run_id)
        if runs & bad_runs:
            raise ValueError(f"{d}: contains quarantined runs {sorted(runs & bad_runs)}")
        out.append(sha)
    return out


def prepare(user: str, slug: str, out: Path, overrides: dict, timeout_s: int, kernel_sources=(),
            build_out: Path = paths.SAM_ROOT / "build" / "kaggle", results_dir: str = None,
            generated_root: str = "data/generated", model_sources=()) -> dict:
    sys.path.insert(0, str(paths.SAM_ROOT / "scripts"))
    import package_kaggle_inputs as pk
    built = pk.build(Path(build_out), user, paths.staged_dir())
    out.mkdir(parents=True, exist_ok=True)
    nb = override_config(json.loads(NOTEBOOK.read_text()),
                         {**overrides, "EXPECTED_CODE_SHA": json.dumps(built["code_digest"])})
    (out / CODE_FILE).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    datasets = [f"{user}/sam-sot-mt-code", f"{user}/sam-sot-mt-staged"]
    meta = kernel_metadata(user, slug, datasets, list(kernel_sources), model_sources)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    errs = validate(out, meta)
    s = cli_slugify(meta["title"])
    if s is not None and s != slug:
        errs.append(f"CLI slugify gives {s!r}")
    if errs:
        raise ValueError(f"deployment invalid: {errs}")
    deploy = {"prepared_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "kernel": meta["id"],
              "accelerator": ACCELERATOR, "timeout_s": timeout_s, "code_digest": built["code_digest"],
              "code_dataset": datasets[0], "staged_dataset": datasets[1],
              "code_build_dir": str(built["code"]), "notebook_sha256": sha256_file(out / CODE_FILE),
              "overrides": overrides, "kernel_sources": list(kernel_sources),
              "translator": json.loads(overrides["TRANSLATOR"]), "cli_slug_check": s,
              "results_dir": results_dir or f"results/{slug}", "generated_root": generated_root,
              "model_sources": list(model_sources)}
    (out / "deploy.json").write_text(json.dumps(deploy, indent=1) + "\n")
    return deploy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS) + ["pilot", "remainder", "preflight", "full"], default="smoke5")
    ap.add_argument("--model", choices=["nllb", "qwen3-4b", "phi4-mini", "llama"], help="preflight / full: the translator")
    ap.add_argument("--batch", type=int, help="pilot: batch number from data/manifests/madlad_pilot/PLAN.json")
    ap.add_argument("--resume-attempt", type=int, default=1,
                    help="pilot / remainder: 2, 3, ... continues a run that stopped early (download it first)")
    ap.add_argument("--kaggle-user")
    ap.add_argument("--kernel-slug")
    ap.add_argument("--out")
    ap.add_argument("--timeout-s", type=int)
    ap.add_argument("--resume-from-kernel", nargs="*", default=[],
                    help="earlier kernels (<owner>/<slug>) whose output to attach; PREVIOUS_EXPORTS='auto' finds it")
    a = ap.parse_args()
    if a.preset in {"preflight", "full"}:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import model_runs as mr
        if not a.model:
            raise SystemExit("--model is required for --preset preflight / full")
        P = mr.preflight_preset(a.model) if a.preset == "preflight" else mr.full_preset(a.model, a.resume_attempt)
        if P.get("estimate"):
            print("measured preflight -> estimate:", json.dumps({k: v for k, v in P["estimate"].items()
                                                                 if k not in {"items", "gpu_memory"}}, default=str))
    else:
        P = (pilot_preset(a.batch, a.resume_attempt) if a.preset == "pilot" else
             remainder_preset(a.resume_attempt) if a.preset == "remainder" else PRESETS[a.preset])
    out = Path(a.out or P["out"])
    if a.preset in {"remainder", "preflight", "full"} and (out / "last_push.json").exists():
        raise SystemExit(f"{out} was already pushed; continue with --resume-attempt {a.resume_attempt + 1}")
    overrides = dict(P["overrides"])
    if P.get("contains"):                  # lineage: Stage 2 + every earlier record unchanged in the base
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import verify_madlad_pilot as vp
        r = vp.verify(P["resume_exports"][0], [d for d in P["contains"] if d != P["resume_exports"][0]])
        print(f"base verified: {r['records']} records, {r['new_records']} new, {r['complete_items']} complete items")
    if P.get("resume_exports"):
        overrides["EXPECTED_PREVIOUS_MERGED_SHA256"] = json.dumps(
            planned_export_sha(P["resume_exports"], json.loads(overrides["TRANSLATOR"])))
    d = prepare(a.kaggle_user or P["user"], a.kernel_slug or P["slug"], out, overrides,
                a.timeout_s or P["timeout_s"], a.resume_from_kernel or P.get("resume_kernels", []),
                results_dir=P["results_dir"], generated_root=P["generated_root"],
                model_sources=P.get("model_sources", []))
    print(json.dumps(d, indent=1))
    print(f"\nready: {out} (nothing uploaded). Launch with: "
          f"Sam/.venv/bin/python Sam/scripts/launch_kaggle.py --deploy-dir {out.relative_to(paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
