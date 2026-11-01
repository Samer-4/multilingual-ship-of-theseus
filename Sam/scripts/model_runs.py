"""Kaggle runs of the other three translators (NLLB, Qwen, Llama): one real-model preflight each, then one
independent full run each over all 200 original T0 documents. Every model starts from T0; no model ever reads
another model's output (each pathway has its own records, the runner only chains a pathway onto itself).

Preflight (`sam-sot-mt-<model>-preflight`): the same 3 T0 documents for every model
(data/manifests/preflight3.csv: short / median / longest), all six stages T0 -> S1 -> ... -> T3, the sanity gate
(model load, fixed sentences), validation, export, GPU memory. Nothing is reused.

Full run (`sam-sot-mt-<model>-full`, attempt K: `-rK`): all 200 documents, on the model's own downloaded preflight
export (pinned by sha256; its 18 records are kept, never redone), the loaded weights pinned to the preflight's
weights id. The session timeout comes from the preflight's MEASURED seconds per operation, fitted per direction
(en->es and es->en: a + b * input words, least squares on its 9 + 9 operations; the LLMs' Spanish output costs
more tokens per word), applied to every remaining document's own length carried through the measured length
ratios, plus the measured setup time; when the estimate
does not fit one 12 h session, the run stops cleanly at TIME_LIMIT_MINUTES and continues as attempt 2, 3, ...
on the previous attempt's export. Refuses to plan without the downloaded preflight.

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/model_runs.py ids                       # writes data/manifests/preflight3.csv
  Sam/.venv/bin/python Sam/scripts/prepare_kaggle_deploy.py --preset preflight --model nllb
  Sam/.venv/bin/python Sam/scripts/model_runs.py estimate --model nllb     # after the preflight is downloaded
  Sam/.venv/bin/python Sam/scripts/prepare_kaggle_deploy.py --preset full --model nllb [--resume-attempt 2]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import paths, runner, session, smoke  # noqa: E402

USER = "samahmed17"
MODELS = {   # translator -> (slug part, Kaggle model_sources)
    "nllb": ("nllb", []),
    "qwen3-4b": ("qwen3-4b", []),
    "llama": ("llama", ["metaresearch/llama-3.2/transformers/3b-instruct/1"]),   # blocked: kept, not run
    "phi4-mini": ("phi4-mini", []),
}
PREFLIGHT_IDS = "data/manifests/preflight3.csv"
SESSION_LIMIT_S = 12 * 3600          # Kaggle GPU session limit (CLI -t is capped here)
MARGIN_S = 20 * 60                   # setup overrun + last operation + export, kept free before the hard kill
STAGES = ["S1", "T1", "S2", "T2", "S3", "T3"]
COMMON = {"PROMPT_VARIANT": "None", "ROUNDS": "(1, 3)", "MOCK": "False", "ALLOW_MODEL_DOWNLOAD": "True",
          "USE_HF_TOKEN": "False", "CHECK_TOKENIZER_PARITY": "False", "DRIFT_PROBE": "False",
          "MADLAD_DIAGNOSTIC": "False", "SANITY_PROBE": "True", "PREVIOUS_EXPORTS": '"auto"',
          "CODE_DATASET": '"sam-sot-mt-code"', "STAGED_DATASET": '"sam-sot-mt-staged"', "CHECKPOINT_EVERY": "5"}


def slug(model: str, kind: str, attempt: int = 1) -> str:
    return f"sam-sot-mt-{MODELS[model][0]}-{kind}" + (f"-r{attempt}" if attempt > 1 else "")


def tag(kind: str, attempt: int = 1) -> str:
    return kind + (f"_r{attempt}" if attempt > 1 else "")


def export_dir(model: str, kind: str, attempt: int = 1) -> str:
    return f"data/generated/{MODELS[model][0]}/{tag(kind, attempt)}/xsum/{model}"


def results_dir(model: str, kind: str, attempt: int = 1) -> str:
    return f"results/{MODELS[model][0]}/{tag(kind, attempt)}"


def _common(model: str, kind: str, attempt: int = 1) -> dict:
    return {"user": USER, "slug": slug(model, kind, attempt),
            "out": paths.SAM_ROOT / "kaggle" / MODELS[model][0] / tag(kind, attempt),
            "results_dir": results_dir(model, kind, attempt),
            "generated_root": f"data/generated/{MODELS[model][0]}/{tag(kind, attempt)}",
            "model_sources": MODELS[model][1]}


def preflight_preset(model: str) -> dict:
    return {**_common(model, "preflight"), "timeout_s": 5400, "resume_kernels": [], "resume_exports": [],
            "overrides": {**COMMON, "TRANSLATOR": json.dumps(model), "ITEM_IDS_FILE": json.dumps(PREFLIGHT_IDS),
                          "MAX_ITEMS": "None", "TIME_LIMIT_MINUTES": "60", "EXPORT_EVERY": "6",
                          "EXPECTED_WEIGHTS_ID": "None"}}


# -- measured runtime ------------------------------------------------------------------------------------

def measured(model: str) -> dict:
    """Per-operation timing, length ratios, setup time, weights id and GPU memory of the downloaded preflight."""
    exp, res = paths.SAM_ROOT / export_dir(model, "preflight"), paths.SAM_ROOT / results_dir(model, "preflight")
    if not (exp / runner.RESUME).exists():
        raise FileNotFoundError(f"{exp}: the {model} preflight is not downloaded; download it first")
    d = pd.read_parquet(exp / runner.DELTA)
    d = d[d.translator == model]
    meta = d.generation_meta.map(json.loads)
    words = d.input_text.str.split().str.len()
    secs = meta.map(lambda m: m["seconds"]).astype(float)
    n = len(d)

    def fit(mask):
        x, y = words[mask], secs[mask]
        b = ((x - x.mean()) * (y - y.mean())).sum() / max(1e-9, ((x - x.mean()) ** 2).sum())
        return float(y.mean() - b * x.mean()), float(b)
    direction = d.output_stage.str[0]
    fits = {k: fit(direction == k) for k in ("S", "T")}
    pred = pd.Series([max(1.0, fits[k][0] + fits[k][1] * w) for k, w in zip(direction, words)], index=d.index)
    a, b = fit(direction.isin(["S", "T"]))
    out_words = d.output_text.fillna("").str.split().str.len()
    ratio = (out_words / words).groupby(d.output_stage.str[0]).median().to_dict()     # S: en->es, T: es->en
    events = [json.loads(x) for x in (res / "progress_events.jsonl").read_text().splitlines()] \
        if (res / "progress_events.jsonl").exists() else []
    t = {e["event"]: e["ts"] for e in reversed(events)}
    setup = (pd.Timestamp(t["run_start"]) - pd.Timestamp(t["notebook_start"])).total_seconds() \
        if {"run_start", "notebook_start"} <= set(t) else None
    runs = pd.read_parquet(exp / runner.RUNS)
    lr = [json.loads(x) for x in runs.load_report if isinstance(x, str) and x != "null"]
    gpu = json.loads((res / "gpu_memory.json").read_text()) if (res / "gpu_memory.json").exists() else None
    return {"operations": n, "valid": int((d.validation_status == "valid").sum()), "fit_a_s": round(a, 3),
            "fit_b_s_per_word": round(b, 5),
            "fit_by_direction": {"en->es": [round(fits["S"][0], 3), round(fits["S"][1], 5)],
                                 "es->en": [round(fits["T"][0], 3), round(fits["T"][1], 5)]},
            "mae_s": round(float((secs - pred).abs().mean()), 2),
            "seconds_total": round(float(secs.sum()), 1), "words_per_s": round(float(words.sum() / secs.sum()), 2),
            "ratio_es_over_en": ratio.get("S"), "ratio_en_over_es": ratio.get("T"), "setup_s": setup,
            "weights_id": (lr[-1] if lr else {}).get("weights_id"), "gpu_memory": gpu,
            "items": sorted(set(d.item_id))}


def estimate(model: str) -> dict:
    """Seconds for the remaining full-run operations: each document's six inputs are T0 words carried through
    the measured median length ratios; minus the preflight's own (already done) items."""
    m = measured(model)
    manifest, _ = session.load_inputs()
    todo = manifest[~manifest.item_id.isin(m["items"])]
    r_s, r_t = m["ratio_es_over_en"] or 1.1, m["ratio_en_over_es"] or 0.9
    total = 0.0
    for w in todo.t0_words:
        x = float(w)
        for st in STAGES:
            fa, fb = m["fit_by_direction"]["en->es" if st[0] == "S" else "es->en"]
            total += max(1.0, fa + fb * x)
            x *= r_s if st[0] == "S" else r_t
    setup = m["setup_s"] if m["setup_s"] is not None else 600.0
    per_session = SESSION_LIMIT_S - setup - MARGIN_S
    sessions = max(1, math.ceil(total / per_session))
    timeout = SESSION_LIMIT_S if sessions > 1 else int(min(SESSION_LIMIT_S, max(3600, 1.5 * total + setup + MARGIN_S)))
    limit_min = int((timeout - setup - MARGIN_S) // 60)
    return {**m, "remaining_items": len(todo), "remaining_operations": 6 * len(todo),
            "est_translate_s": round(total), "est_sessions": sessions, "timeout_s": timeout,
            "time_limit_minutes": limit_min, "setup_s_used": setup}


def full_preset(model: str, attempt: int = 1) -> dict:
    est = estimate(model)
    if attempt == 1:
        resume_k, resume_e = [f"{USER}/{slug(model, 'preflight')}"], [export_dir(model, "preflight")]
    else:
        resume_k, resume_e = [f"{USER}/{slug(model, 'full', attempt - 1)}"], [export_dir(model, "full", attempt - 1)]
    return {**_common(model, "full", attempt), "timeout_s": est["timeout_s"], "estimate": est,
            "resume_kernels": resume_k, "resume_exports": resume_e,
            "overrides": {**COMMON, "TRANSLATOR": json.dumps(model), "ITEM_IDS_FILE": "None", "MAX_ITEMS": "None",
                          "TIME_LIMIT_MINUTES": str(est["time_limit_minutes"]), "EXPORT_EVERY": "20",
                          "EXPECTED_WEIGHTS_ID": json.dumps(est["weights_id"])}}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ids", help=f"write {PREFLIGHT_IDS} (never overwrites)")
    e = sub.add_parser("estimate", help="full-run estimate from the downloaded preflight")
    e.add_argument("--model", choices=sorted(MODELS), required=True)
    a = ap.parse_args()
    if a.cmd == "ids":
        out = paths.SAM_ROOT / PREFLIGHT_IDS
        if out.exists():
            raise SystemExit(f"{out} exists; the preflight documents are never re-chosen")
        manifest, _ = session.load_inputs()
        ids = smoke.select_preflight_items(manifest)
        ids.to_csv(out, index=False)
        print(ids.to_string(index=False), f"\n-> {out}")
    else:
        print(json.dumps(estimate(a.model), indent=1, default=str))


if __name__ == "__main__":
    main()
