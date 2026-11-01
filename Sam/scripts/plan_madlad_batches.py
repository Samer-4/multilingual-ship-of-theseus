"""Plan the full MADLAD pilot (S2 -> T3 for every item with a valid round-1 T1 and no S2 yet) as Kaggle batches.

Base = the Stage 2 staged table + the smoke-test export (data/generated/smoke5_v2/...: 400 Stage 2 records and the
20 verified S2-T3 records), loaded with all chain/duplicate/conflict checks; staged hashes are verified first. The
remaining items are taken in manifest order (the round-robin over source labels), so each batch keeps the source
mix, and cut into consecutive batches of about --target-size documents with balanced estimated runtime.

Runtime estimate per operation (seconds) = a + b * (words per segment) * ceil(segments / 4), fitted on the 20 real
Kaggle T4 operations of verify1 + smoke5_v2 (MAE 1.3 s). Words: T1 for S2/S3, 1.08 x T1 for T2/T3 (Spanish is
longer); segments: round 1's own counts (chunks_en, chunks_es). Fixed per session: 15 s tokenizer parity +
85 s model load + the drift probes; container start-up is not measured and not included.

Writes (never overwrites): data/manifests/madlad_pilot/batch_NN.csv (IDs, article/group IDs, labels, estimates; no
text) and data/manifests/madlad_pilot/PLAN.json.
Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/plan_madlad_batches.py
The consolidated remainder (one session for everything after pilot batch 1, on batch 1's export):
  Sam/.venv/bin/python Sam/scripts/plan_madlad_batches.py --base-export data/generated/madlad_pilot/b01/xsum/madlad-3b \
      --out data/manifests/madlad_pilot_remainder --name remainder --target-size 1000 --min-size 1 --max-size 1000 \
      --probes-per-batch 7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import paths, runner, session  # noqa: E402
from sot_mt.files import load_yaml, sha256_file  # noqa: E402

BASE_EXPORT = paths.SAM_ROOT / "data" / "generated" / "smoke5_v2" / "xsum" / "madlad-3b"
OUT = paths.SAM_ROOT / "data" / "manifests" / "madlad_pilot"
FIT = {"a": -0.35, "b": 0.0958, "n": 20, "mae_s": 1.29, "source": "verify1 v2 + smoke5 v3 generation_meta.seconds"}
SESSION_FIXED_S = 15 + 85


def op_seconds(words: float, segments: int) -> float:
    segments = max(1, int(segments))
    return max(1.0, FIT["a"] + FIT["b"] * words / segments * math.ceil(segments / 4))


def remaining(base_export: Path = BASE_EXPORT) -> tuple:
    manifest, staged = session.load_inputs()
    exp_id = load_yaml("experiment.yaml")["experiment_id"]
    table = runner.load_tables([staged, base_export / runner.MERGED], manifest)
    pw = table[(table.experiment_id == exp_id) & (table.translator == "madlad-3b") & (table.prompt_variant == "mt-default")]
    st = pw.pivot(index="item_id", columns="output_stage", values="validation_status")
    todo = st.index[(st.S1 == "valid") & (st.T1 == "valid") & st.get("S2", pd.Series(index=st.index)).isna()]
    t1 = pw[pw.output_stage == "T1"].set_index("item_id")
    legacy = pw[pw.output_stage == "S1"].set_index("item_id").legacy_fields.map(json.loads)
    m = manifest.set_index("item_id").loc[todo].reset_index().sort_values("sample_index")
    m["t1_words"] = m.item_id.map(t1.output_text.str.split().str.len())
    m["round1_segments_en_es"] = m.item_id.map(legacy.map(lambda d: d["chunks_en"]))
    m["round1_segments_es_en"] = m.item_id.map(legacy.map(lambda d: d["chunks_es"]))
    m["est_seconds"] = [round(2 * (op_seconds(w, a) + op_seconds(1.08 * w, b)), 1)
                        for w, a, b in zip(m.t1_words, m.round1_segments_en_es, m.round1_segments_es_en)]
    done = st.index[st.get("T3", pd.Series(index=st.index)).notna()]
    return manifest, table, m.reset_index(drop=True), list(done)


def split(m: pd.DataFrame, target_size: int, min_size: int, max_size: int) -> list:
    """Consecutive batches (manifest order kept) with about equal estimated runtime: ceil(N / target) batches,
    item i goes to batch floor(time before i / (total / n)). Falls back to equal counts if a batch would leave
    [min_size, max_size]."""
    n = max(1, math.ceil(len(m) / target_size))
    before = m.est_seconds.cumsum() - m.est_seconds
    idx = (before / (m.est_seconds.sum() / n)).astype(int).clip(upper=n - 1)
    sizes = idx.value_counts()
    if len(sizes) != n or sizes.min() < min_size or sizes.max() > max_size:
        idx = pd.Series([i * n // len(m) for i in range(len(m))], index=m.index)
    return [m[idx == k] for k in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-size", type=int, default=25)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--max-size", type=int, default=30)
    ap.add_argument("--probes-per-batch", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--base-export", default=str(BASE_EXPORT.relative_to(paths.SAM_ROOT)),
                    help="downloaded export (relative to Sam/) whose merged table is the base")
    ap.add_argument("--name", default="batch", help="file prefix: <name>_NN.csv")
    a = ap.parse_args()
    base_export = paths.SAM_ROOT / a.base_export
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already holds a plan; batches are never re-cut in place (use --out)")
    manifest, table, m, done = remaining(base_export)
    labels = sorted(manifest.source_label.unique())
    batches = split(m, a.target_size, a.min_size, a.max_size)
    out.mkdir(parents=True)
    cols = ["item_id", "dataset", "article_id", "group_id", "source_label", "sample_index", "t1_words",
            "round1_segments_en_es", "round1_segments_es_en", "est_seconds"]
    plan = []
    for k, b in enumerate(batches, 1):
        name = f"{a.name}_{k:02d}"
        b = b.assign(batch=name)
        b[cols + ["batch"]].to_csv(out / f"{name}.csv", index=False)
        probe_s = sum(op_seconds(w, s) for w, s in zip(b.t1_words[:a.probes_per_batch],
                                                       b.round1_segments_en_es[:a.probes_per_batch]))
        est = b.est_seconds.sum() + SESSION_FIXED_S + probe_s
        plan.append({"batch": name, "items": len(b), "operations": 4 * len(b), "t1_words": int(b.t1_words.sum()),
                     "source_labels": dict(b.source_label.value_counts().sort_index()),
                     "all_labels_present": set(b.source_label) == set(labels),
                     "est_translate_s": round(b.est_seconds.sum()), "est_session_s": round(est),
                     "file": f"{name}.csv"})
    info = {"experiment_id": load_yaml("experiment.yaml")["experiment_id"], "translator": "madlad-3b",
            "prompt_variant": "mt-default", "generation_config_id": "705a8bd4992e",
            "base": {"staged_generations_sha256": sha256_file(paths.staged_dir() / "generations.parquet"),
                     "export": str(base_export.relative_to(paths.SAM_ROOT)),
                     "export_merged_sha256": sha256_file(base_export / runner.MERGED), "records": len(table)},
            "already_complete_items": sorted(done), "remaining_items": len(m), "remaining_operations": 4 * len(m),
            "runtime_model": FIT, "session_fixed_s": SESSION_FIXED_S, "probes_per_batch": a.probes_per_batch,
            "batches": plan}
    (out / "PLAN.json").write_text(json.dumps(info, indent=1, default=int) + "\n")
    print(pd.DataFrame(plan).drop(columns=["file", "source_labels"]).to_string(index=False))
    print(f"\n{len(m)} items / {4 * len(m)} operations in {len(plan)} batches -> {out}")


if __name__ == "__main__":
    main()
