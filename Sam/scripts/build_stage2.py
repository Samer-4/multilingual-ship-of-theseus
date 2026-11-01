"""Stage 2 build: stage corpus -> T0 manifest -> import MADLAD round 1 -> validation -> metric regression.

CPU only, no models. Safe to re-run: immutable files are only rewritten if identical, and the MADLAD
import is idempotent (second run inserts 0 records).

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/build_stage2.py --corpus-src ../Ship_of_theseus_paraphrased_copus
  Sam/.venv/bin/python Sam/scripts/build_stage2.py          # once the corpus is staged
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import corpus, legacy_madlad, manifest, paths, regression, schema, summaries  # noqa: E402
from sot_mt.files import load_yaml, sha256_file, write_csv, write_immutable_parquet  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
                                       for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-src", help="local clone of the upstream corpus (only needed to stage it)")
    a = ap.parse_args()

    cfg, tcfg = load_yaml("experiment.yaml"), load_yaml("translators.yaml")
    out = paths.output_root()
    staged, reports = out / "data" / "staged", out / "reports" / "stage2"
    cdir = paths.corpus_dir()

    # -- corpus ------------------------------------------------------------------------------------
    if a.corpus_src:
        corpus.stage_corpus(Path(a.corpus_src), cdir, cfg["corpus"]["files"], cfg["corpus"]["commit"])
    info = corpus.staged_info(cdir)
    if not info:
        sys.exit(f"corpus not staged at {cdir}; run once with --corpus-src <upstream clone> or set SOT_CORPUS_DIR")
    if info["repo_commit"] != cfg["corpus"]["commit"]:
        sys.exit(f"staged corpus commit {info['repo_commit']} != config {cfg['corpus']['commit']}")
    ds = cfg["sample"]["dataset"]
    raw = corpus.load_paraphrased(cdir, ds)

    # -- Task A: manifest ----------------------------------------------------------------------------
    legacy_rel = cfg["legacy"]["madlad_chain_csv"]
    legacy_path = paths.repo_file(legacy_rel)
    legacy_sha_before = sha256_file(legacy_path)
    legacy = legacy_madlad.read_legacy_chain(legacy_path)
    man = manifest.build_manifest(raw, cfg, cdir / corpus.paraphrased_file(ds), info["repo_commit"])
    manifest.check_against_legacy(man, legacy)
    man_status = write_immutable_parquet(man, staged / "t0_manifest.parquet")
    write_csv(man.drop(columns=["t0_text"]), out / "data" / "manifests" / f"{cfg['sample']['sample_id']}_ids.csv")
    prof = manifest.profile(man)

    # -- Task B/C: import MADLAD round 1 -----------------------------------------------------------
    gcid, gcfg = legacy_madlad.generation_config(tcfg)
    write_csv(pd.DataFrame([{"generation_config_id": gcid, "translator": gcfg["translator"],
                             "prompt_variant": gcfg["prompt_variant"], "model_checkpoint": gcfg["model_checkpoint"],
                             "verified": tcfg["translators"][gcfg["translator"]]["verified"],
                             "config_json": json.dumps(gcfg, sort_keys=True)}]),
              out / "data" / "tables" / "generation_configs.csv")
    rows = legacy_madlad.import_rows(legacy, man, cfg, tcfg, legacy_rel, legacy_sha_before)
    gen_path = staged / "generations.parquet"
    existing = pd.read_parquet(gen_path) if gen_path.exists() else schema.empty_table()
    gen, stats = schema.upsert(existing, rows, man, allow_invalid_predecessor=True)
    if stats["inserted"]:                     # never rewrite the staged Stage 2 table when nothing changed
        gen.to_parquet(gen_path, index=False)
    violations = schema.chain_violations(gen, man)
    if sha256_file(legacy_path) != legacy_sha_before:
        sys.exit("legacy CSV changed during the build")

    vsum = summaries.validation_summary(gen, man)
    att = schema.attrition(gen, man)
    flags = summaries.item_flags(gen)
    complete_t1 = schema.complete_cases(gen, man, "T1")
    elig_s2 = schema.eligible_inputs(gen, man, cfg["experiment_id"], "madlad-3b", "mt-default", "S2")
    legacy_valid = set(legacy.item_id[~legacy.bad])
    changed_vs_legacy = sorted(set(complete_t1) ^ legacy_valid)

    # -- Task D: metric regression -------------------------------------------------------------------
    t1 = gen[(gen.output_stage == "T1")].set_index("item_id")
    lf = t1.legacy_fields.map(json.loads)
    pairs = (man.set_index("item_id").loc[sorted(legacy_valid, key=man.item_id.tolist().index)]
             .assign(B_F=t1.output_text, seconds=lf.map(lambda d: d["seconds_round_trip"]))
             .reset_index()[["item_id", "source_label", "t0_words", "seconds", "t0_text", "B_F"]]
             .rename(columns={"source_label": "source", "t0_words": "words", "t0_text": "S"}))
    recomputed = legacy_madlad.recompute_legacy_metrics(pairs, paths.repo_file(cfg["legacy"]["madlad_notebook"]))
    # round_trip parsing: pandas' default fast float parser can be off by 1 ULP from the written value
    recorded = pd.read_csv(paths.repo_file(cfg["legacy"]["madlad_metrics_csv"]), float_precision="round_trip")
    per_item = regression.per_item_comparison(recomputed, recorded)
    reg = regression.summary(per_item, cfg["readme_madlad_medians"])
    pilot_items = regression.pilot_reference(pairs)
    pilot = regression.pilot_summary(pilot_items)
    clashes = regression.compare_name_clashes(per_item, pilot_items)

    # -- write reports -------------------------------------------------------------------------------
    for name, df in {"sample_counts": prof["counts"], "sample_sources": prof["sources"],
                     "sample_length": prof["length"], "sample_texts_per_article": prof["texts_per_article"],
                     "sample_shared_prefix": prof["shared_prefix"], "validation_summary": vsum,
                     "attrition_by_pathway_stage": att, "flagged_records": flags,
                     "metric_regression_summary": reg, "metric_regression_per_item": per_item,
                     "metric_pilot_reference": pilot, "metric_name_clashes": clashes}.items():
        write_csv(df, reports / f"{name}.csv")
    staged_files = {p.name: sha256_file(p) for p in sorted(staged.glob("*.parquet"))}
    (staged / "STAGED_FILES.json").write_text(json.dumps(
        {"files": staged_files, "corpus_commit": info["repo_commit"], "legacy_chain_csv_sha256": legacy_sha_before},
        indent=1))

    report = f"""# Stage 2 build report (generated by Sam/scripts/build_stage2.py)

Every number below is computed by the build script; nothing is typed by hand.

## Task A: sample manifest
- Corpus: `{cfg['corpus']['repo']}` @ `{info['repo_commit']}`, file `{corpus.paraphrased_file(ds)}` (sha256 `{info['files'][corpus.paraphrased_file(ds)][:16]}…`)
- Reproduced selection matches the legacy MADLAD chain CSV: **same {len(man)} item_ids, same order, identical T0 text** (check_against_legacy passed)
- Manifest file: `data/staged/t0_manifest.parquet` ({man_status})

{md_table(prof['counts'])}

Source distribution:

{md_table(prof['sources'])}

T0 length (whitespace words):

{md_table(prof['length'].reset_index(drop=True))}

Texts per article:

{md_table(prof['texts_per_article'])}

Shared opening with the Human original of the same article (observation only, no text removed):

{md_table(prof['shared_prefix'])}

## Task C: MADLAD round-1 import
- Records: {len(gen)} total (inserted this run: {stats['inserted']}, unchanged: {stats['unchanged']}); chain violations: {len(violations)}
- Legacy CSV sha256 before/after: `{legacy_sha_before[:16]}…` (unchanged)

{md_table(vsum.drop(columns=['experiment_id']))}

Attrition:

{md_table(att.drop(columns=['experiment_id']))}

Flagged records (invalid or with warnings):

{md_table(flags.drop(columns=['experiment_id']))}

- Items valid through T1 (complete cases): {len(complete_t1)}; legacy valid: {len(legacy_valid)}; differing: {changed_vs_legacy or 'none'}
- Items eligible to start round 2 (S2): {len(elig_s2)}

## Task D: metric regression
{md_table(reg[['metric_id', 'n_compared', 'n_exact', 'max_abs_diff', 'recomputed_median', 'readme_median', 'status']])}

Same name, different definition (medians on the same {len(pairs)} T0/T1 pairs):

{md_table(clashes)}
"""
    (reports / "STAGE2_REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
