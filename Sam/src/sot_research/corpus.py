"""Canonical research table, cohorts and coverage, built from the finalized translation datasets.

Inputs (read only):
  data/staged/t0_manifest.parquet                       the 200 original T0 documents (one row per item)
  data/generated/<name>_final/<dataset>/<translator>/   FINAL.json + generations_merged.parquet, one per translator

Outputs:
  data/research/texts.parquet (local, gitignored: contains text) one row per (item, translator, stage). T0 appears
      ONCE per item, with translator "original"; every translator has a full S1..T3 grid whose `status` is valid,
      invalid, blocked (an earlier stage of the chain is not valid) or missing. Spanish stages are kept.
  results/research/data_coverage/  text-free: chains.csv, cohort.csv, coverage_by_stage.csv,
      coverage_by_source.csv, translators.csv, lineage_report.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import paths

STAGES = ["S1", "T1", "S2", "T2", "S3", "T3"]
EN_STAGES = ["T0", "T1", "T2", "T3"]
ORIGINAL = "original"
ID_COLS = ["dataset", "item_id", "article_id", "group_id", "source_label", "is_human"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def research_config() -> dict:
    return yaml.safe_load((paths.CONFIG / "research.yaml").read_text())


def discover(generated: Path = paths.GENERATED) -> pd.DataFrame:
    """Every finalized translator dataset: one row per (dataset, translator) with its FINAL.json facts."""
    rows = []
    for fin in sorted(generated.glob("*_final/*/*/FINAL.json")):
        meta = json.loads(fin.read_text())
        rows.append({"dataset": fin.parent.parent.name, "translator": fin.parent.name, "final_dir": fin.parent,
                     "merged_sha256": meta["generations_merged_sha256"], "finalized_at": meta.get("finalized_at"),
                     "source_export": meta.get("source_export"), "complete_items_final": meta.get("complete_items")})
    if not rows:
        raise FileNotFoundError(f"no finalized translator datasets under {generated}")
    return pd.DataFrame(rows)


def translator_order(names) -> list:
    pref = list(research_config().get("display_names", {}))
    return sorted(set(names), key=lambda n: (pref.index(n) if n in pref else len(pref), n))


def display_name(t: str) -> str:
    return research_config().get("display_names", {}).get(t, t)


def _weights_ids(source_export: str | None) -> list:
    """Weights fingerprints recorded at model load by the runs that produced the dataset (None for legacy runs)."""
    if not source_export:
        return []
    runs = paths.SAM_ROOT / source_export / "runs.parquet"
    if not runs.exists():
        return []
    r = pd.read_parquet(runs)
    if "load_report" not in r:
        return []
    out = {json.loads(x).get("weights_id") for x in r.load_report if isinstance(x, str) and x not in ("null", "")}
    return sorted(x for x in out if x)


def load_translations(found: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for f in found.itertuples(index=False):
        p = Path(f.final_dir) / "generations_merged.parquet"
        actual = sha256_file(p)
        if actual != f.merged_sha256:
            raise ValueError(f"{p}: sha256 {actual} != FINAL.json {f.merged_sha256}")
        d = pd.read_parquet(p)
        d = d[(d.translator == f.translator) & (d.dataset == f.dataset)].copy()
        if d.duplicated(["item_id", "output_stage"]).any():
            raise ValueError(f"{f.translator}: duplicate (item, stage) records in its final dataset")
        d["final_sha256"] = f.merged_sha256
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def build(manifest_path: Path = paths.STAGED / "t0_manifest.parquet", generated: Path = paths.GENERATED):
    """Returns (texts, chains, info). `texts` has one row per item for T0 plus a full stage grid per translator."""
    manifest = pd.read_parquet(manifest_path)
    found = discover(generated)
    rec = load_translations(found)
    manifest = manifest[manifest.dataset.isin(found.dataset)]
    t0 = manifest[ID_COLS + ["t0_text", "t0_sha256", "shared_prefix_words_with_human"]].copy()
    bad = t0[t0.t0_text.map(sha256_text) != t0.t0_sha256]
    if len(bad):
        raise ValueError(f"{len(bad)} T0 texts do not match their manifest sha256")

    base = t0.rename(columns={"t0_text": "text", "t0_sha256": "text_sha256"}).assign(
        translator=ORIGINAL, stage="T0", round=0, language="en", status="valid", validation_reason=None,
        validation_warnings=None, input_sha256=None, model_checkpoint=None, generation_config_id=None, run_id=None,
        record_id=None, final_sha256=None)

    names = translator_order(found.translator)
    grid = pd.MultiIndex.from_product([manifest.item_id, names, STAGES], names=["item_id", "translator", "stage"])
    g = pd.DataFrame(index=grid).reset_index()
    keep = ["item_id", "translator", "output_stage", "output_text", "output_sha256", "validation_status",
            "validation_reason", "validation_warnings", "input_sha256", "model_checkpoint", "generation_config_id",
            "run_id", "record_id", "final_sha256", "output_language"]
    g = g.merge(rec[keep].rename(columns={"output_stage": "stage", "output_text": "text",
                                          "output_sha256": "text_sha256", "validation_status": "status",
                                          "output_language": "language"}),
                on=["item_id", "translator", "stage"], how="left")
    g = g.merge(t0[ID_COLS + ["shared_prefix_words_with_human"]], on="item_id", how="left")
    g["round"] = g.stage.str[1].astype(int)
    g["language"] = g.language.fillna(g.stage.map(lambda s: "es" if s[0] == "S" else "en"))
    # A missing record is "blocked" when an earlier stage of the same chain is not valid, else "missing".
    g = g.sort_values(["translator", "item_id"]).reset_index(drop=True)
    g["_order"] = g.stage.map(STAGES.index)
    g = g.sort_values(["translator", "item_id", "_order"])
    prior_ok = (g.status.eq("valid").groupby([g.translator, g.item_id]).cumprod()
                .groupby([g.translator, g.item_id]).shift(1, fill_value=1).astype(bool))
    g["status"] = np.where(g.status.notna(), g.status, np.where(prior_ok, "missing", "blocked"))
    g = g.drop(columns="_order")

    texts = pd.concat([base, g], ignore_index=True)
    texts["valid"] = texts.status.eq("valid")
    texts["text_id"] = texts.item_id + "|" + texts.translator + "|" + texts.stage
    texts["words"] = texts.text.fillna("").str.split().str.len().where(texts.text.notna())
    texts["chars"] = texts.text.str.len()
    stage_rank = {s: i for i, s in enumerate(["T0"] + STAGES)}
    texts["_s"] = texts.stage.map(stage_rank)
    texts["_t"] = texts.translator.map({ORIGINAL: -1, **{n: i for i, n in enumerate(names)}})
    texts = texts.sort_values(["item_id", "_t", "_s"]).drop(columns=["_s", "_t"]).reset_index(drop=True)
    if texts.text_id.duplicated().any():
        raise ValueError("duplicate text_id")

    lineage = check_lineage(texts)
    chains = chain_table(texts, names)
    info = {"translators": names, "found": found, "lineage": lineage,
            "weights_ids": {f.translator: _weights_ids(f.source_export) for f in found.itertuples()}}
    return texts, chains, info


def check_lineage(texts: pd.DataFrame) -> dict:
    """Every valid S1 must read its item's T0; every later valid stage must read the same chain's previous stage."""
    t = texts.set_index(["item_id", "translator", "stage"])
    t0 = texts[texts.stage == "T0"].set_index("item_id").text_sha256
    failures = []
    for r in texts[(texts.translator != ORIGINAL) & texts.valid].itertuples(index=False):
        i = STAGES.index(r.stage)
        want = t0[r.item_id] if i == 0 else t.loc[(r.item_id, r.translator, STAGES[i - 1]), "text_sha256"]
        if r.input_sha256 != want:
            failures.append({"item_id": r.item_id, "translator": r.translator, "stage": r.stage})
    checked = int(((texts.translator != ORIGINAL) & texts.valid).sum())
    configs = (texts[texts.translator != ORIGINAL].dropna(subset=["generation_config_id"])
               .groupby("translator").generation_config_id.unique().map(list).to_dict())
    if failures:
        raise ValueError(f"lineage broken for {len(failures)} records, e.g. {failures[:3]}")
    return {"valid_records_checked": checked, "failures": failures, "generation_config_ids": configs}


def chain_table(texts: pd.DataFrame, names: list) -> pd.DataFrame:
    """One row per (item, translator): status per stage, complete (S1..T3 all valid); cohort flags."""
    tr = texts[texts.translator != ORIGINAL]
    st = tr.pivot(index=["item_id", "translator"], columns="stage", values="status")[STAGES].reset_index()
    st["complete"] = (st[STAGES] == "valid").all(axis=1)
    st["first_failure"] = st[STAGES].apply(lambda r: next((s for s in STAGES if r[s] != "valid"), None), axis=1)
    ids = texts[texts.stage == "T0"][["item_id", "article_id", "group_id", "source_label", "is_human"]]
    st = st.merge(ids, on="item_id", how="left")
    matched = st.groupby("item_id").complete.all()
    st["matched"] = st.item_id.map(matched)
    return st[["item_id", "article_id", "group_id", "source_label", "is_human", "translator", *STAGES,
               "complete", "first_failure", "matched"]]


def cohort_table(chains: pd.DataFrame) -> pd.DataFrame:
    """One row per item: complete flag for each translator and the matched (complete for every translator) flag."""
    w = chains.pivot(index="item_id", columns="translator", values="complete")
    w.columns = [f"complete__{c}" for c in w.columns]
    ids = chains.drop_duplicates("item_id").set_index("item_id")[["article_id", "group_id", "source_label"]]
    out = ids.join(w)
    out["matched"] = w.all(axis=1)
    return out.reset_index()


def coverage_tables(texts: pd.DataFrame) -> tuple:
    tr = texts[texts.translator != ORIGINAL]
    by_stage = (tr.groupby(["translator", "stage", "status"]).size().unstack("status", fill_value=0)
                .reindex(columns=["valid", "invalid", "blocked", "missing"], fill_value=0).reset_index())
    by_stage["stage"] = pd.Categorical(by_stage.stage, STAGES, ordered=True)
    by_stage = by_stage.sort_values(["translator", "stage"])
    by_source = (tr[tr.language == "en"].groupby(["translator", "source_label", "stage"]).valid.agg(["sum", "size"])
                 .rename(columns={"sum": "valid", "size": "items"}).reset_index())
    by_source["missing_or_invalid"] = by_source["items"] - by_source["valid"]
    return by_stage, by_source


def write(texts: pd.DataFrame, chains: pd.DataFrame, info: dict) -> dict:
    """Persist the local text table and the committed text-free coverage tables."""
    texts.to_parquet(paths.local("texts.parquet"), index=False)
    out = paths.results("data_coverage")
    chains.to_csv(out / "chains.csv", index=False)
    cohort = cohort_table(chains)
    cohort.to_csv(out / "cohort.csv", index=False)
    by_stage, by_source = coverage_tables(texts)
    by_stage.to_csv(out / "coverage_by_stage.csv", index=False)
    by_source.to_csv(out / "coverage_by_source.csv", index=False)
    found = info["found"]
    tr_rows = []
    for f in found.itertuples(index=False):
        part = texts[texts.translator == f.translator]
        tr_rows.append({"translator": f.translator, "display_name": display_name(f.translator),
                        "dataset": f.dataset, "model_checkpoint": part.model_checkpoint.dropna().unique().tolist(),
                        "generation_config_id": part.generation_config_id.dropna().unique().tolist(),
                        "weights_ids": info["weights_ids"].get(f.translator, []),
                        "final_dir": str(Path(f.final_dir).relative_to(paths.SAM_ROOT)),
                        "generations_merged_sha256": f.merged_sha256, "finalized_at": f.finalized_at,
                        "complete_chains": int(chains[chains.translator == f.translator].complete.sum()),
                        "items": int(chains[chains.translator == f.translator].item_id.nunique())})
    pd.DataFrame(tr_rows).to_csv(out / "translators.csv", index=False)
    texts_sha = sha256_file(paths.local("texts.parquet"))
    summary = {"items": int(texts[texts.stage == "T0"].item_id.nunique()),
               "articles": int(texts[texts.stage == "T0"].article_id.nunique()),
               "source_labels": sorted(texts.source_label.unique().tolist()),
               "translators": info["translators"], "matched_items": int(cohort.matched.sum()),
               "complete_chains": {t: int(chains[chains.translator == t].complete.sum()) for t in info["translators"]},
               "texts_parquet_sha256": texts_sha, "lineage": info["lineage"]}
    (out / "lineage_report.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def load_texts() -> pd.DataFrame:
    p = paths.local("texts.parquet")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run Sam/scripts/research/build_corpus.py first")
    return pd.read_parquet(p)


def english(texts: pd.DataFrame, valid_only: bool = True) -> pd.DataFrame:
    e = texts[texts.language == "en"]
    return e[e.valid] if valid_only else e
