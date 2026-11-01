"""Long-form generation table: one row per translation operation (one direction of one round trip).

Stages: T0 (original English) -> S1 -> T1 -> S2 -> T2 -> S3 -> T3. S = Spanish, T = English.
A pathway is (experiment_id, translator, prompt_variant). Rows only ever chain within their own pathway.

Composite key: (experiment_id, translator, prompt_variant, item_id, output_stage); record_id is its hash.
Chain rules, checked on every write:
  * the row's input_stage is the stage immediately before its output_stage;
  * input_text equals the T0 text (for S1) or the same pathway's predecessor output, byte for byte;
  * a valid row never has an invalid predecessor;
  * a NEW row may only be created from a valid predecessor (never feed an invalid generation forward);
  * one model_checkpoint per translator and one generation_config_id per pathway (no config mixing).
"""
from __future__ import annotations

import pandas as pd

from .files import sha256_text

STAGES = ["T0", "S1", "T1", "S2", "T2", "S3", "T3"]
GENERATED_STAGES = STAGES[1:]
LANGUAGE = {"T": "en", "S": "es"}
STATUSES = {"valid", "invalid"}
PATHWAY = ["experiment_id", "translator", "prompt_variant"]
KEY = PATHWAY + ["item_id", "output_stage"]

GENERATION_COLUMNS = [
    "record_id", "experiment_id", "dataset", "item_id", "article_id", "source_label",
    "translator", "prompt_variant", "model_checkpoint", "model_revision",
    "round_trip", "direction", "input_stage", "output_stage", "input_language", "output_language",
    "input_text", "output_text", "input_sha256", "output_sha256",
    "validation_status", "validation_reason", "validation_warnings",
    "generation_config_id", "run_id", "provenance", "legacy_fields", "generation_meta",
]
# Columns added after Stage 2. Tables written before they existed are read with them set to None.
ADDED_COLUMNS = ["generation_meta"]
REQUIRED = [c for c in GENERATION_COLUMNS if c not in {"model_revision", "output_text", "output_sha256",
                                                         "validation_reason", "validation_warnings",
                                                         "legacy_fields", "generation_meta"}]


class SchemaError(ValueError):
    pass


class DuplicateRecordError(SchemaError):
    pass


class ConflictError(SchemaError):
    pass


class ChainError(SchemaError):
    pass


def stage_info(output_stage: str) -> dict:
    """Everything implied by an output stage: round trip, direction, input stage and languages."""
    if output_stage not in GENERATED_STAGES:
        raise SchemaError(f"not a generated stage: {output_stage!r}")
    i = STAGES.index(output_stage)
    input_stage = STAGES[i - 1]
    src, tgt = LANGUAGE[input_stage[0]], LANGUAGE[output_stage[0]]
    return {"round_trip": int(output_stage[1]), "direction": f"{src}->{tgt}", "input_stage": input_stage,
            "input_language": src, "output_language": tgt}


def record_id(row) -> str:
    return sha256_text("|".join(str(row[k]) for k in KEY))[:16]


def empty_table() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="int64" if c == "round_trip" else "object") for c in GENERATION_COLUMNS})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Add post-Stage-2 columns (as None) to an older table and put columns in schema order."""
    df = df.copy()
    for c in ADDED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df), index=df.index, dtype="object")
    return df


def _norm(v):
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


def check_rows(new: pd.DataFrame, t0: pd.DataFrame) -> None:
    """Row-level checks that need no other generation rows."""
    missing = [c for c in GENERATION_COLUMNS if c not in new.columns and c not in ADDED_COLUMNS]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    nulls = {c: int(new[c].isna().sum()) for c in REQUIRED if new[c].isna().any()}
    if nulls:
        raise SchemaError(f"null values in required columns: {nulls}")
    for r in new.itertuples(index=False):
        info = stage_info(r.output_stage)
        for k, v in info.items():
            if getattr(r, k) != v:
                raise SchemaError(f"{r.item_id} {r.output_stage}: {k}={getattr(r, k)!r}, expected {v!r}")
        if r.validation_status not in STATUSES:
            raise SchemaError(f"{r.item_id} {r.output_stage}: bad validation_status {r.validation_status!r}")
        if r.validation_status == "invalid" and not _norm(r.validation_reason):
            raise SchemaError(f"{r.item_id} {r.output_stage}: invalid record without validation_reason")
        if r.input_sha256 != sha256_text(r.input_text):
            raise SchemaError(f"{r.item_id} {r.output_stage}: input_sha256 does not match input_text")
        if _norm(r.output_text) is not None and r.output_sha256 != sha256_text(r.output_text):
            raise SchemaError(f"{r.item_id} {r.output_stage}: output_sha256 does not match output_text")
        if r.record_id != record_id(r._asdict()):
            raise SchemaError(f"{r.item_id} {r.output_stage}: record_id is not the hash of the composite key")
    labels = t0.set_index("item_id")[["dataset", "article_id", "source_label"]]
    unknown = set(new.item_id) - set(labels.index)
    if unknown:
        raise SchemaError(f"item_ids not in the T0 manifest: {sorted(unknown)[:5]}")
    got = new.set_index("item_id")[["dataset", "article_id", "source_label"]]
    bad = (got != labels.loc[got.index]).any(axis=1)
    if bad.any():
        raise SchemaError(f"dataset/article_id/source_label differ from the manifest for {sorted(got.index[bad])[:5]}")


def chain_violations(gen: pd.DataFrame, t0: pd.DataFrame) -> list:
    """All chain-rule violations in a full table (empty list = consistent)."""
    out = []
    if gen.duplicated(KEY).any():
        out.append(f"duplicate composite keys: {gen.loc[gen.duplicated(KEY), KEY].values.tolist()[:3]}")
    models = gen.groupby(["experiment_id", "translator"]).model_checkpoint.nunique()
    for k in models[models > 1].index:
        out.append(f"translator {k} has more than one model_checkpoint")
    configs = gen.groupby(PATHWAY).generation_config_id.nunique()
    for k in configs[configs > 1].index:
        out.append(f"pathway {k} mixes generation_config_ids")
    t0_text = t0.set_index("item_id").t0_text
    by_key = gen.set_index(KEY)
    for r in gen.itertuples(index=False):
        if r.input_stage == "T0":
            pred_text, pred_status = t0_text.get(r.item_id), "valid"
        else:
            k = (r.experiment_id, r.translator, r.prompt_variant, r.item_id, r.input_stage)
            if k not in by_key.index:
                out.append(f"{r.item_id} {r.output_stage} ({r.translator}/{r.prompt_variant}): predecessor "
                           f"{r.input_stage} missing in this pathway")
                continue
            p = by_key.loc[k]
            pred_text, pred_status = p.output_text, p.validation_status
        if r.input_text != pred_text:
            out.append(f"{r.item_id} {r.output_stage} ({r.translator}/{r.prompt_variant}): input_text is not the "
                       f"pathway's {r.input_stage} text")
        if pred_status != "valid" and r.validation_status == "valid":
            out.append(f"{r.item_id} {r.output_stage}: valid record derived from invalid {r.input_stage}")
    return out


def upsert(existing: pd.DataFrame, new: pd.DataFrame, t0: pd.DataFrame,
           allow_invalid_predecessor: bool = False):
    """Insert new records idempotently. Returns (table, stats).
    - a composite key repeated inside `new` raises DuplicateRecordError;
    - a key already present with identical content is skipped; with different content raises ConflictError;
    - inserted rows must chain onto a valid predecessor unless allow_invalid_predecessor (historical imports
      whose quality flag covers a whole round trip; see legacy_madlad.py)."""
    new = conform(new)
    check_rows(new, t0)
    new = new[GENERATION_COLUMNS].reset_index(drop=True)
    if new.duplicated(KEY).any():
        raise DuplicateRecordError(f"composite key repeated in batch: {new.loc[new.duplicated(KEY), KEY].values.tolist()[:3]}")
    existing = conform(existing)[GENERATION_COLUMNS] if len(existing) else empty_table()
    old = existing.set_index("record_id")
    insert, unchanged = [], 0
    for i, r in new.iterrows():
        if r.record_id in old.index:
            before = old.loc[r.record_id]
            diff = [c for c in GENERATION_COLUMNS[1:] if _norm(before[c]) != _norm(r[c])]
            if diff:
                raise ConflictError(f"{r.item_id} {r.output_stage}: existing record differs in {diff}")
            unchanged += 1
        else:
            insert.append(i)
    ins = new.loc[insert]
    if not len(ins):
        return existing.reset_index(drop=True), {"inserted": 0, "unchanged": unchanged}
    table = pd.concat([existing, ins], ignore_index=True) if len(existing) else ins.reset_index(drop=True)
    problems = chain_violations(table, t0)
    if not allow_invalid_predecessor:
        valid = table.set_index(KEY).validation_status
        for r in ins.itertuples(index=False):
            if r.input_stage != "T0":
                k = (r.experiment_id, r.translator, r.prompt_variant, r.item_id, r.input_stage)
                if valid.get(k) != "valid":
                    problems.append(f"{r.item_id} {r.output_stage}: new record built from a non-valid {r.input_stage}")
    if problems:
        raise ChainError("; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
    order = {s: i for i, s in enumerate(STAGES)}
    table = (table.assign(_s=table.output_stage.map(order)).sort_values(PATHWAY + ["item_id", "_s"])
             .drop(columns="_s").reset_index(drop=True))
    table["round_trip"] = table.round_trip.astype("int64")
    return table, {"inserted": len(ins), "unchanged": unchanged}


def eligible_inputs(gen: pd.DataFrame, t0: pd.DataFrame, experiment_id: str, translator: str,
                    prompt_variant: str, output_stage: str) -> pd.DataFrame:
    """Items that may be generated next at `output_stage`: predecessor valid, stage not yet generated."""
    input_stage = stage_info(output_stage)["input_stage"]
    p = gen[(gen.experiment_id == experiment_id) & (gen.translator == translator) &
            (gen.prompt_variant == prompt_variant)]
    done = set(p.item_id[p.output_stage == output_stage])
    if input_stage == "T0":
        src = t0[["item_id", "t0_text"]].rename(columns={"t0_text": "input_text"})
    else:
        src = p[(p.output_stage == input_stage) & (p.validation_status == "valid")][["item_id", "output_text"]]
        src = src.rename(columns={"output_text": "input_text"})
    return src[~src.item_id.isin(done)].assign(input_stage=input_stage).reset_index(drop=True)


def stage_status(gen: pd.DataFrame, t0: pd.DataFrame) -> pd.DataFrame:
    """One row per pathway x item x generated stage:
    valid / invalid; not_attempted (input valid, ready to generate); pending (input not generated yet);
    blocked (an upstream record is invalid, so this stage must never be generated)."""
    rows = []
    status = gen.set_index(KEY).validation_status
    for pw in gen[PATHWAY].drop_duplicates().itertuples(index=False):
        for item in t0.item_id:
            prev = "valid"
            for stage in GENERATED_STAGES:
                s = status.get((*pw, item, stage))
                if s is None:
                    s = {"valid": "not_attempted", "invalid": "blocked", "blocked": "blocked"}.get(prev, "pending")
                rows.append({**pw._asdict(), "item_id": item, "output_stage": stage, "status": s})
                prev = s
    return pd.DataFrame(rows)


def attrition(gen: pd.DataFrame, t0: pd.DataFrame) -> pd.DataFrame:
    """Counts per pathway and stage: how many items are valid, invalid, blocked or still to generate."""
    st = stage_status(gen, t0)
    t = st.pivot_table(index=PATHWAY + ["output_stage"], columns="status", values="item_id", aggfunc="count",
                       fill_value=0).reset_index()
    for c in ["valid", "invalid", "blocked", "not_attempted", "pending"]:
        if c not in t:
            t[c] = 0
    t["items"] = len(t0)
    t["_s"] = t.output_stage.map({s: i for i, s in enumerate(STAGES)})
    t = t.sort_values(PATHWAY + ["_s"]).drop(columns="_s")
    t.columns.name = None
    return t[PATHWAY + ["output_stage", "items", "valid", "invalid", "blocked", "not_attempted",
                        "pending"]].reset_index(drop=True)


def complete_cases(gen: pd.DataFrame, t0: pd.DataFrame, through_stage: str, pathways=None) -> list:
    """item_ids valid at every stage up to `through_stage` in every pathway (or the given ones)."""
    st = stage_status(gen, t0)
    stages = GENERATED_STAGES[:GENERATED_STAGES.index(through_stage) + 1]
    st = st[st.output_stage.isin(stages)]
    if pathways is not None:
        keep = pd.DataFrame(pathways, columns=PATHWAY)
        st = st.merge(keep, on=PATHWAY)
    ok = st.groupby("item_id").status.apply(lambda s: (s == "valid").all())
    return [i for i in t0.item_id if ok.get(i, False)]
