"""Resumable round-trip runner: one pathway, one translation operation at a time.

Inputs (read-only): the T0 manifest and one or more base generation tables (the Stage 2 staged table,
plus the merged export of any earlier session). Outputs, all in `work_dir` and written atomically:

  generations_delta.parquet  completed operations from this work dir (valid AND invalid records)
  failures.parquet           attempts that produced no record (exceptions, CUDA OOM): retried later
  runs.parquet               one row per run: runtime environment, settings, outcome
  generations_merged.parquet + RESUME.json   written by export(): the input for the next session
  validation_report.csv, attrition.csv, new_translations.csv   written by export(): for inspection

Order of work is stage-major: every selected item finishes S_r before any starts T_r. For each stage the
next inputs come from schema.eligible_inputs: the pathway's own valid predecessor output, never T0
after round 1, never another pathway. Records are buffered in memory and flushed every
`checkpoint_every` operations, at the end of each stage, and on KeyboardInterrupt. A hard crash
loses at most the unflushed buffer. Those operations are simply redone (greedy decoding is
deterministic). An operation in progress never produces a record. With `export_every`, the full export
(the next session's input) is also rewritten every N operations and at each stage end.
"""
from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from . import schema
from .checks import check_step, roundtrip_content_warnings, roundtrip_warnings
from .files import atomic_write_parquet, atomic_write_text, sha256_file, sha256_text
from .progress import Progress, words
from .translators import generation_config

DELTA, FAILURES, RUNS, MERGED, RESUME = ("generations_delta.parquet", "failures.parquet", "runs.parquet",
                                         "generations_merged.parquet", "RESUME.json")
REPORT, ATTRITION, NEW_TEXTS = "validation_report.csv", "attrition.csv", "new_translations.csv"
FAILURE_COLUMNS = ["attempt_id", "experiment_id", "translator", "prompt_variant", "item_id", "output_stage",
                   "input_sha256", "attempt", "error_type", "error_message", "run_id", "generation_config_id",
                   "created_at"]


class ConfigMismatchError(RuntimeError):
    pass


class MockDataError(RuntimeError):
    """Mock (fake) translations found where a real model run would build on them."""


MOCK_PREFIX = "MOCK-"
# Codes that mean the engine produced no usable translation at all (as opposed to a wrong-language or a
# length warning). A run where most recent operations end like this has a broken engine, not bad inputs.
DEGENERATE = ("engine:generation_cap", "check:repetition", "engine:empty_segment", "check:blank",
              "check:not_target_language")


class QuarantineError(RuntimeError):
    """A base table contains records of a run declared unusable (experiment.yaml: quarantined_runs)."""


@dataclass
class RunSettings:
    translator: str
    prompt_variant: str
    rounds: tuple = (1, 3)               # first and last round trip to work on
    stop_after: Optional[str] = None     # e.g. "S2": do not go beyond this stage in this run
    dataset: Optional[str] = None
    article_ids: Optional[list] = None
    item_ids: Optional[list] = None
    max_items: Optional[int] = None      # first N selected items in manifest order
    checkpoint_every: int = 5
    max_attempts: int = 2                # failed attempts per operation before it is skipped
    max_operations: Optional[int] = None
    time_limit_minutes: Optional[float] = None
    # Fail-fast quality gate: stop when, among the last `gate_window` completed operations (at least
    # `gate_min_ops`), the share with a DEGENERATE code reaches `gate_max_share`. None disables it.
    gate_min_ops: Optional[int] = 3
    gate_window: int = 10
    gate_max_share: float = 0.5
    # Rolling export: every `export_every` operations (and at each stage end) also write the full export
    # (generations_merged.parquet + RESUME.json) to `export_dir`, so a session killed without reaching its
    # export cell still leaves a resumable export. None = export only when asked.
    export_every: Optional[int] = None
    export_dir: Optional[str] = None


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stages_for(rounds: tuple, stop_after: Optional[str] = None) -> list:
    first, last = rounds
    out = [s for s in schema.GENERATED_STAGES if first <= int(s[1]) <= last]
    if stop_after:
        out = out[:out.index(stop_after) + 1]
    return out


def select_items(manifest: pd.DataFrame, s: RunSettings) -> pd.DataFrame:
    m = manifest.sort_values("sample_index") if "sample_index" in manifest else manifest
    if s.dataset:
        m = m[m.dataset == s.dataset]
    if s.article_ids:
        m = m[m.article_id.isin(s.article_ids)]
    if s.item_ids:
        m = m[m.item_id.isin(s.item_ids)]
    return (m.head(s.max_items) if s.max_items else m).reset_index(drop=True)


def load_tables(tables: list, manifest: pd.DataFrame) -> pd.DataFrame:
    """Union of generation tables (paths or DataFrames), with duplicate/conflict/chain checks. Records were
    validated when created, so historical round-trip-level flags are accepted here."""
    gen = schema.empty_table()
    for t in tables:
        df = pd.read_parquet(t) if isinstance(t, (str, Path)) else t
        gen, _ = schema.upsert(gen, df, manifest, allow_invalid_predecessor=True)
    return gen


def read_failures(paths: list) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths if Path(p).exists()]
    if not frames:
        return pd.DataFrame(columns=FAILURE_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates("attempt_id")[FAILURE_COLUMNS]


def runtime_info(entry: dict) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["torch", "transformers", "accelerate", "sentencepiece"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        info["cuda"], info["gpus"] = False, []
    pins = (entry.get("settings") or {}).get("libraries") or {}
    info["library_mismatch"] = {k: {"config": str(v), "runtime": info.get(k)} for k, v in pins.items()
                                if info.get(k) is not None and info.get(k) != str(v)}
    return info


class Runner:
    def __init__(self, settings: RunSettings, translator, manifest: pd.DataFrame, base: pd.DataFrame,
                 work_dir, exp_cfg: dict, tcfg: dict, run_id: Optional[str] = None,
                 prior_failure_paths: tuple = (), log=print, progress: Optional[Progress] = None):
        self.s, self.tr, self.manifest, self.cfg, self.tcfg, self.log = settings, translator, manifest, exp_cfg, tcfg, log
        self.progress = progress or Progress()          # silent unless given an emitter (notebook, CLI)
        self.exp = exp_cfg["experiment_id"]
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"{settings.translator}-{settings.prompt_variant}-{now().replace(':', '')}-{uuid.uuid4().hex[:6]}"
        self.is_mock = bool(getattr(translator, "is_mock", False))
        if self.is_mock and not self.run_id.startswith(MOCK_PREFIX):
            self.run_id = MOCK_PREFIX + self.run_id
        self._check_identity()
        delta_path = self.work / DELTA
        self.delta = schema.conform(pd.read_parquet(delta_path)) if delta_path.exists() else schema.empty_table()
        bad_runs = set(exp_cfg.get("quarantined_runs") or [])
        for name, t in [("base", base), ("work-dir delta", self.delta)]:
            hit = sorted(set(t.run_id.astype(str)) & bad_runs) if len(t) else []
            if hit:
                raise QuarantineError(f"{name} contains records of quarantined run(s) {hit} (experiment.yaml "
                                      "quarantined_runs); build on the Stage 2 table or a clean export instead")
        if not self.is_mock:
            for name, t in [("base", base), ("work-dir delta", self.delta)]:
                if len(t) and t.run_id.astype(str).str.startswith(MOCK_PREFIX).any():
                    raise MockDataError(f"{name} contains mock records (run_id {MOCK_PREFIX}...); "
                                        "real runs never build on fake translations")
        self.table, _ = schema.upsert(base, self.delta, manifest)
        self._check_pathway_config()
        self.failures = read_failures([*prior_failure_paths, self.work / FAILURES])
        self.items = select_items(manifest, settings)
        self.stages = stages_for(settings.rounds, settings.stop_after)
        self.pending, self.new_failures = [], []
        self.recent: list = []           # degenerate? for this run's completed operations (quality gate)

    # -- consistency ------------------------------------------------------------------------------
    def _check_identity(self) -> None:
        s, t = self.s, self.tr
        if (t.translator, t.prompt_variant) != (s.translator, s.prompt_variant):
            raise ConfigMismatchError(f"adapter is {t.translator}/{t.prompt_variant}, run asks for "
                                      f"{s.translator}/{s.prompt_variant}")
        expected, _ = generation_config(self.tcfg, s.translator, s.prompt_variant)
        if t.generation_config_id != expected:
            raise ConfigMismatchError(f"adapter config {t.generation_config_id} != translators.yaml {expected}")

    def _pathway(self, table=None) -> pd.DataFrame:
        t = self.table if table is None else table
        return t[(t.experiment_id == self.exp) & (t.translator == self.s.translator) &
                 (t.prompt_variant == self.s.prompt_variant)]

    def _check_pathway_config(self) -> None:
        p = self._pathway()
        ids = set(p.generation_config_id)
        if ids - {self.tr.generation_config_id}:
            raise ConfigMismatchError(f"existing {self.s.translator}/{self.s.prompt_variant} records use config "
                                      f"{sorted(ids)}, adapter uses {self.tr.generation_config_id}")
        models = set(self.table[self.table.translator == self.s.translator].model_checkpoint)
        if models - {self.tr.model_checkpoint}:
            raise ConfigMismatchError(f"existing {self.s.translator} records use {sorted(models)}")

    # -- planning ---------------------------------------------------------------------------------
    def _exhausted(self) -> set:
        if not len(self.failures) and not self.new_failures:
            return set()
        f = pd.concat([self.failures, pd.DataFrame(self.new_failures, columns=FAILURE_COLUMNS)])
        f = f[(f.translator == self.s.translator) & (f.prompt_variant == self.s.prompt_variant)]
        n = f.groupby(["item_id", "output_stage", "input_sha256"]).size()
        return {k for k, v in n.items() if v >= self.s.max_attempts}

    def next_operations(self, stage: str) -> pd.DataFrame:
        e = schema.eligible_inputs(self.table, self.manifest, self.exp, self.s.translator, self.s.prompt_variant, stage)
        e = e[e.item_id.isin(self.items.item_id)]
        ex = self._exhausted()
        keep = pd.Series([(i, stage, sha256_text(t)) not in ex for i, t in zip(e.item_id, e.input_text)],
                         index=e.index, dtype=bool)
        order = {i: k for k, i in enumerate(self.items.item_id)}
        return e[keep].assign(_o=lambda d: d.item_id.map(order)).sort_values("_o").drop(columns="_o")

    def progress_plan(self) -> tuple:
        """({item: [stages this run expects to perform]}, {item: words of its next input}). An item is planned
        from its first missing stage, if that stage's input exists and is valid and retries are not exhausted."""
        pw = self._pathway()
        recs = {(r.item_id, r.output_stage): r for r in pw.itertuples(index=False)}
        t0 = self.manifest.set_index("item_id").t0_text
        ex = self._exhausted()
        plan, nxt = {}, {}
        for item in self.items.item_id:
            for k, stage in enumerate(self.stages):
                if (item, stage) in recs:
                    continue
                prev = schema.stage_info(stage)["input_stage"]
                if prev == "T0":
                    text = t0[item]
                else:
                    r = recs.get((item, prev))
                    if r is None or r.validation_status != "valid":
                        break
                    text = r.output_text
                if (item, stage, sha256_text(text)) not in ex:
                    plan[item], nxt[item] = self.stages[k:], words(text)
                break
        return plan, nxt

    def plan(self) -> pd.DataFrame:
        """Per stage for the selected items: done, ready now, waiting on an earlier stage, blocked."""
        st = schema.stage_status(self._pathway(), self.manifest) if len(self._pathway()) else None
        rows = []
        for stage in self.stages:
            ready = len(self.next_operations(stage))
            if st is None:
                sub = pd.Series(["not_attempted" if stage == "S1" else "pending"] * len(self.items))
            else:
                sub = st[(st.output_stage == stage) & st.item_id.isin(self.items.item_id)].status
            rows.append({"stage": stage, "items": len(self.items), "valid": int((sub == "valid").sum()),
                         "invalid": int((sub == "invalid").sum()), "ready_now": ready,
                         "waiting": int((sub == "pending").sum()), "blocked": int((sub == "blocked").sum())})
        return pd.DataFrame(rows)

    # -- execution --------------------------------------------------------------------------------
    def _build_row(self, item_id: str, stage: str, input_text: str, res, attempt: int) -> dict:
        info = schema.stage_info(stage)
        t0 = self.manifest.set_index("item_id").loc[item_id]
        block, warn = [], []
        if res.error:
            block.append(f"engine:{res.error}")
        else:
            if res.finish == "length":
                block.append("engine:generation_cap")
            chk = check_step(input_text, res.text, info["output_language"], self.cfg["validation"])
            block += chk["blocking"]
            warn += chk["warnings"]
            if stage.startswith("T"):
                prev_en = self.table.set_index(schema.KEY).loc[
                    (self.exp, self.s.translator, self.s.prompt_variant, item_id, info["input_stage"]), "input_text"]
                warn += roundtrip_warnings(prev_en, res.text)
                warn += roundtrip_content_warnings(prev_en, res.text, self.cfg["validation"])
        out = res.text if not res.error else None
        meta = {**res.meta, "finish": res.finish, "attempt": attempt, "created_at": now(), "engine_error": res.error}
        row = {"experiment_id": self.exp, "dataset": t0.dataset, "item_id": item_id, "article_id": t0.article_id,
               "source_label": t0.source_label, "translator": self.tr.translator,
               "prompt_variant": self.tr.prompt_variant, "model_checkpoint": self.tr.model_checkpoint,
               "model_revision": self.tr.model_revision, **info, "output_stage": stage,
               "input_text": input_text, "output_text": out, "input_sha256": sha256_text(input_text),
               "output_sha256": None if out is None else sha256_text(out),
               "validation_status": "invalid" if block else "valid", "validation_reason": ";".join(block) or None,
               "validation_warnings": ";".join(warn) or None, "generation_config_id": self.tr.generation_config_id,
               "run_id": self.run_id, "provenance": f"generated:{self.run_id}", "legacy_fields": None,
               "generation_meta": json.dumps(meta, sort_keys=True, default=str)}
        row["record_id"] = schema.record_id(row)
        return row

    def _attempt_number(self, item_id: str, stage: str, input_sha: str) -> int:
        f = pd.concat([self.failures, pd.DataFrame(self.new_failures, columns=FAILURE_COLUMNS)])
        f = f[(f.translator == self.s.translator) & (f.prompt_variant == self.s.prompt_variant) &
              (f.item_id == item_id) & (f.output_stage == stage) & (f.input_sha256 == input_sha)]
        return len(f) + 1

    def checkpoint(self) -> None:
        if self.pending:
            new = schema.conform(pd.DataFrame(self.pending))[schema.GENERATION_COLUMNS]
            # Full validation (duplicates, conflicts, chain, config) happens on the combined table; the delta is
            # not a self-contained chain (its S2 rows chain onto base T1 rows), so it is only appended to.
            self.table, _ = schema.upsert(self.table, new, self.manifest)
            self.delta = pd.concat([self.delta, new], ignore_index=True) if len(self.delta) else new
            atomic_write_parquet(self.delta, self.work / DELTA)
            self.pending = []
        if self.new_failures:
            self.failures = pd.concat([self.failures, pd.DataFrame(self.new_failures, columns=FAILURE_COLUMNS)],
                                      ignore_index=True)
            atomic_write_parquet(self.failures, self.work / FAILURES)
            self.new_failures = []

    def _write_run(self, status: str, counts: dict, started: str) -> None:
        path = self.work / RUNS
        runs = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        runs = runs[runs.run_id != self.run_id] if len(runs) else runs
        row = {"run_id": self.run_id, "started_at": started, "updated_at": now(), "status": status,
               **self.tr.identity(), "settings": json.dumps(asdict(self.s), default=str),
               "runtime": json.dumps(runtime_info(self.tr.entry), default=str),
               "allow_download": bool(getattr(self.tr, "allow_download", False)),
               "load_report": json.dumps(getattr(self.tr, "load_report", None), default=str), **counts}
        atomic_write_parquet(pd.concat([runs, pd.DataFrame([row])], ignore_index=True), path)

    def run(self, clock=time.monotonic, unload_after: bool = True) -> dict:
        """Work through the selected stages. The model is unloaded afterwards (also after an interruption or
        error) unless unload_after=False, so a re-run cell never holds two copies in GPU memory."""
        try:
            out = self._run(clock)
        except KeyboardInterrupt:
            self.progress.finish("interrupted")
            raise
        except Exception as e:  # noqa: BLE001
            self.progress.finish("failed", error=f"{type(e).__name__}: {str(e)[:300]}")
            raise
        finally:
            if unload_after:
                self.tr.unload()
        self.progress.finish(out["status"], **{k: v for k, v in out.items() if k != "status"})
        return out

    def _run(self, clock) -> dict:
        started, t_start = now(), clock()
        counts = {"operations": 0, "valid": 0, "invalid": 0, "failed_attempts": 0}
        status, exported_at = "completed", 0
        self._write_run("running", counts, started)
        P = self.progress
        plan, nxt = self.progress_plan()
        P.start({"translator": self.s.translator, "prompt_variant": self.s.prompt_variant,
                 "model": self.tr.model_checkpoint, "dataset": ",".join(sorted(set(self.items.dataset))),
                 "experiment_id": self.exp, "run_id": self.run_id}, plan, nxt)
        P.set_phase("loading_model", note="includes the download on first use")
        t_load, already = time.monotonic(), getattr(self.tr, "model", None) is not None
        try:
            self.tr.load()
            P.loaded(time.monotonic() - t_load, already_loaded=already)
        except BaseException as e:
            self._write_run(f"failed:load:{type(e).__name__}", counts, started)
            raise
        try:
            for stage in self.stages:
                info = schema.stage_info(stage)
                ops = self.next_operations(stage)
                if len(ops):
                    self.log(f"{stage}: {len(ops)} operations")
                for op in ops.itertuples(index=False):
                    if self.s.max_operations is not None and counts["operations"] >= self.s.max_operations:
                        status = "stopped:max_operations"
                        break
                    if self.s.time_limit_minutes and clock() - t_start > self.s.time_limit_minutes * 60:
                        status = "stopped:time_limit"
                        break
                    counts["operations"] += 1
                    attempt = self._attempt_number(op.item_id, stage, sha256_text(op.input_text))
                    P.op_start(op.item_id, stage, op.input_text)
                    t_op = time.monotonic()
                    try:
                        res = self.tr.translate(op.input_text, info["input_language"], info["output_language"])
                    except Exception as e:  # noqa: BLE001  transient: logged, retried later (KeyboardInterrupt passes)
                        P.op_done(op.item_id, stage, "failed", time.monotonic() - t_op, op.input_text, None,
                                  reason=type(e).__name__)
                        counts["failed_attempts"] += 1
                        self.new_failures.append({
                            "attempt_id": sha256_text(f"{self.run_id}|{op.item_id}|{stage}|{attempt}")[:16],
                            "experiment_id": self.exp, "translator": self.s.translator,
                            "prompt_variant": self.s.prompt_variant, "item_id": op.item_id, "output_stage": stage,
                            "input_sha256": sha256_text(op.input_text), "attempt": attempt,
                            "error_type": type(e).__name__, "error_message": str(e)[:500], "run_id": self.run_id,
                            "generation_config_id": self.tr.generation_config_id, "created_at": now()})
                        self.log(f"  {op.item_id} {stage}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]})")
                    else:
                        secs = time.monotonic() - t_op
                        P.set_phase("validating", emit=False)
                        row = self._build_row(op.item_id, stage, op.input_text, res, attempt)
                        counts[row["validation_status"]] += 1
                        P.op_done(op.item_id, stage, row["validation_status"], secs, op.input_text, row["output_text"],
                                  n_segments=(res.meta.get("segmentation") or {}).get("n_segments"),
                                  reason=row["validation_reason"])
                        self.pending.append(row)
                        if row["validation_status"] == "invalid":
                            self.log(f"  {op.item_id} {stage}: INVALID {row['validation_reason']}")
                        self.recent.append(any(c in (row["validation_reason"] or "") for c in DEGENERATE))
                        if self._gate_tripped():
                            status = "stopped:quality_gate"
                            w = self.recent[-self.s.gate_window:]
                            self.log(f"QUALITY GATE: {sum(w)}/{len(w)} recent operations degenerate "
                                     f"({', '.join(DEGENERATE)}); stopping to save GPU time")
                            P.event("quality_gate", degenerate=sum(w), window=len(w))
                    if len(self.pending) + len(self.new_failures) >= self.s.checkpoint_every:
                        self.checkpoint()
                    if self._rolling_export_due(counts["operations"] - exported_at):
                        self.export(self.s.export_dir)
                        exported_at = counts["operations"]
                    if status == "stopped:quality_gate":
                        break
                self.checkpoint()
                if self._rolling_export_due(counts["operations"] - exported_at, stage_end=True):
                    self.export(self.s.export_dir)
                    exported_at = counts["operations"]
                if status != "completed":
                    break
        except KeyboardInterrupt:
            self.checkpoint()
            self._write_run("interrupted", counts, started)
            raise
        self.checkpoint()
        self._write_run(status, counts, started)
        return {"run_id": self.run_id, "status": status, **counts}

    def _rolling_export_due(self, ops_since: int, stage_end: bool = False) -> bool:
        if not (self.s.export_every and self.s.export_dir) or ops_since <= 0:
            return False
        return stage_end or ops_since >= self.s.export_every

    def _gate_tripped(self) -> bool:
        if not self.s.gate_min_ops:
            return False
        w = self.recent[-self.s.gate_window:]
        return len(w) >= self.s.gate_min_ops and sum(w) / len(w) >= self.s.gate_max_share

    # -- export -----------------------------------------------------------------------------------
    def export(self, out_dir=None) -> dict:
        """Everything the next session needs: attach out_dir as a Kaggle dataset and list
        generations_merged.parquet (and failures.parquet) as base inputs."""
        out = Path(out_dir) if out_dir else self.work
        self.progress.set_phase("exporting", export_dir=str(out))
        self.checkpoint()
        atomic_write_parquet(self.table, out / MERGED)
        if out != self.work:
            for name in [DELTA, RUNS]:
                if (self.work / name).exists():
                    atomic_write_parquet(pd.read_parquet(self.work / name), out / name)
        if len(self.failures):            # earlier sessions' failures too, so retry counts carry over
            atomic_write_parquet(self.failures, out / FAILURES)
        att = schema.attrition(self._pathway(), self.manifest) if len(self._pathway()) else pd.DataFrame()
        atomic_write_text(att.to_csv(index=False), out / ATTRITION)
        atomic_write_text(validation_report(self.delta).to_csv(index=False), out / REPORT)
        texts = self.delta[["item_id", "output_stage", "validation_status", "validation_reason", "input_text",
                            "output_text", "run_id"]] if len(self.delta) else pd.DataFrame()
        atomic_write_text(texts.to_csv(index=False), out / NEW_TEXTS)
        files = {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != RESUME}
        resume = {"exported_at": now(), "experiment_id": self.exp, "pathway": [self.s.translator, self.s.prompt_variant],
                  "generation_config_id": self.tr.generation_config_id, "records_total": len(self.table),
                  "records_this_work_dir": len(self.delta), "failed_attempts": len(self.failures),
                  "attrition": att.drop(columns=["experiment_id"], errors="ignore").to_dict("records"),
                  "files": files,
                  "next_session": "save the notebook version, add its output as an input, and list this "
                                  "folder in PREVIOUS_EXPORTS (CLI: --base generations_merged.parquet "
                                  "--prior-failures failures.parquet)"}
        atomic_write_text(json.dumps(resume, indent=1, default=str), out / RESUME)
        self.progress.set_phase("exported", emit=False)
        self.progress.event("export_done", export_dir=str(out), files=len(files))
        return resume


def validation_report(gen: pd.DataFrame) -> pd.DataFrame:
    """Text-free, one row per record: status, reasons, warnings, lengths and engine metadata."""
    cols = ["item_id", "article_id", "source_label", "output_stage", "validation_status", "validation_reason",
            "validation_warnings", "input_words", "output_words", "length_ratio", "finish", "n_segments",
            "segment_hit_eos", "oom_retries", "seconds", "run_id"]
    rows = []
    for r in gen.itertuples(index=False):
        meta = json.loads(r.generation_meta) if isinstance(r.generation_meta, str) else {}
        n_in = len(str(r.input_text).split())
        n_out = len(str(r.output_text).split()) if isinstance(r.output_text, str) else 0
        rows.append({"item_id": r.item_id, "article_id": r.article_id, "source_label": r.source_label,
                     "output_stage": r.output_stage, "validation_status": r.validation_status,
                     "validation_reason": r.validation_reason, "validation_warnings": r.validation_warnings,
                     "input_words": n_in, "output_words": n_out, "length_ratio": round(n_out / max(1, n_in), 3),
                     "finish": meta.get("finish"), "n_segments": (meta.get("segmentation") or {}).get("n_segments"),
                     "segment_hit_eos": meta.get("segment_hit_eos"), "oom_retries": meta.get("oom_retries"),
                     "seconds": meta.get("seconds"), "run_id": r.run_id})
    return pd.DataFrame(rows, columns=cols)
