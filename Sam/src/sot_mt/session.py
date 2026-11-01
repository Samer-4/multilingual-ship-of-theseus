"""Setup shared by notebooks/00_generate_translations.ipynb and scripts/run_translations.py."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import paths, runner, schema
from .files import load_yaml, sha256_file
from .translators import MockTranslator, build_translator


def verify_staged(staged: Path) -> dict:
    """Check t0_manifest.parquet / generations.parquet against STAGED_FILES.json (written by Stage 2)."""
    info = json.loads((Path(staged) / "STAGED_FILES.json").read_text())
    for name, sha in info["files"].items():
        got = sha256_file(Path(staged) / name)
        if got != sha:
            raise RuntimeError(f"{name}: sha256 {got[:12]} != STAGED_FILES.json {sha[:12]}; staged inputs changed")
    return info


def load_inputs(staged=None, extra_tables=()) -> tuple:
    """(manifest, base generation table). Base = staged Stage 2 table + any earlier session exports."""
    staged = Path(staged) if staged else paths.staged_dir()
    verify_staged(staged)
    manifest = pd.read_parquet(staged / "t0_manifest.parquet")
    base = runner.load_tables([staged / "generations.parquet", *extra_tables], manifest)
    return manifest, base


def translators_overview(tcfg=None) -> pd.DataFrame:
    tcfg = tcfg or load_yaml("translators.yaml")
    rows = []
    for name, e in tcfg["translators"].items():
        for pv in e["prompt_variants"]:
            rows.append({"translator": name, "prompt_variant": pv, "model_checkpoint": e["model_checkpoint"],
                         "family": e["family"], "verified": e.get("verified", False),
                         "gated": bool(e.get("requires_hf_token")), "import_only": bool(e.get("import_only")),
                         "generation_config_id": __import__("sot_mt.translators", fromlist=["x"])
                         .generation_config(tcfg, name, pv)[0]})
    return pd.DataFrame(rows)


def completeness(base: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attrition for every pathway present in the table."""
    return schema.attrition(base, manifest) if len(base) else pd.DataFrame()


def default_work_dir(translator: str, prompt_variant: str, mock: bool = False) -> Path:
    return paths.output_root() / ("runs_mock" if mock else "runs") / f"{translator}__{prompt_variant}"


def open_runner(translator: str, prompt_variant: str, manifest: pd.DataFrame, base: pd.DataFrame,
                work_dir=None, prior_failure_paths=(), mock: bool = False, allow_download: bool = False,
                use_hf_token: bool = False, log=print, progress=None, **settings) -> runner.Runner:
    cfg, tcfg = load_yaml("experiment.yaml"), load_yaml("translators.yaml")
    if mock:
        adapter = MockTranslator(tcfg, translator, prompt_variant)
    else:
        adapter = build_translator(tcfg, translator, prompt_variant, allow_download=allow_download,
                                   use_hf_token=use_hf_token)
    work = Path(work_dir) if work_dir else default_work_dir(translator, prompt_variant, mock)
    return runner.Runner(runner.RunSettings(translator, prompt_variant, **settings), adapter, manifest, base, work,
                         cfg, tcfg, prior_failure_paths=tuple(prior_failure_paths), log=log, progress=progress)
