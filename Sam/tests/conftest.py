import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import corpus, legacy_madlad, manifest, paths, schema  # noqa: E402
from sot_mt.files import load_yaml, sha256_text  # noqa: E402

# -- synthetic fixtures ----------------------------------------------------------------------------


@pytest.fixture
def t0():
    """Three T0 texts about two articles."""
    return pd.DataFrame({
        "item_id": ["xsum|Human|a1", "xsum|OpenAI|a1", "xsum|Human|a2"],
        "dataset": "xsum", "article_id": ["a1", "a1", "a2"], "source_label": ["Human", "OpenAI", "Human"],
        "t0_text": ["The cat sat on the mat.", "The cat sat on the mat and slept.", "Rain fell over the city."],
    })


@pytest.fixture
def make_row(t0):
    labels = t0.set_index("item_id")

    def make(item_id, stage, input_text, output_text, status="valid", reason=None,
             translator="madlad-3b", prompt_variant="mt-default", model="google/madlad400-3b-mt"):
        lab = labels.loc[item_id]
        row = {"experiment_id": "exp", "dataset": lab.dataset, "item_id": item_id, "article_id": lab.article_id,
               "source_label": lab.source_label, "translator": translator, "prompt_variant": prompt_variant,
               "model_checkpoint": model, "model_revision": None, **schema.stage_info(stage),
               "output_stage": stage, "input_text": input_text, "output_text": output_text,
               "input_sha256": sha256_text(input_text),
               "output_sha256": None if output_text is None else sha256_text(output_text),
               "validation_status": status, "validation_reason": reason, "validation_warnings": None,
               "generation_config_id": "cfg", "run_id": "run", "provenance": "test", "legacy_fields": None}
        row["record_id"] = schema.record_id(row)
        return row
    return make


@pytest.fixture
def full_chain(t0, make_row):
    """A valid T0 -> S1 -> T1 -> S2 -> T2 -> S3 -> T3 chain for one item, as a list of rows."""
    item = "xsum|Human|a1"
    text = t0.set_index("item_id").t0_text[item]
    rows = []
    for stage in schema.GENERATED_STAGES:
        out = f"{stage} of {item}"
        rows.append(make_row(item, stage, text, out))
        text = out
    return rows


# -- real data (skipped when the staged corpus or team files are absent, e.g. on Kaggle) -----------


@pytest.fixture(scope="session")
def cfg():
    return load_yaml("experiment.yaml")


@pytest.fixture(scope="session")
def real(cfg):
    cdir = paths.corpus_dir()
    legacy_path = paths.repo_file(cfg["legacy"]["madlad_chain_csv"])
    if not corpus.staged_info(cdir) or not legacy_path.exists():
        pytest.skip(f"real data unavailable (staged corpus at {cdir}, legacy CSV at {legacy_path})")
    info = corpus.staged_info(cdir)
    raw = corpus.load_paraphrased(cdir, cfg["sample"]["dataset"])
    man = manifest.build_manifest(raw, cfg, cdir / corpus.paraphrased_file("xsum"), info["repo_commit"])
    legacy = legacy_madlad.read_legacy_chain(legacy_path)
    tcfg = load_yaml("translators.yaml")
    from sot_mt.files import sha256_file
    sha = sha256_file(legacy_path)
    rows = legacy_madlad.import_rows(legacy, man, cfg, tcfg, cfg["legacy"]["madlad_chain_csv"], sha)
    gen, stats = schema.upsert(schema.empty_table(), rows, man, allow_invalid_predecessor=True)
    return {"raw": raw, "manifest": man, "legacy": legacy, "legacy_path": legacy_path, "legacy_sha": sha,
            "rows": rows, "gen": gen, "stats": stats, "tcfg": tcfg}
