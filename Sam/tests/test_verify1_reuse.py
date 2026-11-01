"""Reuse of the verified one-document chain (kernel sam-sot-mt-madlad-verify1 v2) in the five-document rerun:
the four records keep their provenance, the Stage 2 files are untouched, and the Kaggle notebook only accepts
the exact export that was downloaded and checked locally. CPU only."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sot_mt import paths, runner, schema, session, smoke
from sot_mt import translators as T
from sot_mt.files import load_yaml

SAM = paths.SAM_ROOT
TCFG, CFG = load_yaml("translators.yaml"), load_yaml("experiment.yaml")
V1 = SAM / "data" / "generated" / "verify1" / "xsum" / "madlad-3b"
ITEM = "xsum|LLAMA|xsum-4729"
SMOKE_IDS = SAM / "data" / "manifests" / "stage3b_madlad_smoke5.csv"


def _prep():
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location("prep", SAM / "scripts" / "prepare_kaggle_deploy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def real():
    if not (V1 / "generations_merged.parquet").exists() or not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("verify1 download or staged data unavailable")
    manifest, base = session.load_inputs()
    merged = runner.load_tables([V1 / "generations_merged.parquet"], manifest)
    return manifest, base, merged


def test_verify1_chain_is_complete_and_correctly_linked(real):
    manifest, base, merged = real
    ch = merged[(merged.item_id == ITEM) & (merged.translator == "madlad-3b")].set_index("output_stage")
    assert list(ch.index.sort_values()) == sorted(schema.GENERATED_STAGES)
    assert (ch.validation_status == "valid").all() and set(ch.generation_config_id) == {"705a8bd4992e"}
    assert set(ch.model_checkpoint) == {"google/madlad400-3b-mt"}
    for st, prev in [("S2", "T1"), ("T2", "S2"), ("S3", "T2"), ("T3", "S3")]:
        r = ch.loc[st]
        meta = json.loads(r.generation_meta)
        assert r.input_stage == prev and r.input_text == ch.loc[prev].output_text
        assert meta["embedding_wiring"] == "decoder" and meta["runtime_patch"] == "madlad-embed-decoder-v1"
        assert meta["finish"] == "stop" and all(meta["segment_hit_eos"])
    assert not set(ch.run_id) & set(CFG["quarantined_runs"])
    assert schema.chain_violations(merged, manifest) == []


def test_stage2_records_are_unchanged_inside_the_verify1_export(real):
    manifest, base, merged = real
    cols = ["record_id", "input_sha256", "output_sha256", "validation_status", "run_id"]
    a = base[cols].sort_values("record_id").reset_index(drop=True)
    b = merged[merged.record_id.isin(base.record_id)][cols].sort_values("record_id").reset_index(drop=True)
    assert a.fillna("").equals(b.fillna("")) and len(merged) == 404


def test_reuse_plans_16_operations_and_keeps_the_verified_records(real, tmp_path):
    manifest, _, merged = real
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=smoke.read_item_ids(SMOKE_IDS)),
                      T.MockTranslator(TCFG, "madlad-3b", "mt-default"), manifest, merged, tmp_path, CFG, TCFG,
                      log=lambda *a: None)
    plan, _ = r.progress_plan()
    assert ITEM not in plan and sum(len(v) for v in plan.values()) == 16
    before = merged[merged.item_id == ITEM].set_index("record_id")[["output_sha256", "run_id", "generation_meta"]]
    assert r.run()["operations"] == 16
    after = r.table[r.table.item_id == ITEM].set_index("record_id")[["output_sha256", "run_id", "generation_meta"]]
    assert after.sort_index().equals(before.sort_index())                    # provenance unchanged
    assert set(pd.read_parquet(tmp_path / runner.DELTA).item_id) == set(plan)  # nothing redone for xsum-4729


def test_planned_export_sha_pins_the_checked_download(real, tmp_path):
    prep = _prep()
    sha = prep.planned_export_sha([V1])
    assert sha == [hashlib.sha256((V1 / "generations_merged.parquet").read_bytes()).hexdigest()]
    bad = tmp_path / "tampered"
    shutil.copytree(V1, bad)
    res = json.loads((bad / "RESUME.json").read_text())
    res["files"]["generations_merged.parquet"] = "0" * 64
    (bad / "RESUME.json").write_text(json.dumps(res))
    with pytest.raises(ValueError, match="RESUME"):
        prep.planned_export_sha([bad])
    quarantined = SAM / "data" / "generated" / "xsum" / "madlad-3b"
    if (quarantined / "RESUME.json").exists():
        with pytest.raises(ValueError, match="quarantined"):
            prep.planned_export_sha([quarantined])


def test_smoke_rerun_deployment_reuses_verify1_in_new_folders(real, tmp_path):
    prep = _prep()
    P = prep.PRESETS["smoke5"]
    o = {**P["overrides"], "EXPECTED_PREVIOUS_MERGED_SHA256": json.dumps(prep.planned_export_sha(P["resume_exports"]))}
    d = prep.prepare(P["user"], P["slug"], tmp_path / "dep", o, P["timeout_s"], P["resume_kernels"],
                     build_out=tmp_path / "build", results_dir=P["results_dir"], generated_root=P["generated_root"])
    meta = json.loads((tmp_path / "dep" / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["samahmed17/sam-sot-mt-madlad-verify1"]
    assert d["results_dir"] == "results/madlad_smoke5_v2" and d["generated_root"] == "data/generated/smoke5_v2"
    cfg = "".join(next(c for c in json.loads((tmp_path / "dep" / "00_generate_translations.ipynb").read_text())
                       ["cells"] if c["cell_type"] == "code")["source"])
    for line in ["MAX_ITEMS = 5", "DRIFT_PROBE = True", "REFERENCE_GATE = True", "MADLAD_DIAGNOSTIC = False",
                 'PREVIOUS_EXPORTS = "auto"', "EXPECTED_PREVIOUS_MERGED_SHA256 = ["]:
        assert line in cfg, line


# -- the notebook's guard on attached exports (simulated Kaggle, MOCK) -----------------------------------

def _nb_script(tmp, inp, out_dir, extra):
    nb = json.loads((SAM / "notebooks" / "00_generate_translations.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    s = tmp / f"nb_{out_dir.parent.name}.py"
    s.write_text("\n\n".join(["display = print", cells[0], f"MOCK = True\nOUTPUT_DIR = {str(out_dir)!r}\n{extra}",
                              *cells[1:]]))
    return subprocess.run([sys.executable, "-I", str(s)], capture_output=True, text=True, cwd=tmp,
                          env={"SAM_KAGGLE_INPUT": str(inp), "PATH": "/usr/bin:/bin", "HOME": str(tmp)})


@pytest.fixture(scope="module")
def attached(tmp_path_factory, real):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location("pk", SAM / "scripts" / "package_kaggle_inputs.py")
    pk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pk)
    tmp = tmp_path_factory.mktemp("guard")
    built = pk.build(tmp / "build", "tester", paths.staged_dir())
    inp = tmp / "input"
    shutil.copytree(built["code"], inp / "sam-sot-mt-code")
    shutil.copytree(built["staged"], inp / "sam-sot-mt-staged")
    res = _nb_script(tmp, inp, tmp / "s1" / "sam_outputs", "")
    assert res.returncode == 0, res.stderr[-2000:]
    shutil.copytree(tmp / "s1" / "sam_outputs", inp / "previous-kernel-output" / "sam_outputs")
    merged = inp / "previous-kernel-output/sam_outputs/export/madlad-3b__mt-default/generations_merged.parquet"
    return tmp, inp, hashlib.sha256(merged.read_bytes()).hexdigest()


def test_notebook_accepts_the_planned_export(attached):
    tmp, inp, sha = attached
    res = _nb_script(tmp, inp, tmp / "ok" / "sam_outputs", f"EXPECTED_PREVIOUS_MERGED_SHA256 = [{sha!r}]")
    assert res.returncode == 0, res.stderr[-2000:]
    assert "previous exports match the planned sha256" in res.stdout and "this run: 0 records" in res.stdout


def test_notebook_refuses_a_different_export(attached):
    tmp, inp, _ = attached
    res = _nb_script(tmp, inp, tmp / "bad" / "sam_outputs", f"EXPECTED_PREVIOUS_MERGED_SHA256 = [{'0' * 64!r}]")
    assert res.returncode != 0 and "are not the planned ones" in res.stderr


def test_notebook_refuses_when_the_planned_export_is_missing(attached, tmp_path):
    _, inp, sha = attached
    bare = tmp_path / "input"
    shutil.copytree(inp / "sam-sot-mt-code", bare / "sam-sot-mt-code")
    shutil.copytree(inp / "sam-sot-mt-staged", bare / "sam-sot-mt-staged")
    res = _nb_script(tmp_path, bare, tmp_path / "none" / "sam_outputs", f"EXPECTED_PREVIOUS_MERGED_SHA256 = [{sha!r}]")
    assert res.returncode != 0 and "are not the planned ones" in res.stderr
