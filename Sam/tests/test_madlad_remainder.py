"""Consolidated MADLAD remainder (one session for the 168 items after pilot batch 1): manifest, rolling export
and recovery, deployment, final integrity check. CPU only; the batch-1 download is the base."""
import importlib.util
import json
import shutil
import sys

import pandas as pd
import pytest

from sot_mt import paths, runner, schema, session
from sot_mt import translators as T
from sot_mt.files import load_yaml, sha256_file

SAM = paths.SAM_ROOT
TCFG, CFG = load_yaml("translators.yaml"), load_yaml("experiment.yaml")
B01 = SAM / "data" / "generated" / "madlad_pilot" / "b01" / "xsum" / "madlad-3b"
SMOKE = SAM / "data" / "generated" / "smoke5_v2" / "xsum" / "madlad-3b"
REM = SAM / "data" / "manifests" / "madlad_pilot_remainder"
PILOT = SAM / "data" / "manifests" / "madlad_pilot"


def _script(name):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def real():
    if not (B01 / runner.MERGED).exists() or not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("batch-1 download or staged data unavailable")
    manifest, staged = session.load_inputs()
    return manifest, staged, runner.load_tables([B01 / runner.MERGED], manifest)


def _ids():
    return pd.read_csv(REM / "remainder_01.csv").item_id.tolist()


def test_remainder_is_exactly_the_168_unfinished_items(real):
    manifest, _, table = real
    ids = _ids()
    pw = table[table.translator == "madlad-3b"]
    st = pw.pivot(index="item_id", columns="output_stage", values="validation_status")
    assert len(ids) == len(set(ids)) == 168
    assert set(ids) == set(st.index[(st.T1 == "valid") & st.S2.isna()])
    assert not set(ids) & set(st.index[st.T3.notna()]) and len(st.index[st.T3 == "valid"]) == 29
    later = {i for k in range(2, 9) for i in pd.read_csv(PILOT / f"batch_{k:02d}.csv").item_id}
    assert set(ids) == later                                       # = pilot batches 2-8
    b = pd.read_csv(REM / "remainder_01.csv")
    m = manifest.set_index("item_id")
    assert set(b.source_label) == set(manifest.source_label) and b.source_label.nunique() == 7
    assert (b.article_id.values == m.loc[b.item_id, "article_id"].values).all()
    assert (b.group_id.values == m.loc[b.item_id, "group_id"].values).all()
    plan = json.loads((REM / "PLAN.json").read_text())
    assert plan["base"]["export_merged_sha256"] == sha256_file(B01 / runner.MERGED)
    assert plan["remaining_operations"] == 672


def test_batch1_base_passes_the_integrity_check(real):
    r = _script("verify_madlad_pilot").verify(B01, [SMOKE])
    assert (r["records"], r["stage2_records"], r["new_records"], r["complete_items"]) == (516, 400, 116, 29)
    assert r["earlier_exports_kept"][str(SMOKE)] == {"records": 420, "missing": 0, "changed": 0}
    assert (r["expected_new"], r["expected_total"]) == (788, 1188)


def test_verifier_detects_a_changed_earlier_record(real, tmp_path):
    vp = _script("verify_madlad_pilot")
    bad = tmp_path / "smoke"
    bad.mkdir()
    t = pd.read_parquet(SMOKE / runner.MERGED)
    i = t.index[t.output_stage == "S2"][0]
    t.loc[i, "output_sha256"] = "0" * 64
    t.to_parquet(bad / runner.MERGED)
    with pytest.raises(vp.IntegrityError, match="1 changed"):
        vp.verify(B01, [bad])


def test_full_mock_remainder_reaches_1188_and_keeps_the_116(real, tmp_path):
    manifest, _, table = real
    before = table[table.generation_meta.notna()].set_index("record_id")[["output_sha256", "run_id"]].sort_index()
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=_ids()),
                      T.MockTranslator(TCFG, "madlad-3b", "mt-default"), manifest, table, tmp_path, CFG, TCFG,
                      log=lambda *a: None)
    assert r.run()["operations"] == 672
    t = r.table
    assert len(t) == 1188 and t.record_id.is_unique and not t.duplicated(schema.KEY).any()
    assert len(t[t.output_stage.isin(["S2", "T2", "S3", "T3"])]) == 788
    assert schema.chain_violations(t, manifest) == []
    assert t.set_index("record_id").loc[before.index, ["output_sha256", "run_id"]].sort_index().equals(before)


def test_rolling_export_survives_a_hard_stop_and_resume_skips_completed_work(real, tmp_path):
    """The session dies at operation 30 and the notebook's export cell never runs: the latest rolling export
    (12 items: S2 ends at op 12, T2 at op 24) is consistent (RESUME hashes, pinnable) and a new session redoes
    only what it lacks."""
    manifest, _, table = real
    ids, exp = _ids()[:12], tmp_path / "export"
    mk = lambda script, base, work, **kw: runner.Runner(  # noqa: E731
        runner.RunSettings("madlad-3b", "mt-default", item_ids=ids, **kw),
        T.MockTranslator(TCFG, "madlad-3b", "mt-default", script=script), manifest, base, work, CFG, TCFG,
        log=lambda *a: None)
    r = mk({30: "interrupt"}, table, tmp_path / "w1", export_every=20, export_dir=str(exp))
    with pytest.raises(KeyboardInterrupt):
        r.run()
    resume = json.loads((exp / runner.RESUME).read_text())
    assert resume["records_this_work_dir"] == 24 and resume["records_total"] == 516 + 24
    assert _script("prepare_kaggle_deploy").planned_export_sha([exp]) == [sha256_file(exp / runner.MERGED)]
    base2 = runner.load_tables([exp / runner.MERGED], manifest)
    r2 = mk(None, base2, tmp_path / "w2")
    assert r2.run()["operations"] == 48 - 24
    assert len(r2.table) == 516 + 48 and schema.chain_violations(r2.table, manifest) == []


def test_rolling_export_also_at_each_stage_end(real, tmp_path):
    manifest, _, table = real
    exp = tmp_path / "export"
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=_ids()[:3], stop_after="S2",
                                         export_every=20, export_dir=str(exp)),
                      T.MockTranslator(TCFG, "madlad-3b", "mt-default"), manifest, table, tmp_path / "w", CFG, TCFG,
                      log=lambda *a: None)
    r.run()
    assert json.loads((exp / runner.RESUME).read_text())["records_this_work_dir"] == 3   # < export_every


def test_remainder_deployment(real, tmp_path):
    prep = _script("prepare_kaggle_deploy")
    P = prep.remainder_preset()
    assert P["slug"] == "sam-sot-mt-madlad-pilot-remainder" and P["resume_kernels"] == ["samahmed17/sam-sot-mt-madlad-pilot-b01"]
    assert P["timeout_s"] == 14400 and int(P["overrides"]["TIME_LIMIT_MINUTES"]) * 60 + 10 * 60 < P["timeout_s"]
    o = {**P["overrides"], "EXPECTED_PREVIOUS_MERGED_SHA256": json.dumps(prep.planned_export_sha(P["resume_exports"]))}
    d = prep.prepare(P["user"], P["slug"], tmp_path / "dep", o, P["timeout_s"], P["resume_kernels"],
                     build_out=tmp_path / "build", results_dir=P["results_dir"], generated_root=P["generated_root"])
    meta = json.loads((tmp_path / "dep" / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["kernel_sources"] == ["samahmed17/sam-sot-mt-madlad-pilot-b01"]
    assert (d["results_dir"], d["generated_root"]) == ("results/madlad_pilot/remainder", "data/generated/madlad_pilot/remainder")
    nb = json.loads((tmp_path / "dep" / "00_generate_translations.ipynb").read_text())
    cfg = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    for line in ['ITEM_IDS_FILE = "data/manifests/madlad_pilot_remainder/remainder_01.csv"', "TIME_LIMIT_MINUTES = 215",
                 "EXPORT_EVERY = 20", "DRIFT_PROBE_MAX_ITEMS = 7", "REFERENCE_GATE = True", "DRIFT_PROBE = True",
                 "CHECK_TOKENIZER_PARITY = True", "MADLAD_DIAGNOSTIC = False", f'"{sha256_file(B01 / runner.MERGED)}"']:
        assert line in cfg, line
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "export_every=EXPORT_EVERY, export_dir=str(EXPORT_DIR)" in src
    assert "except BaseException:\n    # keep every completed operation resumable" in src


def test_remainder_resume_attaches_only_the_previous_attempt():
    P = _script("prepare_kaggle_deploy").remainder_preset(attempt=2)
    assert P["slug"] == "sam-sot-mt-madlad-pilot-remainder-r2"
    assert P["resume_kernels"] == ["samahmed17/sam-sot-mt-madlad-pilot-remainder"]
    assert P["resume_exports"] == ["data/generated/madlad_pilot/remainder/xsum/madlad-3b"]
    assert P["results_dir"] == "results/madlad_pilot/remainder_r2"
    assert str(B01.relative_to(SAM)) in P["contains"] and str(SMOKE.relative_to(SAM)) in P["contains"]


def test_deployed_remainder_folder_matches_the_preset():
    d = json.loads((SAM / "kaggle" / "madlad_pilot" / "remainder" / "deploy.json").read_text())
    assert d["kernel"] == "samahmed17/sam-sot-mt-madlad-pilot-remainder" and d["timeout_s"] == 14400
    assert json.loads(d["overrides"]["EXPECTED_PREVIOUS_MERGED_SHA256"]) == [sha256_file(B01 / runner.MERGED)]
