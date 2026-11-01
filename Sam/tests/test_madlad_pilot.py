"""Full MADLAD pilot preparation: content warnings (nonblocking), batch plan, per-batch deployment, partial-export
resume. CPU only; the smoke-test download (data/generated/smoke5_v2) is the base and the calibration data."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from sot_mt import checks, paths, runner, session, smoke
from sot_mt import translators as T
from sot_mt.files import load_yaml, sha256_file

SAM = paths.SAM_ROOT
TCFG, CFG = load_yaml("translators.yaml"), load_yaml("experiment.yaml")
SMOKE = SAM / "data" / "generated" / "smoke5_v2" / "xsum" / "madlad-3b"
PLAN = SAM / "data" / "manifests" / "madlad_pilot" / "PLAN.json"
INVALID = {"xsum|BigScience|xsum-4713", "xsum|LLAMA|xsum-4714", "xsum|LLAMA|xsum-4733"}


def _script(name):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def real():
    if not (SMOKE / runner.MERGED).exists() or not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("smoke5_v2 download or staged data unavailable")
    manifest, staged = session.load_inputs()
    return manifest, staged, runner.load_tables([staged, SMOKE / runner.MERGED], manifest)


# -- content warnings ------------------------------------------------------------------------------------

def test_helpers():
    assert checks.sentence_count('He said. "It\'s an outrage." Tom won RS:X 2017. Would you like more?') == 4
    assert checks.sentence_count("no final stop") == 1 and checks.sentence_count("") == 0
    assert checks.numbers("£400,000 and 1.4bn; 2.000 millones") == {"400000", "14", "2000"}
    assert checks.numbers("400.000") == checks.numbers("400,000")
    assert checks.names("The Spurs beat Arsenal. Zana Marjanovic stars with Tor in the UK.") == {"spurs", "arsenal", "marjanovic", "tor"}


def test_warnings_flag_the_three_known_omissions_and_not_the_clean_chains(real):
    manifest, _, table = real
    v = CFG["validation"]
    t0 = manifest.set_index("item_id").t0_text
    pw = table[(table.translator == "madlad-3b") & table.output_stage.isin(["S2", "T2", "S3", "T3"])]
    flags = {}
    for r in pw.itertuples():
        w = checks.content_warnings(r.input_text, r.output_text, v)
        if r.output_stage.startswith("T"):
            prev = table[(table.item_id == r.item_id) & (table.translator == "madlad-3b") &
                         (table.output_stage == f"T{int(r.output_stage[1]) - 1}")].output_text.iloc[0]
            w += checks.roundtrip_content_warnings(prev, r.output_text, v)
        flags.setdefault(r.item_id, []).extend(w)
    assert any("sentence_drop" in w for w in flags["xsum|LLAMA|xsum-4729"])            # first sentence lost in S2
    assert any("entity_loss" in w and "marjanovic" in w for w in flags["xsum|Human|xsum-4721"])
    assert any("entity_loss" in w and "squires" in w for w in flags["xsum|OpenAI|xsum-4710"])
    assert any("roundtrip_length_loss" in w for w in flags["xsum|OpenAI|xsum-4710"])
    assert any("number_change" in w and "2000" in w for w in flags["xsum|LLAMA|xsum-4729"])
    assert flags["xsum|BigScience|xsum-4720"] == [] and flags["xsum|Tsinghua|xsum-4719"] == []


def test_content_warnings_never_block(real, tmp_path):
    manifest, _, table = real

    class Dropper(T.MockTranslator):            # fluent output that drops a sentence and changes a number
        def translate(self, text, src, tgt):
            r = super().translate(text, src, tgt)
            r.text = ("Este es el texto que se traduce con el modelo para la prueba del sistema en 1999."
                      if tgt == "es" else "This is the text that was translated with the model for the test in 1999.")
            return r
    item = json.loads(PLAN.read_text())["batches"][0]
    ids = pd.read_csv(PLAN.parent / item["file"]).item_id.tolist()[:2]
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=ids, stop_after="T2"),
                      Dropper(TCFG, "madlad-3b", "mt-default"), manifest, table, tmp_path, CFG, TCFG, run_id="d",
                      log=lambda *a: None)
    out = r.run()
    delta = pd.read_parquet(tmp_path / runner.DELTA)
    assert out["valid"] == 4 and (delta.validation_status == "valid").all()
    s2 = delta[delta.output_stage == "S2"]
    assert s2.validation_warnings.str.contains("check:number_change").all()       # S2 introduced "1999"
    assert delta.validation_warnings.str.contains("check:sentence_drop").any()


def test_stage2_rebuild_uses_stage2_rules_only():
    v = CFG["validation"]
    assert {"target_language", "content"} <= set(v) and not {"target_language", "content"} & set(checks.stage2_rules(v))
    out = checks.check_step("One. Two. Three.", "Uno 5.", "es", checks.stage2_rules(v))
    assert not any(w.startswith(("check:sentence_drop", "check:number_change")) for w in out["warnings"])


# -- batch plan --------------------------------------------------------------------------------------

def test_plan_covers_exactly_the_remaining_192_items(real):
    manifest, staged, table = real
    plan = json.loads(PLAN.read_text())
    batches = [pd.read_csv(PLAN.parent / b["file"]) for b in plan["batches"]]
    ids = [i for b in batches for i in b.item_id]
    pw = table[table.translator == "madlad-3b"]
    st = pw.pivot(index="item_id", columns="output_stage", values="validation_status")
    expected = set(st.index[(st.T1 == "valid") & st.S2.isna()])
    assert len(ids) == len(set(ids)) == 192 and set(ids) == expected
    assert not set(ids) & INVALID and not set(ids) & set(plan["already_complete_items"])
    assert len(plan["already_complete_items"]) == 5 and plan["remaining_operations"] == 768
    m = manifest.set_index("item_id")
    for b in batches:
        assert 20 <= len(b) <= 30
        assert set(b.source_label) == set(manifest.source_label) and len(set(manifest.source_label)) == 7
        assert (b.article_id.values == m.loc[b.item_id, "article_id"].values).all()
        assert (b.group_id.values == m.loc[b.item_id, "group_id"].values).all()
        assert (b.dataset == "xsum").all() and b.sample_index.is_monotonic_increasing
    assert plan["base"]["staged_generations_sha256"] == sha256_file(paths.staged_dir() / "generations.parquet")
    assert plan["base"]["export_merged_sha256"] == sha256_file(SMOKE / runner.MERGED)
    assert plan["generation_config_id"] == T.generation_config(TCFG, "madlad-3b", "mt-default")[0]


def test_planner_refuses_to_recut_an_existing_plan(monkeypatch):
    planner = _script("plan_madlad_batches")
    monkeypatch.setattr(sys, "argv", ["plan", "--out", str(PLAN.parent)])
    with pytest.raises(SystemExit, match="already holds a plan"):
        planner.main()


def test_runtime_model_reproduces_the_measured_smoke_runtime(real):
    planner = _script("plan_madlad_batches")
    d = pd.read_parquet(SMOKE / runner.DELTA)
    meta = d.generation_meta.map(json.loads)
    pred = sum(planner.op_seconds(len(t.split()), m["segmentation"]["n_segments"]) for t, m in zip(d.input_text, meta))
    meas = sum(m["seconds"] for m in meta)
    assert abs(pred - meas) / meas < 0.1


def test_batch1_mock_run_does_96_operations_and_keeps_the_smoke_records(real, tmp_path):
    manifest, _, table = real
    ids = pd.read_csv(PLAN.parent / "batch_01.csv").item_id.tolist()
    before = table[table.generation_meta.notna()].set_index("record_id").output_sha256.sort_index()
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=ids),
                      T.MockTranslator(TCFG, "madlad-3b", "mt-default"), manifest, table, tmp_path, CFG, TCFG,
                      log=lambda *a: None)
    assert r.run()["operations"] == 96
    after = r.table.set_index("record_id").output_sha256.loc[before.index].sort_index()
    assert after.equals(before)


# -- per-batch deployment and resume ---------------------------------------------------------------------

def test_batch1_deployment(real, tmp_path):
    prep = _script("prepare_kaggle_deploy")
    P = prep.pilot_preset(1)
    assert P["slug"] == "sam-sot-mt-madlad-pilot-b01" and P["resume_kernels"] == ["samahmed17/sam-sot-mt-madlad-smoke5"]
    o = {**P["overrides"], "EXPECTED_PREVIOUS_MERGED_SHA256": json.dumps(prep.planned_export_sha(P["resume_exports"]))}
    d = prep.prepare(P["user"], P["slug"], tmp_path / "dep", o, P["timeout_s"], P["resume_kernels"],
                     build_out=tmp_path / "build", results_dir=P["results_dir"], generated_root=P["generated_root"])
    meta = json.loads((tmp_path / "dep" / "kernel-metadata.json").read_text())
    assert meta["is_private"] and meta["enable_gpu"] and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["kernel_sources"] == ["samahmed17/sam-sot-mt-madlad-smoke5"]
    assert d["results_dir"] == "results/madlad_pilot/b01" and 3600 <= d["timeout_s"] <= 43200
    cfg = "".join(next(c for c in json.loads((tmp_path / "dep" / "00_generate_translations.ipynb").read_text())
                       ["cells"] if c["cell_type"] == "code")["source"])
    for line in ['ITEM_IDS_FILE = "data/manifests/madlad_pilot/batch_01.csv"', "DRIFT_PROBE_MAX_ITEMS = 2",
                 "REFERENCE_GATE = True", f'"{sha256_file(SMOKE / runner.MERGED)}"']:
        assert line in cfg, line


def test_resume_attempt_requires_the_previous_attempt_to_be_downloaded(real):
    prep = _script("prepare_kaggle_deploy")
    P = prep.pilot_preset(1, attempt=2)
    assert P["slug"] == "sam-sot-mt-madlad-pilot-b01-r2"
    assert P["resume_kernels"] == ["samahmed17/sam-sot-mt-madlad-smoke5", "samahmed17/sam-sot-mt-madlad-pilot-b01"]
    assert P["resume_exports"][1] == "data/generated/madlad_pilot/b01/xsum/madlad-3b"
    if not (SAM / P["resume_exports"][1]).exists():
        with pytest.raises(FileNotFoundError):
            prep.planned_export_sha(P["resume_exports"])


def test_interrupted_run_exports_a_resumable_partial(real, tmp_path):
    manifest, _, table = real
    ids = pd.read_csv(PLAN.parent / "batch_01.csv").item_id.tolist()[:3]
    mk = lambda script, base, work: runner.Runner(  # noqa: E731
        runner.RunSettings("madlad-3b", "mt-default", item_ids=ids),
        T.MockTranslator(TCFG, "madlad-3b", "mt-default", script=script), manifest, base, work, CFG, TCFG,
        log=lambda *a: None)
    r = mk({5: "interrupt"}, table, tmp_path / "w1")
    with pytest.raises(KeyboardInterrupt):
        r.run()
    resume = r.export(tmp_path / "export")                       # what the notebook's except-branch does
    assert resume["records_this_work_dir"] == 4
    base2 = runner.load_tables([tmp_path / "export" / runner.MERGED], manifest)
    assert mk(None, base2, tmp_path / "w2").run()["operations"] == 12 - 4     # a new session resumes
    nb = "".join("".join(c["source"]) for c in json.loads((SAM / "notebooks" / "00_generate_translations.ipynb")
                                                          .read_text())["cells"])
    assert "except BaseException:\n    # keep every completed operation resumable" in nb
    assert "list(run.items.item_id)[:DRIFT_PROBE_MAX_ITEMS]" in nb
    assert len(base2) == len(table) + 4
