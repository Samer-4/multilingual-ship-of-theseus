import copy
import json

import pandas as pd
import pytest

from sot_mt import legacy_qwen, runner as R, schema
from sot_mt import translators as T
from sot_mt.files import load_yaml, sha256_file

TCFG = load_yaml("translators.yaml")
CFG = dict(load_yaml("experiment.yaml"), experiment_id="exp")


@pytest.fixture(autouse=True)
def no_loaded_models():
    T._LOADED.clear()
    yield
    T._LOADED.clear()


@pytest.fixture
def man(t0):
    return t0.assign(sample_index=range(len(t0)))


def make_runner(man, work, translator="nllb", pv="mt-default", base=None, script=None, cfg=CFG, tcfg=TCFG,
                run_id="run-test", **kw):
    tr = T.MockTranslator(tcfg, translator, pv, script=script)
    settings = R.RunSettings(translator=translator, prompt_variant=pv, **kw)
    return R.Runner(settings, tr, man, schema.empty_table() if base is None else base, work, cfg, tcfg,
                    run_id=run_id, log=lambda *a: None)


def outputs(table):
    return table.set_index(["translator", "item_id", "output_stage"]).output_text.to_dict()


def assert_sequential(table, man):
    """S1 starts from T0; every later input is the same pathway's previous output; nothing restarts."""
    t0 = man.set_index("item_id").t0_text
    by = table.set_index(schema.KEY)
    for r in table.itertuples():
        if r.output_stage == "S1":
            assert r.input_text == t0[r.item_id]
        else:
            prev = by.loc[(r.experiment_id, r.translator, r.prompt_variant, r.item_id, r.input_stage)]
            assert r.input_text == prev.output_text and prev.validation_status == "valid"
            assert r.input_text != t0[r.item_id]
    assert schema.chain_violations(table, man) == []


# -- complete chains and resumption ---------------------------------------------------------------

def test_complete_three_round_chain(man, tmp_path):
    r = make_runner(man, tmp_path)
    assert r.plan().ready_now.tolist() == [3, 0, 0, 0, 0, 0]
    summary = r.run()
    assert summary == {"run_id": "MOCK-run-test", "status": "completed", "operations": 18, "valid": 18, "invalid": 0,
                       "failed_attempts": 0}
    t = r.table
    assert len(t) == 18 and t.groupby("item_id").output_stage.apply(list).map(len).eq(6).all()
    assert (t.direction == t.output_stage.str[0].map({"S": "en->es", "T": "es->en"})).all()
    assert_sequential(t, man)
    assert (tmp_path / R.DELTA).exists() and len(pd.read_parquet(tmp_path / R.DELTA)) == 18
    # labels are carried from the manifest
    j = t.merge(man, on="item_id", suffixes=("", "_m"))
    assert (j.source_label == j.source_label_m).all() and (j.article_id == j.article_id_m).all()
    again = make_runner(man, tmp_path)                       # restart on a finished work dir: nothing to do
    assert again.run()["operations"] == 0


def test_resume_after_s2_and_after_t2(man, tmp_path):
    make_runner(man, tmp_path, stop_after="S2").run()
    d = pd.read_parquet(tmp_path / R.DELTA)
    assert sorted(d.output_stage.unique()) == ["S1", "S2", "T1"]
    r = make_runner(man, tmp_path, stop_after="T2")
    assert r.plan().set_index("stage").ready_now.to_dict() == {"S1": 0, "T1": 0, "S2": 0, "T2": 3}
    assert r.run()["operations"] == 3                         # only T2: S1-S2 are not regenerated
    r = make_runner(man, tmp_path)
    assert r.run()["operations"] == 6                         # S3 and T3
    assert_sequential(r.table, man)
    ref = make_runner(man, tmp_path / "ref")
    ref.run()
    assert outputs(r.table) == outputs(ref.table)             # same result as one uninterrupted run


def test_resume_from_existing_madlad_t1(real, tmp_path):
    from sot_mt import paths
    staged = paths.staged_dir() / "generations.parquet"
    before = sha256_file(staged) if staged.exists() else None
    m, base = real["manifest"], real["gen"]
    flagged = "xsum|BigScience|xsum-4713"
    items = m.item_id.head(3).tolist() + [flagged]
    tr = T.MockTranslator(TCFG, "madlad-3b", "mt-default")
    run = R.Runner(R.RunSettings("madlad-3b", "mt-default", item_ids=items), tr, m, base, tmp_path,
                   load_yaml("experiment.yaml"), TCFG, run_id="madlad-cont", log=lambda *a: None)
    plan = run.plan().set_index("stage")
    assert plan.loc["S1", "valid"] == 3 and plan.loc["S1", "invalid"] == 1 and plan.loc["S1", "ready_now"] == 0
    assert plan.loc["S2", "ready_now"] == 3 and plan.loc["S2", "blocked"] == 1
    assert run.run()["operations"] == 12                      # 3 valid items x S2..T3; flagged item blocked
    new = run.table[run.table.run_id == "MOCK-madlad-cont"]
    assert set(new.output_stage) == {"S2", "T2", "S3", "T3"} and flagged not in set(new.item_id)
    t1 = base[base.output_stage == "T1"].set_index("item_id").output_text
    s2 = new[new.output_stage == "S2"].set_index("item_id")
    assert (s2.input_text == t1.loc[s2.index]).all()          # S2 starts from the legacy T1, not T0
    assert (new.generation_config_id == "705a8bd4992e").all()  # same config as the Stage 2 records
    assert len(pd.read_parquet(tmp_path / R.DELTA)) == 12      # only new records in the delta
    assert run.table[run.table.run_id != "MOCK-madlad-cont"].reset_index(drop=True).equals(base)  # base untouched
    if before:
        assert sha256_file(staged) == before


# -- interruption ---------------------------------------------------------------------------------

class HardCrash(BaseException):
    """Simulates the kernel dying: not caught by the runner, no flush."""


def test_keyboard_interrupt_flushes_completed_operations(man, tmp_path):
    r = make_runner(man, tmp_path, script={5: "interrupt"}, checkpoint_every=100)
    with pytest.raises(KeyboardInterrupt):
        r.run()
    d = pd.read_parquet(tmp_path / R.DELTA)
    assert len(d) == 4 and (d.validation_status == "valid").all()     # 3 S1 + 1 T1; the 5th never existed
    runs = pd.read_parquet(tmp_path / R.RUNS)
    assert runs.status.tolist() == ["interrupted"]
    r2 = make_runner(man, tmp_path, run_id="run-2")
    assert r2.run()["operations"] == 14
    ref = make_runner(man, tmp_path / "ref")
    ref.run()
    assert outputs(r2.table) == outputs(ref.table)


def test_hard_crash_loses_only_unflushed_work(man, tmp_path):
    def crash(text, src, tgt, call):
        if call == 5:
            raise HardCrash()
    r = make_runner(man, tmp_path, script=crash, checkpoint_every=2)
    with pytest.raises(HardCrash):
        r.run()
    # flushed after ops 1-2 and at the end of S1 (op 3); T1 op 4 was still buffered when the crash hit
    assert len(pd.read_parquet(tmp_path / R.DELTA)) == 3
    (tmp_path / ".generations_delta.parquet.999.tmp").write_bytes(b"partial garbage")   # torn temp file
    r2 = make_runner(man, tmp_path, run_id="run-2")
    r2.run()
    ref = make_runner(man, tmp_path / "ref")
    ref.run()
    assert outputs(r2.table) == outputs(ref.table)
    assert_sequential(r2.table, man)


# -- invalid outputs, failures and blocking --------------------------------------------------------

def test_invalid_output_blocks_rest_of_chain(man, tmp_path):
    # calls 1-3 S1, 4-6 T1, 7-9 S2 (manifest order): item 2's S2 loops, item 3's S2 hits the cap
    r = make_runner(man, tmp_path, script={8: "loop", 9: "length"})
    s = r.run()
    assert s["invalid"] == 2
    t = r.table.set_index(["item_id", "output_stage"])
    i2, i3 = man.item_id[1], man.item_id[2]
    assert "check:repetition" in t.loc[(i2, "S2"), "validation_reason"]
    assert t.loc[(i3, "S2"), "validation_reason"] == "engine:generation_cap"
    assert t.loc[(i3, "S2"), "output_text"]                              # text kept, never repaired
    for item in [i2, i3]:
        assert not {"T2", "S3", "T3"} & set(r.table[r.table.item_id == item].output_stage)
    att = schema.attrition(r.table, man).set_index("output_stage")
    assert att.loc["S2", ["valid", "invalid"]].tolist() == [1, 2]
    assert att.loc["T3", ["valid", "blocked"]].tolist() == [1, 2]
    assert make_runner(man, tmp_path, run_id="again").run()["operations"] == 0   # invalid is final


def test_empty_segment_record_has_no_output(man, tmp_path):
    r = make_runner(man, tmp_path, script={1: "empty_segment"}, rounds=(1, 1))
    r.run()
    row = r.table.set_index(["item_id", "output_stage"]).loc[(man.item_id[0], "S1")]
    assert row.validation_status == "invalid" and row.output_text is None
    assert row.validation_reason == "engine:empty_segment"


def test_transient_failures_are_logged_separately_and_retried(man, tmp_path):
    r = make_runner(man, tmp_path, script={2: "raise"}, rounds=(1, 1))
    s = r.run()
    assert s["failed_attempts"] == 1 and s["valid"] == 4               # item 2: no S1, so no T1 either
    f = pd.read_parquet(tmp_path / R.FAILURES)
    assert f.item_id.tolist() == [man.item_id[1]] and f.attempt.tolist() == [1]
    assert man.item_id[1] not in set(pd.read_parquet(tmp_path / R.DELTA).item_id)
    r2 = make_runner(man, tmp_path, rounds=(1, 1), run_id="retry")
    assert r2.run()["operations"] == 2
    meta = json.loads(r2.table.set_index(["item_id", "output_stage"]).loc[(man.item_id[1], "S1"), "generation_meta"])
    assert meta["attempt"] == 2


def test_operation_skipped_after_max_attempts(man, tmp_path):
    always = lambda text, src, tgt, call: "raise" if text == man.t0_text[1] else None  # noqa: E731
    for k in range(2):
        make_runner(man, tmp_path, script=always, rounds=(1, 1), run_id=f"r{k}").run()
    r = make_runner(man, tmp_path, rounds=(1, 1), run_id="r3")
    assert r.plan().set_index("stage").loc["S1", "ready_now"] == 0
    assert r.run()["operations"] == 0


def test_budget_limits_stop_cleanly(man, tmp_path):
    r = make_runner(man, tmp_path, max_operations=4)
    assert r.run()["status"] == "stopped:max_operations"
    assert len(pd.read_parquet(tmp_path / R.DELTA)) == 4
    assert make_runner(man, tmp_path, run_id="b").run()["operations"] == 14


# -- isolation, duplicates, configs ---------------------------------------------------------------

def test_no_cross_model_contamination(man, tmp_path):
    a = make_runner(man, tmp_path / "madlad", translator="madlad-3b")
    a.run()
    b = make_runner(man, tmp_path / "nllb", translator="nllb", base=a.table)
    assert b.plan().set_index("stage").loc["S1", "ready_now"] == 3     # NLLB starts at T0, not at MADLAD's T1
    b.run()
    nllb = b.table[b.table.translator == "nllb"]
    madlad_out = set(a.table.output_text)
    assert not set(nllb.input_text) & madlad_out
    assert_sequential(b.table, man)


def test_config_and_identity_mismatches_are_rejected(man, tmp_path):
    a = make_runner(man, tmp_path / "a", translator="nllb")
    a.run()
    tr = T.MockTranslator(TCFG, "madlad-3b", "mt-default")
    with pytest.raises(R.ConfigMismatchError, match="adapter is madlad-3b"):
        R.Runner(R.RunSettings("nllb", "mt-default"), tr, man, a.table, tmp_path / "b", CFG, TCFG)
    changed = copy.deepcopy(TCFG)
    changed["translators"]["nllb"]["settings"]["decoding"]["num_beams"] = 4
    with pytest.raises(R.ConfigMismatchError, match="existing nllb/mt-default records use config"):
        make_runner(man, tmp_path / "c", base=a.table, tcfg=changed)
    tr = T.MockTranslator(TCFG, "nllb", "mt-default")
    tr.generation_config_id = "tampered0000"
    with pytest.raises(R.ConfigMismatchError, match="translators.yaml"):
        R.Runner(R.RunSettings("nllb", "mt-default"), tr, man, a.table, tmp_path / "d", CFG, TCFG)


def test_real_runs_refuse_mock_records(man, tmp_path):
    a = make_runner(man, tmp_path / "mock", rounds=(1, 1))
    a.run()
    assert a.table.run_id.str.startswith("MOCK-").all()
    real_adapter = T.build_translator(TCFG, "nllb", "mt-default")            # constructed, never loaded
    with pytest.raises(R.MockDataError, match="base contains mock records"):
        R.Runner(R.RunSettings("nllb", "mt-default"), real_adapter, man, a.table, tmp_path / "real", CFG, TCFG)
    with pytest.raises(R.MockDataError, match="work-dir delta"):
        R.Runner(R.RunSettings("nllb", "mt-default"), real_adapter, man, schema.empty_table(), tmp_path / "mock",
                 CFG, TCFG)


def test_stable_ids_across_independent_runs(man, tmp_path):
    a, b = make_runner(man, tmp_path / "a"), make_runner(man, tmp_path / "b", run_id="other")
    a.run(), b.run()
    assert a.table.record_id.tolist() == b.table.record_id.tolist()
    assert outputs(a.table) == outputs(b.table)
    assert a.table.generation_config_id.unique().tolist() == [T.generation_config(TCFG, "nllb", "mt-default")[0]]


def test_duplicate_and_conflicting_tables(man, tmp_path):
    a = make_runner(man, tmp_path / "a", rounds=(1, 1))
    a.run()
    assert len(R.load_tables([a.table, a.table], man)) == 6           # identical duplicates merge
    other = a.table.copy()
    other.loc[0, "output_text"] = "El documento distinto es una traducción."
    other.loc[0, "output_sha256"] = __import__("sot_mt.files", fromlist=["x"]).sha256_text(other.loc[0, "output_text"])
    with pytest.raises(schema.ConflictError):
        R.load_tables([a.table, other], man)
    dup = pd.concat([a.table, a.table.head(1)])
    with pytest.raises(schema.DuplicateRecordError):
        R.load_tables([dup], man)


def test_selection_filters(man, tmp_path):
    r = make_runner(man, tmp_path, article_ids=["a1"], rounds=(1, 1))
    assert r.items.item_id.tolist() == ["xsum|Human|a1", "xsum|OpenAI|a1"]
    assert make_runner(man, tmp_path, max_items=1).items.item_id.tolist() == ["xsum|Human|a1"]
    assert R.stages_for((2, 3)) == ["S2", "T2", "S3", "T3"] and R.stages_for((1, 3), "T1") == ["S1", "T1"]


# -- export / next session ------------------------------------------------------------------------

def test_export_and_resume_in_new_session(man, tmp_path):
    r = make_runner(man, tmp_path / "session1", script={3: "raise"}, stop_after="T1")
    r.run()
    resume = r.export(tmp_path / "export1")
    assert set(resume["files"]) == {R.MERGED, R.DELTA, R.FAILURES, R.RUNS, R.REPORT, R.ATTRITION, R.NEW_TEXTS}
    assert (tmp_path / "export1" / R.RESUME).exists()
    report = pd.read_csv(tmp_path / "export1" / R.REPORT)
    assert len(report) == 4 and "output_text" not in report and set(report.finish) == {"stop"}
    assert len(pd.read_csv(tmp_path / "export1" / R.NEW_TEXTS)) == 4
    base = R.load_tables([schema.empty_table(), tmp_path / "export1" / R.MERGED], man)
    tr = T.MockTranslator(TCFG, "nllb", "mt-default")
    r2 = R.Runner(R.RunSettings("nllb", "mt-default"), tr, man, base, tmp_path / "session2", CFG, TCFG,
                  prior_failure_paths=[tmp_path / "export1" / R.FAILURES], run_id="s2", log=lambda *a: None)
    assert r2.run()["operations"] == 18 - 4
    assert_sequential(r2.table, man)
    assert len(pd.read_parquet(tmp_path / "session2" / R.DELTA)) == 14   # session 2 delta has only new work


def test_export_keeps_earlier_sessions_failures(man, tmp_path):
    # session 1 fails twice on one operation (exhausted); session 2 has no failures of its own, but its
    # export must still carry session 1's attempts, or a third session would retry the exhausted operation
    r = make_runner(man, tmp_path / "s1", script=lambda text, src, tgt, n: "raise" if n <= 2 else None,
                    stop_after="S1", max_items=1)
    r.run(); r.run()
    r.export(tmp_path / "e1")
    tr = T.MockTranslator(TCFG, "nllb", "mt-default")
    base = R.load_tables([tmp_path / "e1" / R.MERGED], man)
    r2 = R.Runner(R.RunSettings("nllb", "mt-default", stop_after="S1", max_items=1), tr, man, base,
                  tmp_path / "s2", CFG, TCFG, prior_failure_paths=[tmp_path / "e1" / R.FAILURES], run_id="s2",
                  log=lambda *a: None)
    assert r2.run()["operations"] == 0
    r2.export(tmp_path / "e2")
    assert len(pd.read_parquet(tmp_path / "e2" / R.FAILURES)) == 2


def test_load_failure_is_recorded_not_left_running(man, tmp_path):
    class NoGPU(T.MockTranslator):
        def _load(self):
            raise T.NoGPUError("no CUDA GPU visible")
    tr = NoGPU(TCFG, "nllb", "mt-default")
    r = R.Runner(R.RunSettings("nllb", "mt-default"), tr, man, schema.empty_table(), tmp_path, CFG, TCFG,
                 run_id="x", log=lambda *a: None)
    with pytest.raises(T.NoGPUError):
        r.run()
    assert pd.read_parquet(tmp_path / R.RUNS).status.tolist() == ["failed:load:NoGPUError"]
    assert not (tmp_path / R.DELTA).exists()


# -- Qwen legacy artifacts --------------------------------------------------------------------------

def test_missing_qwen_artifacts_mean_full_run_from_t0(man, tmp_path):
    with pytest.raises(legacy_qwen.ArtifactNotFoundError):
        legacy_qwen.import_qwen_chain(tmp_path / "chain_es_Qwen3-4B-Instruct-2507.csv", man, CFG, TCFG)
    with pytest.raises(legacy_qwen.ArtifactNotFoundError):
        legacy_qwen.import_qwen_chain(None, man, CFG, TCFG)
    r = make_runner(man, tmp_path, translator="qwen3-4b", pv="free")
    assert r.plan().set_index("stage").loc["S1", "ready_now"] == 3


def test_qwen_artifacts_import_into_separate_pathway(man, tmp_path):
    t0 = man.set_index("item_id").t0_text
    csv = pd.DataFrame({
        "item_id": man.item_id, "S": man.t0_text.values,
        "F": ["El gato se sentó en la alfombra.", "El gato se sentó y durmió.", ""],
        "B_F": ["The cat sat on the rug.", "The cat sat and slept.", ""],
        "finish_F": ["stop", "length", ""], "finish_B_F": ["stop", "stop", ""],
        "bad_reason": ["", "F:truncated", "F:empty"]})
    path = tmp_path / "chain.csv"
    csv.to_csv(path, index=False)
    rows, cov = legacy_qwen.import_qwen_chain(path, man, CFG, TCFG, "free")
    assert cov["found"] == 3 and cov["missing"] == []
    gen, _ = schema.upsert(schema.empty_table(), rows, man, allow_invalid_predecessor=True)
    g = gen.set_index(["item_id", "output_stage"])
    assert g.loc[(man.item_id[0], "S1"), "validation_status"] == "valid"
    assert g.loc[(man.item_id[0], "S1"), "input_text"] == t0[man.item_id[0]]
    assert "legacy:truncated" in g.loc[(man.item_id[1], "S1"), "validation_reason"]
    assert "legacy:upstream_invalid" in g.loc[(man.item_id[1], "T1"), "validation_reason"]
    assert (man.item_id[2], "T1") not in g.index                        # no back translation of a blank S1
    assert set(gen.translator) == {"qwen3-4b-vllm-legacy"}
    # the transformers pathway is unaffected and starts from T0
    r = make_runner(man, tmp_path / "w", translator="qwen3-4b", pv="free", base=gen)
    assert r.plan().set_index("stage").loc["S1", "ready_now"] == 3
    with pytest.raises(T.ImportOnlyTranslatorError):
        T.build_translator(TCFG, "qwen3-4b-vllm-legacy", "free")
