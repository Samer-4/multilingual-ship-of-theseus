"""Regression tests for the first real MADLAD smoke run (Kaggle, transformers 5.16.1): every S2 and every T0 -> S1
probe degenerated into one repeated foreign-script token until the 512-token cap. Cause: 5.x builds T5's decoder
embedding as a separate module and skipped tying it to `shared` (the checkpoint stores different values), while
transformers 4.46.3 (round 1) builds encoder, decoder and `shared` around ONE module. CPU only: models are fakes."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from sot_mt import paths, runner, schema, session, smoke
from sot_mt import translators as T
from sot_mt.files import load_yaml

SAM = paths.SAM_ROOT
TCFG = load_yaml("translators.yaml")
CFG = load_yaml("experiment.yaml")
FAILED_RUN = "madlad-3b-mt-default-2026-10-31T052125Z-fca149"
FAILED_DELTA = SAM / "data" / "generated" / "xsum" / "madlad-3b" / "generations_delta.parquet"
FAILED_PROBE = SAM / "results" / "madlad_smoke5" / "drift_probe.csv"
SMOKE_FILE = SAM / "data" / "manifests" / "stage3b_madlad_smoke5.csv"


@pytest.fixture(scope="module")
def staged():
    if not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("staged Stage 2 data unavailable")
    return session.load_inputs()


# -- the evidence: the real failed outputs are recognised as degenerate -----------------------------------

def test_failed_run_outputs_are_degenerate_by_every_check(staged):
    if not FAILED_DELTA.exists():
        pytest.skip("failed smoke run not downloaded")
    d = pd.read_parquet(FAILED_DELTA)
    assert set(d.run_id) == {FAILED_RUN} and (d.validation_status == "invalid").all()
    for r in d.itertuples():
        meta = json.loads(r.generation_meta)
        assert meta["segment_hit_eos"] and not any(meta["segment_hit_eos"])        # never emitted EOS
        assert all(n == 513 for n in meta["segment_output_tokens"])                  # 512 new + start token
        letters = [ch for ch in r.output_text if not ch.isspace()]
        assert sum(ch.isascii() for ch in letters) == 0                              # no Latin script at all
        assert len(set(letters)) <= 6                                                 # a handful of repeated glyphs
    # the gate would have stopped this run after 3 operations
    s = runner.RunSettings("madlad-3b", "mt-default")
    recent = [any(c in reason for c in runner.DEGENERATE) for reason in d.validation_reason]
    w = recent[-s.gate_window:]
    assert len(w) >= s.gate_min_ops and sum(w) / len(w) >= s.gate_max_share


def test_failed_drift_probes_fail_the_reference_gate(staged):
    if not FAILED_PROBE.exists():
        pytest.skip("failed smoke run not downloaded")
    manifest, base = staged
    s1 = base[(base.translator == "madlad-3b") & (base.output_stage == "S1")].set_index("item_id")
    probe = pd.read_csv(FAILED_PROBE)
    for r in probe.itertuples():
        v = smoke.probe_verdict(s1.input_text[r.item_id], s1.output_text[r.item_id], r.probe_text, r.finish,
                                CFG["validation"])
        assert not v["ok"] and "engine:generation_cap" in v["blocking"] and v["similarity"] < 0.05


def test_probe_verdict_accepts_small_drift_and_rejects_degeneration():
    t0 = "The council approved the new budget on Monday after a long debate."
    r1 = "El consejo aprobó el nuevo presupuesto el lunes tras un largo debate."
    drift = "El consejo aprobó el nuevo presupuesto el lunes después de un largo debate."
    v = smoke.probe_verdict(t0, r1, drift, "stop", CFG["validation"])
    assert v["ok"] and not v["identical"] and v["similarity"] > 0.8
    assert smoke.probe_verdict(t0, r1, r1, "stop", CFG["validation"])["identical"]
    loop = "ง " * 300
    bad = smoke.probe_verdict(t0, r1, loop, "length", CFG["validation"])
    assert not bad["ok"] and "engine:generation_cap" in bad["blocking"]
    other = smoke.probe_verdict(t0, r1, "La lluvia cayó sobre la ciudad durante toda la noche de ayer.", "stop",
                                CFG["validation"])
    assert not other["ok"] and other["blocking"] is None                       # fluent but unrelated: similarity
    assert not smoke.probe_verdict(t0, r1, None, "error", CFG["validation"])["ok"]


# -- the fix: one shared input embedding, as round 1 -------------------------------------------------------

class Emb:
    def __init__(self, values):
        self.weight = list(values)


def fake_t5(shared=(1, 2, 3), encoder=None, decoder=(9, 9, 9)):
    """transformers-5.x-like layout: three separate embedding modules (encoder tied, decoder not)."""
    sh = Emb(shared)
    enc = sh if encoder is None else Emb(encoder)
    return types.SimpleNamespace(config=types.SimpleNamespace(tie_word_embeddings=False, _commit_hash="abc"),
                                 shared=sh, encoder=types.SimpleNamespace(embed_tokens=enc),
                                 decoder=types.SimpleNamespace(embed_tokens=Emb(decoder)),
                                 lm_head=Emb((5, 5, 5)))


EQ = lambda a, b: a == b                                                     # noqa: E731


def test_embedding_report_detects_the_untied_decoder():
    rep = T.t5_embedding_report(fake_t5(), equal=EQ)
    assert rep["encoder_is_shared"] and not rep["decoder_is_shared"]
    assert rep["decoder_equals_shared"] is False and rep["tie_word_embeddings"] is False
    assert rep["commit_hash"] == "abc" and rep["lm_head_equals_shared"] is False


@pytest.mark.parametrize("source,expected", [("shared", [1, 2, 3]), ("decoder", [9, 9, 9])])
def test_tie_makes_one_module_like_transformers_4_46(source, expected):
    m = fake_t5()
    T.tie_t5_input_embeddings(m, source)
    assert m.shared is m.encoder.embed_tokens is m.decoder.embed_tokens
    assert m.shared.weight == expected
    rep = T.t5_embedding_report(m, equal=EQ)
    assert rep["encoder_is_shared"] and rep["decoder_is_shared"]


def _fake_libs(monkeypatch, model):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    torch.float16 = torch.bfloat16 = torch.float32 = "dt"
    torch.equal = EQ
    tf = types.ModuleType("transformers")
    tf.__version__ = "5.16.1"
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: "tok")
    tf.AutoModelForSeq2SeqLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(to=lambda d: types.SimpleNamespace(eval=lambda: model)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)


def test_madlad_adapter_applies_the_verified_decoder_wiring_on_load(monkeypatch):
    m = fake_t5()
    lm = m.lm_head.weight
    _fake_libs(monkeypatch, m)
    a = T.build_translator(TCFG, "madlad-3b", "mt-default")
    assert a.embedding_tie == "decoder"                         # from translators.yaml: runtime_patches
    a.load()
    try:
        assert m.shared is m.encoder.embed_tokens is m.decoder.embed_tokens
        assert m.decoder.embed_tokens.weight == [9, 9, 9] and m.lm_head.weight is lm
        rep = a.load_report
        assert rep["applied"] == "decoder" and rep["runtime_patch"] == "madlad-embed-decoder-v1"
        assert rep["before"]["decoder_equals_shared"] is False and rep["after"]["decoder_is_shared"]
    finally:
        a.unload()


def test_already_one_module_is_left_alone(monkeypatch):
    m = fake_t5()
    T.tie_t5_input_embeddings(m, "shared")                       # what transformers 4.46.3 builds
    _fake_libs(monkeypatch, m)
    a = T.build_translator(TCFG, "madlad-3b", "mt-default")
    a.load()
    try:
        assert a.load_report["applied"] == "none" and a.load_report["already_one_module"]
        assert m.shared.weight == [1, 2, 3]
    finally:
        a.unload()


def test_frozen_madlad_config_is_unchanged():
    assert T.generation_config(TCFG, "madlad-3b", "mt-default")[0] == "705a8bd4992e"


# -- fail-fast quality gate --------------------------------------------------------------------------------

def _runner(staged, tmp_path, script, **settings):
    manifest, base = staged
    tr = T.MockTranslator(TCFG, "madlad-3b", "mt-default", script=script)
    return runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=smoke.read_item_ids(SMOKE_FILE),
                                            **settings), tr, manifest, base, tmp_path, CFG, TCFG, run_id="r",
                         log=lambda *a: None)


def test_quality_gate_stops_a_run_of_degenerate_outputs(staged, tmp_path):
    r = _runner(staged, tmp_path, lambda text, src, tgt, n: "loop")
    out = r.run()
    assert out["status"] == "stopped:quality_gate" and out["operations"] == 3 and out["invalid"] == 3
    assert pd.read_parquet(tmp_path / runner.RUNS).status.tolist() == ["stopped:quality_gate"]
    assert len(pd.read_parquet(tmp_path / runner.DELTA)) == 3      # the real outputs are still kept, invalid


def test_quality_gate_ignores_an_isolated_failure(staged, tmp_path):
    out = _runner(staged, tmp_path, {2: "loop"}).run()
    assert out["status"] == "completed" and out["invalid"] == 1 and out["operations"] == 17


def test_quality_gate_can_be_disabled(staged, tmp_path):
    out = _runner(staged, tmp_path, lambda text, src, tgt, n: "length", gate_min_ops=None).run()
    assert out["status"] == "completed" and out["operations"] == 5        # every chain stops at S2


# -- quarantine ------------------------------------------------------------------------------------------

def test_failed_run_is_quarantined_in_the_config():
    assert FAILED_RUN in CFG["quarantined_runs"]


def test_quarantined_records_cannot_seed_a_new_run(staged, tmp_path):
    manifest, base = staged
    if FAILED_DELTA.exists():
        bad = runner.load_tables([base, pd.read_parquet(FAILED_DELTA)], manifest)
    else:
        r = _runner(staged, tmp_path / "a", None)
        r.run_id = FAILED_RUN
        r.run()
        bad = runner.load_tables([base, pd.read_parquet(tmp_path / "a" / runner.DELTA)], manifest)
    with pytest.raises(runner.QuarantineError, match=FAILED_RUN):
        runner.Runner(runner.RunSettings("madlad-3b", "mt-default"), T.MockTranslator(TCFG, "madlad-3b", "mt-default"),
                      manifest, bad, tmp_path / "b", CFG, TCFG, log=lambda *a: None)


# -- notebook and one-document deployment -------------------------------------------------------------

def test_every_notebook_cell_compiles():
    nb = json.loads((SAM / "notebooks" / "00_generate_translations.ipynb").read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%")))
            compile(src, f"cell{i}", "exec")


def test_verify1_deployment(tmp_path, staged):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location("prep", SAM / "scripts" / "prepare_kaggle_deploy.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    P = prep.PRESETS["verify1"]
    d = prep.prepare(P["user"], P["slug"], tmp_path / "dep", P["overrides"], P["timeout_s"],
                     build_out=tmp_path / "build", results_dir=P["results_dir"], generated_root=P["generated_root"])
    assert d["kernel"] == "samahmed17/sam-sot-mt-madlad-verify1" and d["results_dir"] == "results/madlad_verify1"
    assert d["generated_root"] == "data/generated/verify1"         # never mixed with the quarantined smoke output
    nb = json.loads((tmp_path / "dep" / "00_generate_translations.ipynb").read_text())
    cfg = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    for line in ["MADLAD_DIAGNOSTIC = False", "DRIFT_PROBE = True", "REFERENCE_GATE = True", "MAX_ITEMS = 1",
                 'ITEM_IDS_FILE = "data/manifests/stage3b_madlad_verify1.csv"', "ALLOW_MODEL_DOWNLOAD = True"]:
        assert line in cfg, line
    ids = smoke.read_item_ids(SAM / "data" / "manifests" / "stage3b_madlad_verify1.csv")
    assert ids == ["xsum|LLAMA|xsum-4729"]
    manifest, base = staged
    mad = base[(base.translator == "madlad-3b") & base.item_id.isin(ids)]
    assert (mad.validation_status == "valid").all() and set(mad.output_stage) == {"S1", "T1"}
    assert all(json.loads(x)["chunks_en"] == 2 for x in mad.legacy_fields)          # multi-segment input
