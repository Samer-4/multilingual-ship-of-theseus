"""The verified MADLAD wiring under transformers 5.x (one-document Kaggle diagnostic, kernel
samahmed17/sam-sot-mt-madlad-verify1 v1): encoder, decoder and `shared` must all use the checkpoint's
decoder.embed_tokens; lm_head stays as loaded. CPU only; the diagnostic file is the real Kaggle output."""
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from sot_mt import checks, paths, runner, session, smoke
from sot_mt import translators as T
from sot_mt.files import load_yaml, sha256_file

SAM = paths.SAM_ROOT
TCFG = load_yaml("translators.yaml")
CFG = load_yaml("experiment.yaml")
DIAG = SAM / "results" / "madlad_verify1" / "failed_v1" / "madlad_diagnostic.json"
EQ = lambda a, b: a == b                                                     # noqa: E731


@pytest.fixture(scope="module")
def staged():
    if not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("staged Stage 2 data unavailable")
    return session.load_inputs()


@pytest.fixture(scope="module")
def diag():
    if not DIAG.exists():
        pytest.skip("Kaggle diagnostic not downloaded")
    return json.loads(DIAG.read_text())


class Emb:
    def __init__(self, values):
        self.weight = list(values)


def fake_t5():
    """As transformers 5.16.1 loaded MADLAD: encoder tied to shared, decoder separate, lm_head equal to shared."""
    sh = Emb((1, 2, 3))
    return types.SimpleNamespace(config=types.SimpleNamespace(tie_word_embeddings=True, _commit_hash="fa184c6"),
                                 shared=sh, encoder=types.SimpleNamespace(embed_tokens=sh),
                                 decoder=types.SimpleNamespace(embed_tokens=Emb((9, 9, 9))),
                                 lm_head=types.SimpleNamespace(weight=sh.weight))


def fake_libs(monkeypatch, model):
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


@pytest.fixture
def loaded(monkeypatch):
    m = fake_t5()
    fake_libs(monkeypatch, m)
    a = T.build_translator(TCFG, "madlad-3b", "mt-default")
    a.load()
    yield a, m
    a.unload()


# -- 1. the verified configuration is the one selected ------------------------------------------------------

def test_diagnostic_evidence_selects_decoder(diag, staged):
    manifest, _ = staged
    t0 = manifest.set_index("item_id").t0_text[diag["item_id"]]
    v = diag["variants"]
    assert diag["transformers"] == "5.16.1" and diag["item_id"] == "xsum|LLAMA|xsum-4729"
    assert diag["load_report_as_loaded"]["decoder_equals_shared"] is False
    assert diag["load_report_as_loaded"]["encoder_equals_shared"] is True
    assert diag["load_report_as_loaded"]["lm_head_equals_shared"] is True
    # judged again with today's checks and the raised threshold, not taken from the stored verdicts
    judged = {k: smoke.probe_verdict(t0, diag["round1_S1"], x["text"], x["finish"], CFG["validation"], 0.8)
              for k, x in v.items()}
    assert [k for k, j in judged.items() if j["ok"]] == ["decoder"]
    assert judged["decoder"]["identical"] and v["decoder"]["finish"] == "stop"
    assert judged["as_loaded"]["blocking"].startswith("engine:generation_cap")
    assert "check:not_target_language" in judged["shared"]["blocking"]       # EOS-terminated garbage now caught


def test_config_selects_the_verified_wiring_without_touching_the_frozen_entry():
    patch = TCFG["runtime_patches"]["madlad-3b"]
    assert patch["embedding_tie"] == "decoder" and patch["lm_head"] == "unchanged"
    assert "runtime_patches" not in TCFG["translators"]["madlad-3b"]
    assert T.generation_config(TCFG, "madlad-3b", "mt-default")[0] == "705a8bd4992e"
    assert T.build_translator(TCFG, "madlad-3b", "mt-default").embedding_tie == "decoder"
    assert T.build_translator({**TCFG, "runtime_patches": {}}, "madlad-3b", "mt-default").embedding_tie == "none"


# -- 2. model state across diagnostics ------------------------------------------------------------------

def test_load_applies_decoder_wiring_and_keeps_lm_head(loaded):
    a, m = loaded
    assert m.shared is m.encoder.embed_tokens is m.decoder.embed_tokens and m.shared.weight == [9, 9, 9]
    assert m.lm_head.weight == [1, 2, 3]                                      # output side untouched
    assert a.wiring == "decoder" and a.load_report["applied"] == "decoder"


def test_rewire_restores_the_loaded_modules_before_each_variant(loaded):
    a, m = loaded
    lm = m.lm_head.weight
    a.rewire("none")
    assert m.encoder.embed_tokens.weight == [1, 2, 3] and m.decoder.embed_tokens.weight == [9, 9, 9]
    assert m.decoder.embed_tokens is not m.shared
    a.rewire("decoder")
    a.rewire("shared")                                     # after decoder: still the ORIGINAL shared values
    assert m.shared is m.decoder.embed_tokens and m.shared.weight == [1, 2, 3]
    a.rewire("none")
    assert m.decoder.embed_tokens.weight == [9, 9, 9]      # nothing leaked from the previous variants
    a.rewire(a.embedding_tie)
    assert a.wiring == "decoder" and m.shared.weight == [9, 9, 9] and m.lm_head.weight is lm


def test_rewire_refuses_unknown_modes_and_a_changed_lm_head(loaded):
    a, m = loaded
    with pytest.raises(T.EngineWiringError):
        a.rewire("encoder")
    m.lm_head.weight = [0, 0, 0]
    with pytest.raises(T.EngineWiringError, match="lm_head"):
        a.rewire("decoder")


def test_unverified_wiring_cannot_translate(loaded):
    a, _ = loaded
    a.wiring = None
    with pytest.raises(T.EngineWiringError):
        a.translate("Hello.", "en", "es")
    a.unload()
    with pytest.raises(T.EngineWiringError):
        a.rewire("decoder")


# -- 3. existing MADLAD T1 results are preserved -------------------------------------------------------

def test_stage2_data_and_t1_are_unchanged(staged):
    manifest, base = staged
    info = session.verify_staged(paths.staged_dir())
    assert sha256_file(paths.staged_dir() / "generations.parquet").startswith("a636b3edc37b115a")
    assert info["files"]["generations.parquet"].startswith("a636b3edc37b115a")
    t1 = base[(base.translator == "madlad-3b") & (base.output_stage == "T1")]
    assert len(t1) == 200 and (t1.validation_status == "valid").sum() == 197
    assert set(t1.generation_config_id) == {"705a8bd4992e"} and t1.generation_meta.isna().all()


def test_new_language_check_passes_every_real_stage2_output_and_t0(staged):
    manifest, base = staged
    mad = base[(base.translator == "madlad-3b") & base.output_text.notna()]
    for r in mad.itertuples():
        out = checks.check_step(r.input_text, r.output_text, r.output_language, CFG["validation"])
        assert "check:not_target_language" not in out["blocking"], r.record_id
    for t in manifest.t0_text:
        assert "check:not_target_language" not in checks.check_step("x", t, "en", CFG["validation"])["blocking"]


# -- 4. generation metadata and configuration tracking ---------------------------------------------------

class ScriptedMadlad(T.MadladTranslator):
    """Real MadladTranslator (load, wiring, segmentation, metadata) with the tokenizer and generate faked."""

    def count_tokens(self, text, src, tgt):
        return len(text.split()) + 2

    def _generate(self, segments, src, tgt):
        es = "Este es el texto número {} que se traduce con el modelo para la prueba del sistema."
        en = "This is the text number {} that was translated with the model for the test of the system."
        return [((es if tgt == "es" else en).format(i), True, 20) for i, _ in enumerate(segments)]


def test_records_and_runs_carry_the_runtime_patch(monkeypatch, staged, tmp_path):
    manifest, base = staged
    fake_libs(monkeypatch, fake_t5())
    tr = ScriptedMadlad(TCFG, "madlad-3b", "mt-default")
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=["xsum|LLAMA|xsum-4729"],
                                         stop_after="T2"), tr, manifest, base, tmp_path, CFG, TCFG, log=lambda *a: None)
    out = r.run()
    assert out["status"] == "completed" and out["valid"] == 2, out
    delta = pd.read_parquet(tmp_path / runner.DELTA)
    assert set(delta.generation_config_id) == {"705a8bd4992e"}
    for meta in delta.generation_meta.map(json.loads):
        assert meta["embedding_wiring"] == "decoder" and meta["runtime_patch"] == "madlad-embed-decoder-v1"
        assert meta["segmentation"]["method"] == "madlad_pack" and meta["segment_hit_eos"]
    load = json.loads(pd.read_parquet(tmp_path / runner.RUNS).load_report.iloc[0])
    assert load["applied"] == "decoder" and load["before"]["decoder_equals_shared"] is False
    s2 = delta[delta.output_stage == "S2"].iloc[0]
    t1 = base[(base.translator == "madlad-3b") & (base.output_stage == "T1") & (base.item_id == s2.item_id)].iloc[0]
    assert s2.input_text == t1.output_text                                    # continues from the stored T1


# -- 5/6. the gates -----------------------------------------------------------------------------------------

def test_reference_gate_accepts_the_verified_output(diag):
    v = smoke.probe_verdict("unused", diag["round1_S1"], diag["variants"]["decoder"]["text"], "stop",
                            CFG["validation"], 0.8)
    assert v["ok"] and v["similarity"] == 1.0 and v["blocking"] is None


@pytest.mark.parametrize("variant", ["as_loaded", "shared"])
def test_reference_gate_rejects_degenerate_output(diag, variant):
    x = diag["variants"][variant]
    v = smoke.probe_verdict("unused", diag["round1_S1"], x["text"], x["finish"], CFG["validation"], 0.8)
    assert not v["ok"] and v["similarity"] < 0.05 and v["blocking"]


def test_quality_gate_counts_wrong_script_garbage(staged, tmp_path):
    manifest, base = staged
    garbage = "εαααεόσwwnw semneknie roiες yty eryę εyssęs esetsses k wnsytyy wtyówówów řęężżż z ęεa ε łςσ dla żeε ες"

    class Garbage(T.MockTranslator):
        def translate(self, text, src, tgt):
            r = super().translate(text, src, tgt)
            r.text = garbage
            return r
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default",
                                         item_ids=smoke.read_item_ids(SAM / "data/manifests/stage3b_madlad_smoke5.csv")),
                      Garbage(TCFG, "madlad-3b", "mt-default"), manifest, base, tmp_path, CFG, TCFG, run_id="g",
                      log=lambda *a: None)
    out = r.run()
    assert out["status"] == "stopped:quality_gate" and out["operations"] == 3
    reasons = pd.read_parquet(tmp_path / runner.DELTA).validation_reason
    assert reasons.str.contains("check:not_target_language").all()


def test_verify1_notebook_runs_one_gated_reference_check(tmp_path, staged):
    import importlib.util
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location("prep", SAM / "scripts" / "prepare_kaggle_deploy.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    o = prep.PRESETS["verify1"]["overrides"]
    assert (o["MADLAD_DIAGNOSTIC"], o["DRIFT_PROBE"], o["REFERENCE_GATE"], o["MAX_ITEMS"]) == ("False", "True", "True", "1")
    nb = json.loads((SAM / "notebooks" / "00_generate_translations.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "REFERENCE_MIN_SIMILARITY = 0.8" in src
    assert "run.tr.rewire(variant)" in src and "run.tr.rewire(configured)" in src
    assert "if configured not in passing" in src and "'shared') does not" not in src
