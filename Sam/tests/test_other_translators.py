"""NLLB, Qwen and Llama: preflight selection and deployments, independent six-stage chains from T0, full-run
planning from measured preflight timings, adapter details (fixed chat-template date, neutral decoding, NLLB
embedding tie), the launch queue and the status overview. CPU only, no model."""
import contextlib
import importlib.util
import json
import sys
import types

import pandas as pd
import pytest

from sot_mt import paths, runner, schema, session, smoke
from sot_mt import translators as T
from sot_mt.files import load_yaml, sha256_file

SAM = paths.SAM_ROOT
TCFG, CFG = load_yaml("translators.yaml"), load_yaml("experiment.yaml")
MODELS = ["nllb", "qwen3-4b", "llama", "phi4-mini"]
PV = {"nllb": "mt-default", "qwen3-4b": "free", "llama": "free", "phi4-mini": "free"}


def _script(name):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def real():
    if not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("staged data unavailable")
    return session.load_inputs()


@pytest.fixture(autouse=True)
def no_loaded_models():
    T._LOADED.clear()
    yield
    T._LOADED.clear()


# -- the same three original documents for every model ----------------------------------------------------

def test_preflight_documents_are_three_t0_lengths_from_distinct_sources(real):
    manifest, _ = real
    ids = pd.read_csv(SAM / "data" / "manifests" / "preflight3.csv")
    assert ids.equals(smoke.select_preflight_items(manifest))           # deterministic, chosen from T0 only
    assert list(ids.length_band) == ["short", "medium", "long"] and ids.t0_words.is_monotonic_increasing
    assert ids.t0_words.iloc[-1] == manifest.t0_words.max()
    assert ids.source_label.is_unique and ids.article_id.is_unique
    m = manifest.set_index("item_id")
    assert (ids.group_id.values == m.loc[ids.item_id, "group_id"].values).all()


@pytest.mark.parametrize("model", MODELS)
def test_preflight_deployment(real, tmp_path, model):
    mr, prep = _script("model_runs"), _script("prepare_kaggle_deploy")
    P = mr.preflight_preset(model)
    d = prep.prepare(P["user"], P["slug"], tmp_path / "dep", P["overrides"], P["timeout_s"], P["resume_kernels"],
                     build_out=tmp_path / "build", results_dir=P["results_dir"], generated_root=P["generated_root"],
                     model_sources=P["model_sources"])
    meta = json.loads((tmp_path / "dep" / "kernel-metadata.json").read_text())
    assert meta["id"] == f"samahmed17/sam-sot-mt-{model}-preflight" and meta["is_private"] and meta["enable_gpu"]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["kernel_sources"] == []
    assert meta["model_sources"] == (["metaresearch/llama-3.2/transformers/3b-instruct/1"] if model == "llama" else [])
    assert d["results_dir"] == f"results/{model}/preflight" and d["model_sources"] == meta["model_sources"]
    cfg = "".join(next(c for c in json.loads((tmp_path / "dep" / "00_generate_translations.ipynb").read_text())
                       ["cells"] if c["cell_type"] == "code")["source"])
    for line in [f'TRANSLATOR = "{model}"', 'ITEM_IDS_FILE = "data/manifests/preflight3.csv"', "ROUNDS = (1, 3)",
                 "SANITY_PROBE = True", "USE_HF_TOKEN = False", "EXPORT_EVERY = 6", "MOCK = False",
                 "EXPECTED_PREVIOUS_MERGED_SHA256 = None"]:
        assert line in cfg, line


def test_prepared_preflight_folders_exist():
    for model in MODELS:
        d = SAM / "kaggle" / model / "preflight"
        dep = json.loads((d / "deploy.json").read_text())
        assert dep["kernel"] == f"samahmed17/sam-sot-mt-{model}-preflight"   # launch state is not asserted


# -- independent chains from T0 --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_preflights(real, tmp_path_factory):
    manifest, staged = real
    ids = smoke.read_item_ids(SAM / "data" / "manifests" / "preflight3.csv")
    out = {}
    for model in MODELS:
        work = tmp_path_factory.mktemp(model)
        r = runner.Runner(runner.RunSettings(model, PV[model], item_ids=ids, export_every=6, export_dir=str(work / "export")),
                          T.MockTranslator(TCFG, model, PV[model]), manifest, staged, work / "w", CFG, TCFG,
                          log=lambda *a: None)
        out[model] = (r.run(), r.table, work)
    return out


def test_each_model_runs_six_stages_from_t0_on_its_own_pathway(real, mock_preflights):
    manifest, staged = real
    t0 = manifest.set_index("item_id").t0_text
    for model, (res, table, _) in mock_preflights.items():
        assert res["operations"] == 18 and res["valid"] == 18
        pw = table[table.translator == model]
        assert len(pw) == 18 and set(pw.prompt_variant) == {PV[model]}
        s1 = pw[pw.output_stage == "S1"]
        assert (s1.input_stage == "T0").all() and (s1.input_text.values == t0.loc[s1.item_id].values).all()
        others = set(table[table.translator != model].output_text)
        assert not set(pw.input_text) & (others - set(t0))                # never another model's text
        assert set(pw.generation_config_id) == {T.generation_config(TCFG, model, PV[model])[0]}
        assert schema.chain_violations(table, manifest) == []
        assert table[table.translator == "madlad-3b"].output_sha256.sort_values().tolist() == \
            staged.output_sha256.sort_values().tolist()                   # MADLAD Stage 2 untouched


def test_generalised_verifier_checks_t0_lineage_and_flags_mock(real, mock_preflights):
    vp = _script("verify_madlad_pilot")
    _, _, work = mock_preflights["nllb"]
    with pytest.raises(vp.IntegrityError, match="MOCK"):
        vp.verify(work / "export", translator="nllb")


# -- full runs: planned from the measured preflight -----------------------------------------------------

def test_full_run_refuses_without_a_downloaded_preflight():
    mr = _script("model_runs")
    for model in MODELS:
        if not (SAM / mr.export_dir(model, "preflight") / runner.RESUME).exists():
            with pytest.raises(FileNotFoundError, match="not downloaded"):
                mr.full_preset(model)


def _fake_downloaded_preflight(mr, model, work, tmp_path, a=3.0, b=0.08):
    """A preflight export with known per-operation seconds = a + b * words, and its progress events."""
    exp, res = tmp_path / "exp", tmp_path / "res"
    exp.mkdir(), res.mkdir()
    d = pd.read_parquet(work / "export" / runner.DELTA)
    d["generation_meta"] = [json.dumps({**json.loads(m), "seconds": a + b * len(t.split())})
                            for m, t in zip(d.generation_meta, d.input_text)]
    d.to_parquet(exp / runner.DELTA)
    for name in (runner.MERGED, runner.RESUME):
        (exp / name).write_bytes((work / "export" / name).read_bytes())
    runs = pd.read_parquet(work / "export" / runner.RUNS)
    runs["load_report"] = json.dumps({"weights_id": "abc123"})
    runs.to_parquet(exp / runner.RUNS)
    (res / "progress_events.jsonl").write_text(
        json.dumps({"event": "notebook_start", "ts": "2026-10-31T10:00:00Z"}) + "\n" +
        json.dumps({"event": "run_start", "ts": "2026-10-31T10:05:00Z"}) + "\n")
    return exp, res


def test_full_run_timeout_comes_from_the_measured_preflight(real, mock_preflights, tmp_path, monkeypatch):
    mr, prep = _script("model_runs"), _script("prepare_kaggle_deploy")
    exp, res = _fake_downloaded_preflight(mr, "qwen3-4b", mock_preflights["qwen3-4b"][2], tmp_path)
    monkeypatch.setattr(mr, "export_dir", lambda m, k, a=1: str(exp) if k == "preflight" else f"data/generated/x/{k}")
    monkeypatch.setattr(mr, "results_dir", lambda m, k, a=1: str(res) if k == "preflight" else f"results/x/{k}")
    est = mr.estimate("qwen3-4b")
    assert (est["fit_a_s"], est["fit_b_s_per_word"]) == (pytest.approx(3.0, abs=1e-6), pytest.approx(0.08, abs=1e-6))
    assert est["setup_s"] == 300 and est["remaining_items"] == 197 and est["remaining_operations"] == 1182
    assert est["timeout_s"] <= mr.SESSION_LIMIT_S
    assert est["time_limit_minutes"] * 60 + est["setup_s"] + mr.MARGIN_S <= est["timeout_s"]
    if est["est_sessions"] == 1:
        assert est["timeout_s"] >= est["est_translate_s"] + est["setup_s"]
    P = mr.full_preset("qwen3-4b")
    assert P["resume_kernels"] == ["samahmed17/sam-sot-mt-qwen3-4b-preflight"] and P["resume_exports"] == [str(exp)]
    o = {**P["overrides"], "EXPECTED_PREVIOUS_MERGED_SHA256": json.dumps([sha256_file(exp / runner.MERGED)])}
    for k, v in [("ITEM_IDS_FILE", "None"), ("MAX_ITEMS", "None"), ("EXPORT_EVERY", "20"),
                 ("EXPECTED_WEIGHTS_ID", '"abc123"'), ("SANITY_PROBE", "True")]:
        assert o[k] == v, k
    d = prep.prepare(P["user"], P["slug"], tmp_path / "dep", o, P["timeout_s"], P["resume_kernels"],
                     build_out=tmp_path / "build", results_dir=P["results_dir"], generated_root=P["generated_root"])
    assert d["kernel"] == "samahmed17/sam-sot-mt-qwen3-4b-full"
    P2 = mr.full_preset("qwen3-4b", attempt=2)
    assert P2["slug"] == "sam-sot-mt-qwen3-4b-full-r2" and P2["resume_kernels"] == ["samahmed17/sam-sot-mt-qwen3-4b-full"]


def test_full_run_resumes_from_the_preflight_without_redoing_it(real, mock_preflights, tmp_path):
    manifest, _ = real
    _, _, work = mock_preflights["llama"]
    base = runner.load_tables([work / "export" / runner.MERGED], manifest)
    r = runner.Runner(runner.RunSettings("llama", "free"), T.MockTranslator(TCFG, "llama", "free"), manifest, base,
                      tmp_path / "w", CFG, TCFG, log=lambda *a: None)
    plan, _ = r.progress_plan()
    assert sum(len(v) for v in plan.values()) == 1200 - 18                # all 200 documents, minus the preflight


# -- adapter details --------------------------------------------------------------------------------------

class _Tok:
    pad_token_id, eos_token_id = None, 9

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(kw)
        return "PROMPT " + messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def __call__(self, prompt, **kw):
        n = len(prompt.split())
        return types.SimpleNamespace(input_ids=types.SimpleNamespace(shape=(1, n)), to=lambda dev: _Enc(n))

    def decode(self, ids, skip_special_tokens=True):
        return "Hola mundo."


class _Enc(dict):
    def __init__(self, n):
        super().__init__()
        self.input_ids = types.SimpleNamespace(shape=(1, n))


class _Out:
    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, key):
        return types.SimpleNamespace(tolist=lambda: self.ids[key[1]])


class _Model:
    device = "cuda:0"
    generation_config = types.SimpleNamespace(eos_token_id=[9])

    def __init__(self):
        self.kw = None

    def generate(self, **kw):
        self.kw = kw
        return _Out([0] * 3 + [5, 9])


def test_chat_adapter_fixes_the_template_date_and_neutralises_penalties(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(inference_mode=contextlib.nullcontext))
    a = T.build_translator(TCFG, "llama", "free")
    a.tokenizer, a.model = _Tok(), _Model()
    res = a.translate("Hello world.", "en", "es")
    assert res.finish == "stop" and res.text == "Hola mundo."
    assert a.tokenizer.calls[0]["date_string"] == "26 Jul 2024"
    kw = a.model.kw
    assert kw["do_sample"] is False and kw["repetition_penalty"] == 1.0 and kw["no_repeat_ngram_size"] == 0
    q = T.build_translator(TCFG, "qwen3-4b", "free")
    q.tokenizer, q.model = _Tok(), _Model()
    q.translate("Hello world.", "en", "es")
    assert "date_string" not in q.tokenizer.calls[0]                    # Qwen's template has no date


def test_nllb_embedding_check_refuses_untied_weights():
    W = types.SimpleNamespace
    shared = W(weight="A")
    tied = W(config=W(tie_word_embeddings=True), get_input_embeddings=lambda: shared,
             get_encoder=lambda: W(embed_tokens=shared), get_decoder=lambda: W(embed_tokens=shared),
             get_output_embeddings=lambda: W(weight="A"))
    eq = lambda x, y: x == y  # noqa: E731
    assert all(v for v in T.seq2seq_embedding_report(tied, eq).values())
    broken = W(**{**tied.__dict__, "get_decoder": lambda: W(embed_tokens=W(weight="B"))})
    rep = T.seq2seq_embedding_report(broken, eq)
    assert rep["decoder_equals_shared"] is False and rep["encoder_equals_shared"] is True


def test_madlad_config_is_unchanged():
    assert T.generation_config(TCFG, "madlad-3b", "mt-default")[0] == "705a8bd4992e"


# -- queue and status ------------------------------------------------------------------------------------

def test_status_overview_offline():
    st = _script("status_all")
    rows = [st.local_state(d) for d in st.deployments()]
    assert {r["model"] for r in rows} >= set(MODELS)
    text = st.table(rows)
    assert "not launched" in text and "preflight" in text
