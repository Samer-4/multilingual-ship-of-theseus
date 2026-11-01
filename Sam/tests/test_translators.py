import copy

import pytest

from sot_mt import translators as T
from sot_mt.files import load_yaml

TCFG = load_yaml("translators.yaml")
STAGE2_MADLAD_CONFIG_ID = "705a8bd4992e"     # carried by all 400 Stage 2 records


@pytest.fixture(autouse=True)
def no_loaded_models():
    T._LOADED.clear()
    yield
    T._LOADED.clear()


def words(s):
    return len(s.split()) + 1


class FakeMadlad(T.MadladTranslator):
    """MADLAD adapter logic with the two model-touching methods replaced."""

    def __init__(self, behaviour=None, **kw):
        super().__init__(TCFG, "madlad-3b", "mt-default", **kw)
        self.behaviour, self.batches = behaviour or (lambda batch, call: None), []

    def count_tokens(self, text, src, tgt):
        return words(text)

    def _generate(self, segments, src, tgt):
        self.batches.append(list(segments))
        b = self.behaviour(segments, len(self.batches))
        if isinstance(b, Exception):
            raise b
        return b if b is not None else [(f"<{tgt}>{s}", True, len(s.split())) for s in segments]


class OutOfMemoryError(RuntimeError):          # same class name as torch.cuda.OutOfMemoryError
    pass


# -- configs and construction ---------------------------------------------------------------------

def test_madlad_config_id_is_unchanged_from_stage2():
    assert T.generation_config(TCFG, "madlad-3b", "mt-default")[0] == STAGE2_MADLAD_CONFIG_ID


def test_config_ids_are_stable_and_distinct():
    ids = {(t, pv): T.generation_config(TCFG, t, pv)[0]
           for t, e in TCFG["translators"].items() for pv in e["prompt_variants"]}
    assert len(set(ids.values())) == len(ids)
    assert ids == {k: T.generation_config(copy.deepcopy(TCFG), *k)[0] for k in ids}
    changed = copy.deepcopy(TCFG)
    changed["translators"]["nllb"]["settings"]["batch_size"] = 4
    assert T.generation_config(changed, "nllb", "mt-default")[0] != ids[("nllb", "mt-default")]
    with pytest.raises(KeyError):
        T.generation_config(TCFG, "qwen3-4b", "mt-default")


def test_adapters_and_identity():
    for name, pv, cls in [("madlad-3b", "mt-default", T.MadladTranslator), ("nllb", "mt-default", T.NllbTranslator),
                          ("qwen3-4b", "free", T.ChatTranslator), ("qwen3-4b", "literal", T.ChatTranslator)]:
        a = T.build_translator(TCFG, name, pv)
        assert isinstance(a, cls) and a.model is None                       # constructed, not loaded
        assert a.identity()["model_checkpoint"] == TCFG["translators"][name]["model_checkpoint"]
        assert a._hf_kwargs()["local_files_only"] is True                   # no downloads unless enabled
    assert T.build_translator(TCFG, "nllb", "mt-default", allow_download=True)._hf_kwargs()["local_files_only"] is False


def test_llama_hub_route_requires_explicit_opt_in_and_token(monkeypatch):
    """The hub route (meta-llama/... gated on Hugging Face): opt-in AND a token, never implicitly."""
    tcfg = copy.deepcopy(TCFG)
    tcfg["translators"]["llama"]["requires_hf_token"] = True
    del tcfg["translators"]["llama"]["settings"]["weights"]
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(T.MissingCredentialsError, match="USE_HF_TOKEN"):
        T.build_translator(tcfg, "llama", "free")
    with pytest.raises(T.MissingCredentialsError, match="no HF_TOKEN"):
        T.build_translator(tcfg, "llama", "free", use_hf_token=True)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    with pytest.raises(T.MissingCredentialsError):
        T.build_translator(tcfg, "llama", "free")                       # token present but not enabled
    a = T.build_translator(tcfg, "llama", "free", use_hf_token=True)
    assert a.token == "hf_dummy" and a.model is None and a._hf_kwargs()["token"] == "hf_dummy"


def test_llama_loads_from_the_attached_kaggle_model_without_a_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    a = T.build_translator(TCFG, "llama", "free")
    assert a.token is None and a._hf_kwargs() == {"local_files_only": True}
    monkeypatch.setenv("SAM_KAGGLE_INPUT", str(tmp_path))
    with pytest.raises(T.ModelNotAttachedError, match="licence"):
        a.resolve_load_path()
    mount = tmp_path / "llama-3.2" / "transformers" / "3b-instruct" / "1"
    mount.mkdir(parents=True)
    (mount / "config.json").write_text("{}")
    assert a.resolve_load_path() == str(mount)
    assert len(T.weights_fingerprint(str(mount))) == 16
    assert TCFG["translators"]["llama"]["settings"]["chat_template_kwargs"] == {"date_string": "26 Jul 2024"}


def test_import_only_and_unknown_translators():
    with pytest.raises(T.ImportOnlyTranslatorError):
        T.build_translator(TCFG, "qwen3-4b-vllm-legacy", "free")
    with pytest.raises(KeyError):
        T.build_translator(TCFG, "gpt-x", "free")


def test_only_one_model_loaded_at_a_time():
    a = T.MockTranslator(TCFG, "nllb", "mt-default")
    b = T.MockTranslator(TCFG, "madlad-3b", "mt-default")
    a.load()
    with pytest.raises(T.ModelAlreadyLoadedError):
        b.load()
    a.unload()
    b.load()
    c = T.MockTranslator(TCFG, "qwen3-4b", "free", allow_multiple_models=True)
    c.load()                                                               # explicitly configured
    assert set(T._LOADED) == {"madlad-3b", "qwen3-4b"}


# -- seq2seq logic --------------------------------------------------------------------------------

def test_madlad_segments_batches_and_joins():
    t = FakeMadlad()
    text = " ".join(f"Sentence {i} has a few words." for i in range(60))     # 6 words each, budget 300
    r = t.translate(text, "en", "es")
    segs = r.meta["segmentation"]
    assert r.finish == "stop" and r.error is None
    assert segs["method"] == "madlad_pack" and segs["budget_tokens"] == 300
    assert all(n <= 300 for n in segs["segment_input_tokens"])
    assert r.text == " ".join(f"<es>{s}" for b in t.batches for s in b)
    assert all(len(b) <= 4 for b in t.batches)                                 # worker batch size
    assert r.meta["segment_hit_eos"] == [True] * segs["n_segments"]


def test_missing_eos_marks_generation_cap():
    t = FakeMadlad(lambda batch, call: [(f"x{i}", i != 0, 512) for i in range(len(batch))])
    assert t.translate("One. Two.", "en", "es").finish == "length"


def test_empty_segment_is_a_deterministic_error_without_partial_text():
    t = FakeMadlad(lambda batch, call: [("", True, 1)] + [("ok", True, 1)] * (len(batch) - 1))
    r = t.translate("One. Two.", "en", "es")
    assert r.error == "empty_segment" and r.text is None


def test_segmentation_error_is_deterministic():
    t = FakeMadlad()
    t.count_tokens = lambda text, src, tgt: len(text)            # characters: a long word cannot fit
    r = t.translate("x" * 400, "en", "es")
    assert r.error == "segmentation_error" and r.text is None


def test_oom_halves_batch_then_raises_transient():
    t = FakeMadlad(lambda batch, call: OutOfMemoryError("CUDA out of memory") if len(batch) > 2 else None)
    r = t.translate(" ".join(f"S{i} words here." for i in range(300)), "en", "es")
    assert r.meta["oom_retries"] == 1 and max(r.meta["batch_sizes"]) == 2
    t = FakeMadlad(lambda batch, call: OutOfMemoryError("CUDA out of memory"))
    with pytest.raises(T.TransientTranslationError, match="cuda_oom"):
        t.translate("One. Two.", "en", "es")
    t = FakeMadlad(lambda batch, call: ValueError("other failure"))
    with pytest.raises(ValueError):
        t.translate("One. Two.", "en", "es")


def test_nllb_language_codes_and_sentence_mode():
    a = T.build_translator(TCFG, "nllb", "mt-default")
    assert a.segmentation_method == "sentence" and a.budget == 256
    assert (a._code("en"), a._code("es")) == ("eng_Latn", "spa_Latn")

    class Tok:
        src_lang = None

        def __call__(self, text):
            return type("E", (), {"input_ids": [0] * (len(text.split()) + 2)})()
    a.tokenizer = Tok()
    assert a.count_tokens("hola mundo", "es", "en") == 4 and a.tokenizer.src_lang == "spa_Latn"


# -- chat LLM logic -------------------------------------------------------------------------------

class FakeChat(T.ChatTranslator):
    def __init__(self, out, **kw):
        super().__init__(TCFG, "qwen3-4b", "free", **kw)
        self.out, self.seen = out, []

    def _generate(self, system, text):
        self.seen.append(system)
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def test_chat_prompts_think_stripping_and_finish():
    t = FakeChat(("<think>plan</think>\nHola mundo.", True, 3, 40, 5, 70))
    r = t.translate("Hello world.", "en", "es")
    assert r.text == "Hola mundo." and r.finish == "stop" and r.meta["think_stripped"]
    assert t.seen == [TCFG["translators"]["qwen3-4b"]["prompt_variants"]["free"]["system_en_es"]]
    t.translate("Hola.", "es", "en")
    assert t.seen[-1] == TCFG["translators"]["qwen3-4b"]["prompt_variants"]["free"]["system_es_en"]
    assert FakeChat(("Hola", False, 3, 40, 70, 70)).translate("Hi", "en", "es").finish == "length"
    with pytest.raises(T.TransientTranslationError):
        FakeChat(OutOfMemoryError("out of memory")).translate("Hi", "en", "es")


def test_qwen_and_llama_free_prompts_are_identical():
    q = TCFG["translators"]["qwen3-4b"]["prompt_variants"]["free"]
    assert TCFG["translators"]["llama"]["prompt_variants"]["free"] == q
    assert TCFG["translators"]["qwen3-4b-vllm-legacy"]["prompt_variants"]["free"] == q


def test_legacy_max_new_tokens_rule_and_messages():
    assert T.llm_max_new_tokens(100, 150, 8192) == 264
    assert T.llm_max_new_tokens(5000, 5000, 8192) == 8192 - 5000 - 8
    assert T.llm_max_new_tokens(0, 8190, 8192) == 16
    assert T.chat_messages("S", "U") == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert T.chat_messages("S", "U", system_role=False) == [{"role": "user", "content": "S\n\nU"}]
    assert T.is_oom(OutOfMemoryError("x")) and T.is_oom(RuntimeError("CUDA out of memory")) and not T.is_oom(ValueError())
