"""Stage 3B preflight: smoke items, MADLAD tokenizer parity, GPU guard, transformers compatibility shims,
Kaggle packaging, and the notebook run end to end (MOCK) from the packaged folders under a fake
/kaggle/input. None of this loads a model; real model compatibility is only established on Kaggle."""
import importlib.util
import json
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from sot_mt import paths, schema, session, smoke
from sot_mt import translators as T
from sot_mt.files import load_yaml, sha256_text

SAM = paths.SAM_ROOT
TCFG = load_yaml("translators.yaml")
EXP = load_yaml("experiment.yaml")["experiment_id"]
SMOKE_FILE = SAM / "data" / "manifests" / "stage3b_madlad_smoke5.csv"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def staged():
    if not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("staged Stage 2 data unavailable")
    return session.load_inputs()


# -- smoke items --------------------------------------------------------------------------------------

def test_smoke_items_are_valid_distinct_and_span_lengths(staged):
    manifest, base = staged
    sel = smoke.select_smoke_items(base, EXP)
    saved = pd.read_csv(SMOKE_FILE)
    assert saved.item_id.tolist() == sel.item_id.tolist()           # file reproduces from the rule
    assert len(sel) == 5 and sel.source_label.is_unique and sel.article_id.is_unique
    assert sel.t1_words.is_monotonic_increasing and set(sel.length_band) == {"short", "medium", "long"}
    mad = base[base.translator == "madlad-3b"]
    invalid = set(mad[mad.validation_status == "invalid"].item_id)
    assert invalid == {"xsum|BigScience|xsum-4713", "xsum|LLAMA|xsum-4714", "xsum|LLAMA|xsum-4733"}
    assert not invalid & set(sel.item_id)
    st = mad[mad.item_id.isin(sel.item_id)].pivot(index="item_id", columns="output_stage", values="validation_status")
    assert (st[["S1", "T1"]] == "valid").all().all() and set(st.columns) == {"S1", "T1"}


def test_smoke_items_give_exactly_20_operations(staged, tmp_path):
    manifest, base = staged
    r = session.open_runner("madlad-3b", "mt-default", manifest, base, work_dir=tmp_path, mock=True,
                            item_ids=smoke.read_item_ids(SMOKE_FILE), log=lambda *a: None)
    plan = r.plan().set_index("stage")
    assert plan.loc["S1", "valid"] == plan.loc["T1", "valid"] == 5 and plan.loc["S2", "ready_now"] == 5
    out = r.run()
    assert (out["operations"], out["valid"] + out["invalid"]) == (20, 20)
    delta = pd.read_parquet(tmp_path / "generations_delta.parquet")
    assert set(delta.output_stage) == {"S2", "T2", "S3", "T3"} and set(delta.item_id) == set(r.items.item_id)
    s2 = delta[delta.output_stage == "S2"].set_index("item_id").input_text
    t1 = base[(base.translator == "madlad-3b") & (base.output_stage == "T1")].set_index("item_id").output_text
    assert all(s2[i] == t1[i] for i in s2.index)                    # S2 input = stored T1, byte for byte


def test_read_item_ids_rejects_duplicates(tmp_path):
    p = tmp_path / "ids.csv"
    pd.DataFrame({"item_id": ["a", "a"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        smoke.read_item_ids(p)


# -- MADLAD tokenizer parity --------------------------------------------------------------------------

def _parity_table(legacy_counts):
    rows = []
    for i, (en, es) in enumerate(legacy_counts):
        rows.append({"experiment_id": "exp", "translator": "madlad-3b", "prompt_variant": "mt-default",
                     "item_id": f"i{i}", "output_stage": "S1",
                     "input_text": "One two three. Four five six. Seven eight.",
                     "output_text": "Uno dos tres. Cuatro cinco.",
                     "legacy_fields": json.dumps({"chunks_en": en, "chunks_es": es})})
    return pd.DataFrame(rows)


def test_segment_parity_detects_a_different_tokenizer():
    words = lambda s, tgt: len(s.split())                            # noqa: E731
    # budget 4 words: English -> [3 words][3 words][2 words] = 3 segments; Spanish -> [3][2] = 2
    rep = smoke.madlad_segment_parity(_parity_table([(3, 2), (3, 1)]), words, 4, "exp")
    assert rep.match.tolist() == [True, True, True, False]
    assert rep.session_segments.tolist() == [3, 2, 3, 2]


def test_segment_parity_covers_every_round1_text(staged):
    manifest, base = staged
    rep = smoke.madlad_segment_parity(base, lambda s, tgt: len(s.split()), 300, EXP)
    assert len(rep) == 400 and rep.item_id.nunique() == 200          # both directions of all 200 items


# -- GPU guard, transformers compatibility --------------------------------------------------------------

def _fake_torch(cuda: bool):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None)
    torch.float16, torch.bfloat16, torch.float32 = "fp16", "bf16", "fp32"
    torch.equal = lambda x, y: x == y
    return torch


@pytest.mark.parametrize("translator,pv", [("madlad-3b", "mt-default"), ("nllb", "mt-default"), ("qwen3-4b", "free")])
def test_no_gpu_fails_clearly_before_any_model_load(monkeypatch, translator, pv):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setitem(sys.modules, "transformers", None)          # importing it would fail: must not be reached
    a = T.build_translator(TCFG, translator, pv)
    with pytest.raises(T.NoGPUError, match="Accelerator"):
        a.load()
    assert a.model is None and translator not in T._LOADED


def test_dtype_keyword_follows_the_transformers_version():
    assert T.dtype_kwarg("4.46.3") == "torch_dtype"                 # MADLAD round 1
    assert T.dtype_kwarg("4.55.4") == "torch_dtype"
    assert T.dtype_kwarg("4.56.0") == "dtype" and T.dtype_kwarg("5.16.1") == "dtype"


def fake_t5(label, shared_w=(1, 2), dec_w=(1, 2)):
    emb = lambda w: types.SimpleNamespace(weight=list(w))          # noqa: E731
    shared = emb(shared_w)
    return types.SimpleNamespace(label=label, config=types.SimpleNamespace(tie_word_embeddings=False),
                                 shared=shared, encoder=types.SimpleNamespace(embed_tokens=shared),
                                 decoder=types.SimpleNamespace(embed_tokens=emb(dec_w)))


def test_model_and_tokenizer_loading_kwargs(monkeypatch):
    calls = {}

    class Tok:
        @staticmethod
        def from_pretrained(name, **kw):
            calls["tok"] = (name, kw)
            return "tokenizer"

    class Model:
        @staticmethod
        def from_pretrained(name, **kw):
            calls["model"] = (name, kw)
            return types.SimpleNamespace(to=lambda dev: types.SimpleNamespace(eval=lambda: fake_t5(f"model@{dev}")))

    tf = types.ModuleType("transformers")
    tf.__version__, tf.AutoTokenizer, tf.AutoModelForSeq2SeqLM = "5.16.1", Tok, Model
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    monkeypatch.setitem(sys.modules, "transformers", tf)
    a = T.build_translator(TCFG, "madlad-3b", "mt-default")
    assert a.load_tokenizer() == "tokenizer" and "model" not in calls       # tokenizer alone: no weights
    a.load()
    assert a.model.label == "model@cuda:0"
    assert calls["tok"] == ("google/madlad400-3b-mt", {"local_files_only": True})
    assert calls["model"][1] == {"dtype": "fp16", "low_cpu_mem_usage": True, "local_files_only": True}
    a.unload()
    tf.__version__ = "4.46.3"
    T.build_translator(TCFG, "madlad-3b", "mt-default", allow_download=True).load()
    assert calls["model"][1] == {"torch_dtype": "fp16", "low_cpu_mem_usage": True, "local_files_only": False}
    T._LOADED.clear()


def test_madlad_config_matches_round1_worker():
    m = TCFG["translators"]["madlad-3b"]
    assert m["model_checkpoint"] == "google/madlad400-3b-mt"
    assert m["settings"]["decoding"] == {"num_beams": 1, "do_sample": False, "max_new_tokens": 512}
    assert m["settings"]["chunking"]["token_budget"] == 300 and m["settings"]["batch_size"] == 4
    assert m["settings"]["dtype"] == "float16"
    assert T.build_translator(TCFG, "madlad-3b", "mt-default").generation_config_id == "705a8bd4992e"


# -- packaging --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def package(tmp_path_factory, staged):
    pk = _load_script("package_kaggle_inputs")
    out = tmp_path_factory.mktemp("kaggle")
    return pk, pk.build(out, "tester", paths.staged_dir())


def test_package_contents_metadata_and_privacy(package):
    pk, built = package
    code, stg = built["code"], built["staged"]
    for rel in ["Sam/src/sot_mt/runner.py", "Sam/src/sot_mt/translators.py", "Sam/src/sot_mt/smoke.py",
                "Sam/configs/translators.yaml", "Sam/configs/experiment.yaml",
                "Sam/notebooks/00_generate_translations.ipynb", "Sam/data/manifests/stage3b_madlad_smoke5.csv",
                "Sam/scripts/run_translations.py", "Sam/requirements.txt"]:
        assert (code / rel).is_file(), rel
    assert not list(code.rglob("__pycache__")) and not list(code.rglob("*.parquet"))   # no text tables in code
    assert sorted(p.name for p in stg.iterdir()) == ["STAGED_FILES.json", "dataset-metadata.json",
                                                     "generations.parquet", "t0_manifest.parquet"]
    session.verify_staged(stg)
    for root, slug in [(code, "sam-sot-mt-code"), (stg, "sam-sot-mt-staged")]:
        meta = json.loads((root / "dataset-metadata.json").read_text())
        assert meta["id"] == f"tester/{slug}" and meta["licenses"] == [{"name": "unknown"}]
        assert 6 <= len(meta["title"]) <= 50 and 6 <= len(slug) <= 50     # Kaggle CLI limits
        assert "public" not in json.dumps(meta).lower().replace("not for redistribution", "")
    assert pk.scan_for_secrets(code) == [] and pk.scan_for_secrets(stg) == []


def test_secret_scan_blocks_credentials(tmp_path):
    pk = _load_script("package_kaggle_inputs")
    (tmp_path / "a.py").write_text('HF = "hf_' + "a" * 34 + '"\n')
    (tmp_path / "kaggle.json").write_text('{"username": "x", "key": "' + "0" * 32 + '"}')
    (tmp_path / "b.md").write_text("contact someone" + "@" + "example.edu\n")
    found = {(f, k) for f, k in pk.scan_for_secrets(tmp_path)}
    assert {("a.py", "huggingface_token"), ("kaggle.json", "credential file name"), ("kaggle.json", "kaggle_key"),
            ("b.md", "email")} <= found


def test_packaged_code_imports_on_its_own(package):
    _, built = package
    src = built["code"] / "Sam" / "src"
    prog = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from sot_mt import runner, session, smoke, translators, legacy_qwen, summaries; "
            "from sot_mt.files import load_yaml; "
            "t = load_yaml('translators.yaml'); "
            "print(translators.__file__); "
            "print(translators.generation_config(t, 'madlad-3b', 'mt-default')[0])")
    out = subprocess.run([sys.executable, "-I", "-c", prog, str(src)], capture_output=True, text=True,
                         cwd=built["code"], check=True).stdout.split()
    assert out[0].startswith(str(src)) and out[1] == "705a8bd4992e"


# -- the notebook, end to end, from the packaged folders (MOCK) --------------------------------------

def _notebook_script(overrides: str) -> str:
    nb = json.loads((SAM / "notebooks" / "00_generate_translations.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n\n".join(["display = print"] + cells[:1] + [overrides] + cells[1:])


@pytest.mark.parametrize("zipped", [False, True])
def test_notebook_runs_from_packaged_inputs(package, tmp_path, zipped):
    _, built = package
    inp = tmp_path / "input"
    if zipped:                       # Kaggle left Sam.zip unextracted: the notebook must unpack it itself
        (inp / "sam-sot-mt-code").mkdir(parents=True)
        with zipfile.ZipFile(inp / "sam-sot-mt-code" / "Sam.zip", "w") as z:
            for p in (built["code"] / "Sam").rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(built["code"]))
    else:
        shutil.copytree(built["code"], inp / "sam-sot-mt-code")
    shutil.copytree(built["staged"], inp / "sam-sot-mt-staged")
    staged_before = {p.name: sha256_text(p.read_bytes().hex()) for p in (inp / "sam-sot-mt-staged").iterdir()}
    out_dir = tmp_path / "working" / "sam_outputs"
    script = tmp_path / "nb.py"
    script.write_text(_notebook_script(f"MOCK = True\nOUTPUT_DIR = {str(out_dir)!r}\n")
                      + "\nprint('SOT_MT_FILE', sys.modules['sot_mt'].__file__)\n")
    env = {"SAM_KAGGLE_INPUT": str(inp), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    res = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "5 selected items" in res.stdout and "this run: 20 records" in res.stdout
    code_root = (tmp_path / "working" / "sam_code") if zipped else (inp / "sam-sot-mt-code")
    assert f"code: {code_root}" in res.stdout
    assert f"SOT_MT_FILE {code_root}" in res.stdout                  # imported from the package, not Sam/src
    exp = out_dir / "export" / "madlad-3b__mt-default"
    for name in ["generations_merged.parquet", "generations_delta.parquet", "runs.parquet", "RESUME.json",
                 "validation_report.csv", "attrition.csv", "new_translations.csv"]:
        assert (exp / name).is_file(), name
    delta = pd.read_parquet(exp / "generations_delta.parquet")
    assert len(delta) == 20 and delta.run_id.str.startswith("MOCK-").all()
    merged = pd.read_parquet(exp / "generations_merged.parquet")
    assert len(merged) == 420 and not schema.chain_violations(merged, pd.read_parquet(
        inp / "sam-sot-mt-staged" / "t0_manifest.parquet"))
    assert len(pd.read_csv(exp / "validation_report.csv")) == 20
    assert staged_before == {p.name: sha256_text(p.read_bytes().hex()) for p in (inp / "sam-sot-mt-staged").iterdir()}


def test_the_three_invalid_madlad_chains_cannot_continue(staged, tmp_path):
    manifest, base = staged
    bad = ["xsum|BigScience|xsum-4713", "xsum|LLAMA|xsum-4714", "xsum|LLAMA|xsum-4733"]
    r = session.open_runner("madlad-3b", "mt-default", manifest, base, work_dir=tmp_path, mock=True,
                            item_ids=bad, log=lambda *a: None)
    plan = r.plan().set_index("stage")
    assert plan.ready_now.sum() == 0 and plan.loc["S2", "blocked"] == 3
    assert r.run()["operations"] == 0 and not (tmp_path / "generations_delta.parquet").exists()


def _run_notebook(inp, out_dir, tmp):
    script = tmp / f"nb_{out_dir.parent.name}.py"
    script.write_text(_notebook_script(f"MOCK = True\nOUTPUT_DIR = {str(out_dir)!r}\n"))
    env = {"SAM_KAGGLE_INPUT": str(inp), "PATH": "/usr/bin:/bin", "HOME": str(tmp)}
    res = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, cwd=tmp, env=env)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res.stdout


def test_new_session_resumes_from_attached_notebook_output(package, tmp_path):
    _, built = package
    inp = tmp_path / "input"
    shutil.copytree(built["code"], inp / "sam-sot-mt-code")
    shutil.copytree(built["staged"], inp / "sam-sot-mt-staged")
    out1 = _run_notebook(inp, tmp_path / "session1" / "sam_outputs", tmp_path)
    assert "previous exports: none" in out1 and "this run: 20 records" in out1
    assert (tmp_path / "session1" / "sam_outputs" / "export" / "madlad-3b__mt-default.zip").is_file()
    # "Save Version" output of session 1, attached to session 2 as an input
    shutil.copytree(tmp_path / "session1" / "sam_outputs", inp / "smoke-notebook-output" / "sam_outputs")
    out2 = _run_notebook(inp, tmp_path / "session2" / "sam_outputs", tmp_path)
    prev = inp / "smoke-notebook-output" / "sam_outputs" / "export" / "madlad-3b__mt-default"
    assert f"previous exports: ['{prev}']" in out2 and "this run: 0 records" in out2
    merged = pd.read_parquet(tmp_path / "session2" / "sam_outputs" / "export" / "madlad-3b__mt-default"
                             / "generations_merged.parquet")
    assert len(merged) == 420 and not merged.duplicated(schema.KEY).any()
