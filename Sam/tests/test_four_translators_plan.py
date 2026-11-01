"""NLLB / Qwen / Phi-4-mini coordination: the real NLLB and Qwen preflights, their full-run deployments, the
Phi-4-mini entry and preflight, the one-command plan (full:MODEL with automatic continuation) and the generic
finalizer. CPU only; skipped where downloads are absent."""
import importlib.util
import json
import sys

import pandas as pd
import pytest

from sot_mt import paths, runner, session
from sot_mt import translators as T
from sot_mt.files import load_yaml, sha256_file

SAM = paths.SAM_ROOT
TCFG = load_yaml("translators.yaml")
LONG = "xsum|Human|xsum-4709"


def _script(name):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pre(model):
    d = SAM / "data" / "generated" / model / "preflight" / "xsum" / model
    if not (d / runner.RESUME).exists():
        pytest.skip(f"{model} preflight not downloaded")
    return d


# -- the real preflights ---------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["nllb", "qwen3-4b"])
def test_real_preflight_is_complete_from_t0_and_not_truncated(model):
    d = _pre(model)
    r = _script("verify_madlad_pilot").verify(d, translator=model)
    assert (r["records"], r["new_records"], r["complete_items"], r["invalid_new"]) == (418, 18, 3, [])
    assert all(v == {"valid": 3} for v in r["new_by_stage"].values()) and r["source_labels"] == 3
    delta = pd.read_parquet(d / runner.DELTA)
    ids = pd.read_csv(SAM / "data" / "manifests" / "preflight3.csv").item_id
    assert set(delta.item_id) == set(ids) and delta.model_checkpoint.nunique() == 1
    meta = delta.generation_meta.map(json.loads)
    assert (meta.map(lambda m: m["finish"]) == "stop").all()
    if model == "qwen3-4b":                                   # every output ended on EOS, well below its cap
        assert meta.map(lambda m: m["hit_eos"] and m["n_out_tokens"] < m["max_new_tokens"]).all()
    else:                                                     # every NLLB segment ended on EOS
        assert meta.map(lambda m: all(m["segment_hit_eos"])).all()
    manifest, _ = session.load_inputs()
    t0 = manifest.set_index("item_id").t0_text[LONG]
    s1 = delta[(delta.item_id == LONG) & (delta.output_stage == "S1")].output_text.iloc[0]
    assert len(s1.split()) > len(t0.split())                 # the 916-word document is translated in full ...
    assert "abolición" in s1[-250:]                           # ... up to its last sentence


# -- full-run deployments ---------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["nllb", "qwen3-4b"])
def test_full_deployment_reuses_its_own_preflight_and_pins_its_weights(model):
    pre = _pre(model)
    dep_dir = SAM / "kaggle" / model / "full"
    dep = json.loads((dep_dir / "deploy.json").read_text())
    meta = json.loads((dep_dir / "kernel-metadata.json").read_text())
    o = dep["overrides"]
    lr = json.loads(pd.read_parquet(pre / runner.RUNS).load_report.iloc[-1])
    assert dep["kernel"] == f"samahmed17/sam-sot-mt-{model}-full"
    assert meta["kernel_sources"] == [f"samahmed17/sam-sot-mt-{model}-preflight"]
    assert json.loads(o["EXPECTED_PREVIOUS_MERGED_SHA256"]) == [sha256_file(pre / runner.MERGED)]
    assert json.loads(o["EXPECTED_WEIGHTS_ID"]) == lr["weights_id"]
    assert (o["ITEM_IDS_FILE"], o["MAX_ITEMS"], o["TRANSLATOR"]) == ("None", "None", json.dumps(model))
    assert (o["EXPORT_EVERY"], o["SANITY_PROBE"], o["ROUNDS"]) == ("20", "True", "(1, 3)")
    est = _script("model_runs").estimate(model)
    assert dep["timeout_s"] == est["timeout_s"] <= 12 * 3600
    assert int(o["TIME_LIMIT_MINUTES"]) * 60 + est["setup_s_used"] + 20 * 60 <= dep["timeout_s"]


def test_qwen_estimate_is_length_aware_and_fits_its_sessions():
    _pre("qwen3-4b")
    est = _script("model_runs").estimate("qwen3-4b")
    s, t = est["fit_by_direction"]["en->es"], est["fit_by_direction"]["es->en"]
    assert s[1] > t[1] > 0                                    # Spanish output costs more per input word
    assert est["mae_s"] < 10 and est["remaining_operations"] == 1182
    naive = 1182 * est["seconds_total"] / est["operations"]   # the long preflight document inflates a mean rate
    assert est["est_translate_s"] < naive
    assert est["est_sessions"] * (est["time_limit_minutes"] * 60) >= est["est_translate_s"]


# -- Phi-4-mini -------------------------------------------------------------------------------------------

def test_phi4_mini_entry_and_adapter():
    e = TCFG["translators"]["phi4-mini"]
    assert e["model_checkpoint"] == "microsoft/Phi-4-mini-instruct" and len(e["model_revision"]) == 40
    assert not e.get("requires_hf_token") and "weights" not in e["settings"]
    assert e["prompt_variants"]["free"] == TCFG["translators"]["qwen3-4b"]["prompt_variants"]["free"]
    assert e["settings"]["decoding"] == TCFG["translators"]["qwen3-4b"]["settings"]["decoding"]
    a = T.build_translator(TCFG, "phi4-mini", "free", allow_download=True)
    assert isinstance(a, T.ChatTranslator) and a._hf_kwargs()["revision"] == e["model_revision"]
    ids = {T.generation_config(TCFG, m, pv)[0] for m, pv in
           [("madlad-3b", "mt-default"), ("nllb", "mt-default"), ("qwen3-4b", "free"), ("llama", "free"),
            ("phi4-mini", "free")]}
    assert len(ids) == 5 and T.generation_config(TCFG, "madlad-3b", "mt-default")[0] == "705a8bd4992e"
    assert "llama" in TCFG["translators"]                     # kept, not run


def test_phi4_mini_preflight_deployment_uses_the_same_documents():
    dep = json.loads((SAM / "kaggle" / "phi4-mini" / "preflight" / "deploy.json").read_text())
    nllb = json.loads((SAM / "kaggle" / "nllb" / "preflight" / "deploy.json").read_text())
    assert dep["kernel"] == "samahmed17/sam-sot-mt-phi4-mini-preflight" and dep["model_sources"] == []
    assert dep["overrides"]["ITEM_IDS_FILE"] == nllb["overrides"]["ITEM_IDS_FILE"] == '"data/manifests/preflight3.csv"'
    assert dep["overrides"]["TRANSLATOR"] == '"phi4-mini"' and dep["results_dir"] == "results/phi4-mini/preflight"


# -- the plan ---------------------------------------------------------------------------------------------

class Fake:
    def __init__(self, statuses):
        self.statuses, self.calls = statuses, []

    def run(self, cmd, dry, out):
        self.calls.append([c.split("/")[-1] for c in cmd[1:]])
        return 0

    def status(self, kernel):
        return (self.statuses.get(kernel.split("/")[1], "COMPLETE"), "ok", "")


def _mkdep(d, name, model="qwen3-4b"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "deploy.json").write_text(json.dumps({"kernel": f"samahmed17/{name}", "translator": model,
                                               "generated_root": str(d / "gen")}))


def test_full_item_prepares_continues_while_operations_remain_and_uploads_code_once(tmp_path, monkeypatch):
    q = _script("run_queue")
    monkeypatch.setattr(q, "full_dir", lambda m, k: tmp_path / m / f"full{k}")
    left = {1: (500, 700), 2: (0, 482)}
    prepared = []

    def prepare(m, k):
        prepared.append((m, k))
        _mkdep(tmp_path / m / f"full{k}", f"{m}-full{k}", m)
        return 0
    f = Fake({})
    pre = tmp_path / "phi-pre"
    _mkdep(pre, "phi-pre", "phi4-mini")
    res = q.plan(["full:qwen3-4b", str(pre)], yes=True, run=f.run, status=f.status, out=lambda *x: None,
                 prepare=prepare, remaining=lambda d: left[int(d[-1])])
    assert prepared == [("qwen3-4b", 1), ("qwen3-4b", 2)]
    assert [r["outcome"] for r in res.values()] == ["downloaded"] * 3
    launches = [c for c in f.calls if c[0] == "launch_kaggle.py"]
    assert len(launches) == 3 and ["--skip-code-upload" in c for c in launches] == [False, True, True]


def test_full_item_stops_without_progress_and_prepare_failure_policy(tmp_path, monkeypatch):
    q = _script("run_queue")
    monkeypatch.setattr(q, "full_dir", lambda m, k: tmp_path / m / f"full{k}")
    _mkdep(tmp_path / "qwen3-4b" / "full1", "q1")
    f = Fake({})
    res = q.plan(["full:qwen3-4b", "full:phi4-mini"], yes=True, run=f.run, status=f.status, out=lambda *x: None,
                 prepare=lambda m, k: 1, remaining=lambda d: (10, 0))
    assert list(res) == [str(tmp_path / "qwen3-4b" / "full1")]          # no progress: stops, phi not started
    res = q.plan(["full:phi4-mini", "full:nllb"], yes=True, keep_going=True, run=f.run, status=f.status,
                 out=lambda *x: None, prepare=lambda m, k: 1, remaining=lambda d: (0, 1))
    assert [r["outcome"] for r in res.values()] == ["prepare_failed", "prepare_failed"]   # --keep-going passes
    res = q.plan(["full:phi4-mini", "full:nllb"], yes=True, run=f.run, status=f.status, out=lambda *x: None,
                 prepare=lambda m, k: 1, remaining=lambda d: (0, 1))
    assert list(res) == ["full:phi4-mini#1"]


def test_plan_adopts_a_pushed_full_run_instead_of_preparing_or_pushing(tmp_path, monkeypatch):
    q = _script("run_queue")
    monkeypatch.setattr(q, "full_dir", lambda m, k: tmp_path / m / f"full{k}")
    d = tmp_path / "nllb" / "full1"
    _mkdep(d, "n1", "nllb")
    (d / "last_push.json").write_text("{}")
    f = Fake({"n1": "RUNNING"})
    f.status = lambda k, it=iter(["RUNNING", "RUNNING", "COMPLETE"]): (next(it, "COMPLETE"), "ok", "")
    res = q.plan(["full:nllb"], yes=True, run=f.run, status=f.status, out=lambda *x: None,
                 prepare=lambda m, k: pytest.fail("must not prepare"), remaining=lambda d: (0, 1))
    assert res[str(d)]["adopted"] and not [c for c in f.calls if c[0] == "launch_kaggle.py"]


# -- finalizer for the other models -------------------------------------------------------------------------

def test_model_finalizer_takes_the_latest_attempt_and_checks_the_earlier_ones(tmp_path, monkeypatch):
    fin = _script("finalize_madlad")
    import model_runs as mr
    for k in (1, 2):
        (tmp_path / f"full{k}").mkdir()
        (tmp_path / f"full{k}" / runner.RESUME).write_text("{}")
    monkeypatch.setattr(mr, "export_dir", lambda m, kind, k=1: str(tmp_path / (f"full{k}" if kind == "full" else "pre")))
    src, earlier, out = fin.model_sources("qwen3-4b")
    assert src == str(tmp_path / "full2") and earlier == [str(tmp_path / "pre"), str(tmp_path / "full1")]
    assert out == "data/generated/qwen3-4b_final"
