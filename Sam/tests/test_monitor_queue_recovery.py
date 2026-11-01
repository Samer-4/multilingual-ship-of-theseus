"""Regression: the NLLB preflight queue stopped because the LOCAL monitor crashed on a sanity-probe event
(KeyError 'identical') while the remote job ran fine. The monitor must render every model's events, and the
queue must decide from the remote status, adopt launched kernels and never push a second time. CPU only."""
import importlib.util
import json
import re
import sys

import pytest

from sot_mt import paths, runner, session
from sot_mt import translators as T
from sot_mt.files import load_yaml
from sot_mt.progress import Progress

SAM = paths.SAM_ROOT
NB = SAM / "notebooks" / "00_generate_translations.ipynb"


def _script(name):
    sys.path.insert(0, str(SAM / "scripts"))
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mon = _script("monitor_kaggle")
q = _script("run_queue")

# -- event schemas, as the notebook emits them (field names copied from 00_generate_translations) ----------
MADLAD_DRIFT = {"event": "probe_done", "probe": "drift", "item_id": "xsum|OpenAI|xsum-4710", "identical": False,
                "similarity": 0.943, "ok": True, "op_seconds": 12.1, "error": None}
MADLAD_DRIFT_OLD = {k: v for k, v in MADLAD_DRIFT.items() if k != "probe"}          # logs before the `probe` field
MADLAD_DIAG = {"event": "probe_done", "probe": "diagnostic", "item_id": "xsum|LLAMA|xsum-4729", "variant": "decoder",
               "identical": True, "similarity": 1.0, "ok": True, "op_seconds": 9.0}
SANITY = {"event": "probe_done", "probe": "sanity", "item_id": "sanity en->es", "direction": "en->es", "ok": True,
          "blocking": "", "expected_words_found": "gobierno|puente|río", "op_seconds": 0.8}
SANITY_V1 = {"event": "probe_done", "item_id": "sanity en->es", "ok": True, "op_seconds": 0.8}   # the NLLB v1 crash
LOADED_MADLAD = {"event": "probe_model_loaded", "probe_load_s": 90.7,
                 "load_report": {"runtime_patch": "madlad-embed-decoder-v1", "embedding_tie": "decoder"}}
LOADED_OTHER = {"event": "probe_model_loaded", "probe_load_s": 41.0, "load_report": {"weights_id": "abc123"}}
SCHEMAS = {
    "madlad-3b": [{"event": "parity_done", "matched": 400, "checked": 400, "tokenizer": "T5TokenizerFast"},
                  LOADED_MADLAD, MADLAD_DRIFT, MADLAD_DRIFT_OLD, MADLAD_DIAG,
                  {"event": "quality_gate", "source": "drift_probe", "failed": 1}],
    **{m: [LOADED_OTHER, SANITY, SANITY_V1, {**SANITY, "ok": False, "blocking": "check:not_target_language",
                                             "expected_words_found": ""},
           {"event": "quality_gate", "source": "sanity_probe", "failed": 1},
           {"event": "gpu_memory", "gpu": [{"device": 0, "name": "Tesla T4", "max_allocated_gb": 3.1,
                                            "max_reserved_gb": 3.5}]}] for m in ("nllb", "qwen3-4b", "llama")},
}


def test_probe_done_with_and_without_identical():
    assert "drift probe" in mon.describe(MADLAD_DRIFT) and "identical=False" in mon.describe(MADLAD_DRIFT)
    assert "drift probe" in mon.describe(MADLAD_DRIFT_OLD)                          # MADLAD display preserved
    assert "diagnostic probe" in mon.describe(MADLAD_DIAG) and "[decoder]" in mon.describe(MADLAD_DIAG)
    for ev in (SANITY, SANITY_V1):
        line = mon.describe(ev)
        assert "sanity probe" in line and "ok=True" in line and "identical" not in line


def test_the_notebook_tags_every_probe_kind():
    src = "".join("".join(c["source"]) for c in json.loads(NB.read_text())["cells"])
    kinds = re.findall(r'PROGRESS\.event\("probe_done", probe="(\w+)"', src)
    assert sorted(kinds) == ["diagnostic", "drift", "sanity"]
    assert src.count('PROGRESS.event("probe_done"') == 3


def _real_runner_events(model, pv):
    """op_start / op_done / phase / run_start / run_end / export_done exactly as Progress emits them."""
    lines = []
    manifest, staged = session.load_inputs()
    tcfg, cfg = load_yaml("translators.yaml"), load_yaml("experiment.yaml")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        r = runner.Runner(runner.RunSettings(model, pv, item_ids=list(manifest.item_id[:1])),
                          T.MockTranslator(tcfg, model, pv), manifest, staged, tmp, cfg, tcfg, log=lambda *a: None,
                          progress=Progress(emit=lines.append))
        r.run()
        r.export(tmp)
    return [json.loads(x.split(" ", 1)[1]) for x in lines]


@pytest.mark.parametrize("model,pv", [("madlad-3b", "mt-default"), ("nllb", "mt-default"), ("qwen3-4b", "free"),
                                      ("llama", "free")])
def test_monitor_renders_every_known_event_of_every_model(model, pv):
    if not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("staged data unavailable")
    events = [{"event": "notebook_start", "translator": model, "mock": False}] + SCHEMAS[model if model in SCHEMAS
                                                                                         else "nllb"]
    if model != "madlad-3b":
        events += _real_runner_events(model, pv)
    events.append({"event": "notebook_done", "status": "completed"})
    out = []
    m = mon.Monitor("k", lambda *a: (0, ""), out=out.append)
    m.status = "RUNNING"
    m.feed([f"{json.dumps({'v': 1, 'seq': i + 1, **e})}" for i, e in enumerate(events)])  # unprefixed: ignored
    m.feed([f"SOTMT_PROGRESS {json.dumps({'v': 1, 'seq': i + 1, **e})}" for i, e in enumerate(events)])
    text = "\n".join(out)
    assert "not renderable" not in text and "unexpected fields" not in text
    if model == "madlad-3b":
        assert "tokenizer parity 400/400" in text and "drift probe" in text
    else:
        assert "sanity probe en->es: ok=True" in text and "peak GPU memory GPU0 3.1 GB" in text
        assert "run finished: completed" in text


def test_malformed_events_are_shown_not_fatal():
    assert "unexpected fields" in mon.describe({"event": "gpu_memory", "gpu": 7})
    out = []
    m = mon.Monitor("k", lambda *a: (0, ""), out=out.append)
    m.show({"event": "op_done", "pct": "x", "total": 3, "done": None})     # render() would raise on these types
    assert out and "not renderable" in out[-1]


# -- queue: remote status decides, launched kernels are adopted ------------------------------------------

def _dep(tmp_path, name, pushed=False, downloaded=False):
    d = tmp_path / name
    gen = tmp_path / "gen" / name
    d.mkdir()
    (d / "deploy.json").write_text(json.dumps({"kernel": f"samahmed17/{name}", "generated_root": str(gen),
                                               "translator": "nllb"}))
    if pushed:
        (d / "last_push.json").write_text(json.dumps({"kernel": f"samahmed17/{name}", "version": 1}))
    if downloaded:
        (gen / "xsum" / "nllb").mkdir(parents=True)
        (gen / "xsum" / "nllb" / "RESUME.json").write_text("{}")
    return str(d)


class Fake:
    """Scripted steps and remote statuses; records every command."""

    def __init__(self, statuses, monitor_rc=0):
        self.statuses, self.monitor_rc, self.calls = dict(statuses), monitor_rc, []

    def run(self, cmd, dry, out):
        self.calls.append([c.split("/")[-1] for c in cmd[1:]])
        return self.monitor_rc if "monitor_kaggle.py" in cmd[1] else 0

    def status(self, kernel):
        s = self.statuses[kernel.split("/")[1]]
        st = s.pop(0) if isinstance(s, list) and len(s) > 1 else (s[0] if isinstance(s, list) else s)
        return (st, "ok", "") if st not in {"auth", "api"} else (None, st, "401 Unauthorized" if st == "auth" else "")

    def names(self, script):
        return [c for c in self.calls if c[0] == script]


def test_monitor_crash_while_the_job_is_running_never_relaunches(tmp_path):
    a, b = _dep(tmp_path, "nllb"), _dep(tmp_path, "qwen")
    f = Fake({"nllb": "RUNNING"}, monitor_rc=1)                     # the monitor keeps crashing, job alive
    res = q.queue([a, b], yes=True, run=f.run, status=f.status, out=lambda *x: None, reattach=2)
    assert res[a]["outcome"] == "still_running" and res[a]["monitor"] == [1, 1, 1]
    assert len(f.names("launch_kaggle.py")) == 1 and not f.names("download_kaggle_outputs.py")
    assert b not in res                                             # qwen waits; nothing pushed twice


def test_monitor_crash_after_the_job_completed_downloads_and_continues(tmp_path):
    a, b = _dep(tmp_path, "nllb"), _dep(tmp_path, "qwen")
    f = Fake({"nllb": ["RUNNING", "COMPLETE"], "qwen": ["RUNNING", "COMPLETE"]}, monitor_rc=1)
    res = q.queue([a, b], yes=True, run=f.run, status=f.status, out=lambda *x: None)
    assert res[a]["outcome"] == res[b]["outcome"] == "downloaded"
    dl = f.names("download_kaggle_outputs.py")
    assert len(dl) == 2 and all("--raw-only" not in c for c in dl)
    launches = f.names("launch_kaggle.py")
    assert len(launches) == 2 and "--skip-code-upload" not in launches[0] and "--skip-code-upload" in launches[1]


def test_adopting_the_pushed_nllb_kernel_downloads_it_and_launches_only_qwen(tmp_path):
    a, b = _dep(tmp_path, "nllb", pushed=True), _dep(tmp_path, "qwen")
    f = Fake({"nllb": "COMPLETE", "qwen": ["RUNNING", "COMPLETE"]})
    res = q.queue([a, b], yes=True, run=f.run, status=f.status, out=lambda *x: None)
    assert res[a]["adopted"] and res[a]["outcome"] == "downloaded" and res[b]["outcome"] == "downloaded"
    launches = f.names("launch_kaggle.py")
    assert len(launches) == 1 and a not in launches[0] and "--skip-code-upload" not in launches[0]
    assert not [c for c in f.names("monitor_kaggle.py") if a in c]  # complete: nothing to follow


def test_adopted_and_already_downloaded_is_skipped(tmp_path):
    a = _dep(tmp_path, "nllb", pushed=True, downloaded=True)
    f = Fake({"nllb": "COMPLETE"})
    assert q.queue([a], yes=True, run=f.run, status=f.status, out=lambda *x: None)[a]["outcome"] == "already_downloaded"
    assert f.calls == []


def test_adopted_running_kernel_is_monitored_not_pushed(tmp_path):
    a = _dep(tmp_path, "nllb", pushed=True)
    f = Fake({"nllb": ["RUNNING", "RUNNING", "COMPLETE"]})
    res = q.queue([a], yes=True, run=f.run, status=f.status, out=lambda *x: None)
    assert res[a]["outcome"] == "downloaded" and not f.names("launch_kaggle.py") and f.names("monitor_kaggle.py")


def test_remote_failure_is_raw_downloaded_and_stops_the_queue(tmp_path):
    a, b = _dep(tmp_path, "nllb"), _dep(tmp_path, "qwen")
    f = Fake({"nllb": ["RUNNING", "ERROR"]}, monitor_rc=1)
    res = q.queue([a, b], yes=True, run=f.run, status=f.status, out=lambda *x: None)
    assert res[a]["outcome"] == "remote_failed" and f.names("download_kaggle_outputs.py")[0][-1] == "--raw-only"
    assert b not in res
    f2 = Fake({"nllb": ["RUNNING", "ERROR"], "qwen": ["RUNNING", "COMPLETE"]}, monitor_rc=1)
    res = q.queue([a, b], yes=True, keep_going=True, run=f2.run, status=f2.status, out=lambda *x: None)
    assert res[b]["outcome"] == "downloaded"                        # --keep-going only past a REMOTE failure


@pytest.mark.parametrize("kind", ["auth", "api"])
def test_unknown_remote_status_stops_without_relaunch_even_with_keep_going(tmp_path, kind):
    a, b = _dep(tmp_path, "nllb", pushed=True), _dep(tmp_path, "qwen")
    f = Fake({"nllb": kind})
    res = q.queue([a, b], yes=True, keep_going=True, run=f.run, status=f.status, out=lambda *x: None)
    assert res[a]["outcome"] == "status_unknown" and kind in res[a]["detail"] and b not in res and f.calls == []


def test_remote_status_retries_api_errors_and_classifies_auth():
    calls = []

    def flaky(*a):
        calls.append(a)
        return (1, "503 Service Unavailable") if len(calls) < 3 else (0, 'x has status "KernelWorkerStatus.COMPLETE"')
    assert q.remote_status("k", flaky, sleep=lambda s: None) == ("COMPLETE", "ok", 'x has status "KernelWorkerStatus.COMPLETE"')
    assert len(calls) == 3
    assert q.remote_status("k", lambda *a: (1, "401 Client Error: Unauthorized"), sleep=lambda s: None)[1] == "auth"
    assert q.remote_status("k", lambda *a: (1, "timeout"), tries=2, sleep=lambda s: None)[:2] == (None, "api")


def test_launcher_refuses_to_push_a_pushed_deployment_again(tmp_path):
    launch = _script("launch_kaggle")
    d = tmp_path / "dep"
    d.mkdir()
    (d / "last_push.json").write_text(json.dumps({"kernel": "samahmed17/x", "version": 1, "pushed_at": "t"}))
    with pytest.raises(launch.LaunchError, match="already pushed"):
        launch.launch(d, None, "kaggle", yes=True)


def test_download_summary_separates_sanity_from_drift_probes():
    dl = _script("download_kaggle_outputs")
    import pandas as pd
    v = {"merged_records": 418, "stage2_records": 400, "new_records": 18}
    text = dl.summarize(v, pd.DataFrame(), [SANITY_V1, {**SANITY, "item_id": "sanity es->en"}], SAM / "x")
    assert "sanity probe: 2/2" in text and "drift probe" not in text
    text = dl.summarize(v, pd.DataFrame(), [MADLAD_DRIFT_OLD, {**MADLAD_DRIFT, "identical": True}], SAM / "x")
    assert "drift probe: 1/2 S1 identical" in text and "sanity" not in text


def test_status_overview_reads_the_latest_event_and_survives_missing_events():
    st = _script("status_all")
    k = 'k has status "KernelWorkerStatus.RUNNING"'
    ev = {"v": 1, "seq": 5, "event": "op_done", "phase": "translating", "done": 4, "total": 18, "valid": 4,
          "invalid": 0, "elapsed_s": 75.0, "eta_text": "ETA 4m"}
    logs = "noise\nSOTMT_PROGRESS " + json.dumps(ev) + "\n"
    r = st.remote_state("k", lambda *a: (0, k) if a[0:2] == ("kernels", "status") else (0, logs))
    assert (r["state"], r["done"], r["total"], r["valid_ops"], r["eta"]) == ("RUNNING", 4, 18, 4, "ETA 4m")
    r = st.remote_state("k", lambda *a: (0, k) if a[1] == "status" else (0, "no progress lines"))
    assert r["state"] == "RUNNING" and r["phase"] == "no progress events yet"
    r = st.remote_state("k", lambda *a: (0, k) if a[1] == "status" else (1, "404"))
    assert r["phase"] == "log unavailable"
    assert "RUNNING" in st.table([{**st.local_state(d), **r} for d in st.deployments()][:1])
