"""Progress events, the terminal monitor, Kaggle deployment and download/filing. No network, no model:
the Kaggle CLI is replaced by fakes, the notebook runs in MOCK mode from the packaged folders."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sot_mt import kaggle_cli, paths, progress, runner, schema, session, smoke
from sot_mt import translators as T
from sot_mt.files import load_yaml, verify_code_tree

SAM = paths.SAM_ROOT
TCFG = load_yaml("translators.yaml")
SMOKE_FILE = SAM / "data" / "manifests" / "stage3b_madlad_smoke5.csv"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SAM / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SAM / "scripts"))
    spec.loader.exec_module(mod)
    return mod


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def collect():
    lines = []
    return lines, progress.Progress(emit=lines.append, heartbeat_s=0)


def events(lines):
    return [json.loads(x[len(progress.PREFIX):]) for x in lines]


@pytest.fixture(scope="module")
def staged():
    if not (paths.staged_dir() / "STAGED_FILES.json").exists():
        pytest.skip("staged Stage 2 data unavailable")
    return session.load_inputs()


def smoke_runner(staged, tmp_path, script=None, prog=None, item_ids=None):
    manifest, base = staged
    tr = T.MockTranslator(TCFG, "madlad-3b", "mt-default", script=script)
    return runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=item_ids or smoke.read_item_ids(SMOKE_FILE)),
                         tr, manifest, base, tmp_path, load_yaml("experiment.yaml"), TCFG, run_id="r1",
                         log=lambda *a: None, progress=prog)


# -- progress counts -----------------------------------------------------------------------------------

def test_progress_counts_through_a_full_smoke_run(staged, tmp_path):
    lines, prog = collect()
    smoke_runner(staged, tmp_path, prog=prog).run()
    ev = events(lines)
    assert all(x.startswith(progress.PREFIX) for x in lines)
    start, end = ev[0], ev[-1]
    assert start["event"] == "run_start" and start["total"] == 20 and start["planned_items"] == 5
    assert start["model"] == "google/madlad400-3b-mt" and start["dataset"] == "xsum"
    phases = [e["phase"] for e in ev if e["event"] == "phase"]
    assert phases == ["loading_model"] and any(e["event"] == "model_loaded" for e in ev)
    done = [e for e in ev if e["event"] == "op_done"]
    assert [e["done"] for e in done] == list(range(1, 21))
    assert [e["stage"] for e in done] == ["S2"] * 5 + ["T2"] * 5 + ["S3"] * 5 + ["T3"] * 5
    starts = [e for e in ev if e["event"] == "op_start"]
    assert len(starts) == 20 and starts[0]["current_stage"] == "S2" and starts[0]["current_item_id"]
    assert end["event"] == "run_end" and end["status"] == "completed" and end["pct"] == 100.0
    assert (end["valid"], end["invalid"], end["failed"], end["dropped"]) == (20, 0, 0, 0)
    assert [e["seq"] for e in ev] == sorted({e["seq"] for e in ev})          # unique, increasing


def test_eta_uses_measured_seconds_per_word_and_excludes_model_load():
    lines, _ = collect()
    clock = Clock()
    p = progress.Progress(emit=lines.append, clock=clock, heartbeat_s=0, min_samples=2)
    p.start({"translator": "x"}, {"a": ["S2", "T2", "S3", "T3"], "b": ["S2", "T2", "S3", "T3"]}, {"a": 100, "b": 300})
    clock.t = 50
    p.loaded(50.0)
    assert p.snapshot()["eta_text"] == "ETA calculating"
    p.op_start("a", "S2", "w " * 100)
    clock.t = 60
    p.op_done("a", "S2", "valid", 10.0, "w " * 100, "w " * 110, n_segments=1)
    assert p.eta() is None and p.snapshot()["eta_text"] == "ETA calculating"   # one sample is not enough
    p.op_done("b", "S2", "valid", 30.0, "w " * 300, "w " * 320, n_segments=2)
    # 40 s / 400 words = 0.1 s/word; remaining: a 3 stages x 110 words + b 3 x 320 = 1290 words
    assert p.eta() == pytest.approx(129.0)
    snap = p.snapshot()
    assert snap["load_s"] == 50.0 and snap["eta_s"] == pytest.approx(129.0) and snap["eta_text"].startswith("ETA 2m")
    assert snap["segments_done"] == 3 and snap["words_per_s"] == 10.0 and snap["ops_per_min"] == 3.0
    assert (snap["done"], snap["total"], snap["pct"]) == (2, 8, 25.0)


def test_multi_segment_documents_weigh_more_in_the_eta():
    p = progress.Progress(heartbeat_s=0, min_samples=1)
    p.start({}, {"short": ["S2", "T2"], "long": ["S2", "T2"]}, {"short": 50, "long": 600})
    p.op_done("short", "S2", "valid", 5.0, "w " * 50, "w " * 50, n_segments=1)
    eta_after_short = p.eta()                     # 0.1 s/word x (50 + 600 x 2) remaining words
    assert eta_after_short == pytest.approx(0.1 * (50 + 1200))
    p.op_done("long", "S2", "valid", 60.0, "w " * 600, "w " * 640, n_segments=3)
    assert p.segments == 4 and p.eta() == pytest.approx(65.0 / 650 * (50 + 640))


def test_madlad_segment_counts_reach_the_progress_events(staged, tmp_path):
    class Segmented(T.MockTranslator):
        def translate(self, text, src, tgt):
            r = super().translate(text, src, tgt)
            r.meta["segmentation"] = {"method": "madlad_pack", "n_segments": 1 + len(text.split()) // 200}
            return r
    manifest, base = staged
    lines, prog = collect()
    tr = Segmented(TCFG, "madlad-3b", "mt-default")
    runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=smoke.read_item_ids(SMOKE_FILE),
                                     stop_after="S2"), tr, manifest, base, tmp_path, load_yaml("experiment.yaml"),
                  TCFG, run_id="r", log=lambda *a: None, progress=prog).run()
    segs = {e["item_id"]: e["n_segments"] for e in events(lines) if e["event"] == "op_done"}
    assert segs["xsum|BigScience|xsum-4720"] == 3 and segs["xsum|OpenAI|xsum-4710"] == 1


# -- invalid, blocked, failed, interrupted ----------------------------------------------------------

def test_invalid_output_drops_the_rest_of_the_chain_and_still_reaches_100(staged, tmp_path):
    lines, prog = collect()
    smoke_runner(staged, tmp_path, script={1: "loop"}, prog=prog).run()
    ev = events(lines)
    first = next(e for e in ev if e["event"] == "op_done")
    assert first["status"] == "invalid" and first["dropped_after"] == 3 and "check:repetition" in first["reason"]
    end = ev[-1]
    assert (end["done"], end["total"], end["pct"]) == (20, 20, 100.0)
    assert (end["valid"], end["invalid"], end["dropped"], end["operations"]) == (16, 1, 3, 17)


def test_transient_failure_counts_as_failed_and_drops_downstream(staged, tmp_path):
    lines, prog = collect()
    smoke_runner(staged, tmp_path, script={2: "oom"}, prog=prog).run()
    end = events(lines)[-1]
    assert (end["failed"], end["dropped"], end["done"], end["pct"]) == (1, 3, 20, 100.0)


def test_already_blocked_chains_are_not_planned(staged, tmp_path):
    lines, prog = collect()
    bad = ["xsum|BigScience|xsum-4713", "xsum|LLAMA|xsum-4714", "xsum|LLAMA|xsum-4733"]
    smoke_runner(staged, tmp_path, prog=prog, item_ids=bad).run()
    ev = events(lines)
    assert ev[0]["total"] == 0 and ev[-1]["status"] == "completed" and ev[-1]["pct"] == 100.0


def test_interrupted_run_reports_interrupted_and_partial_progress(staged, tmp_path):
    lines, prog = collect()
    with pytest.raises(KeyboardInterrupt):
        smoke_runner(staged, tmp_path, script={7: "interrupt"}, prog=prog).run()
    end = events(lines)[-1]
    assert end["event"] == "run_end" and end["status"] == "interrupted" and end["done"] == 6 and end["pct"] == 30.0
    # resuming plans only the remaining 14 operations
    lines2, prog2 = collect()
    smoke_runner(staged, tmp_path, prog=prog2).run()
    ev2 = events(lines2)
    assert ev2[0]["total"] == 14 and ev2[-1]["pct"] == 100.0


def test_load_failure_is_reported_as_failed(staged, tmp_path):
    class NoGPU(T.MockTranslator):
        def _load(self):
            raise T.NoGPUError("no CUDA GPU visible")
    manifest, base = staged
    lines, prog = collect()
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=smoke.read_item_ids(SMOKE_FILE)),
                      NoGPU(TCFG, "madlad-3b", "mt-default"), manifest, base, tmp_path, load_yaml("experiment.yaml"),
                      TCFG, run_id="r", log=lambda *a: None, progress=prog)
    with pytest.raises(T.NoGPUError):
        r.run()
    end = events(lines)[-1]
    assert end["status"] == "failed" and "NoGPUError" in end["error"] and end["phase"] == "failed"


def test_heartbeat_repeats_state_while_silent():
    lines = []
    p = progress.Progress(emit=lines.append, heartbeat_s=0.05)
    p.start({"translator": "x"}, {"a": ["S2"]}, {"a": 10})
    p.set_phase("loading_model")
    import time
    time.sleep(0.3)
    p.finish("completed")
    assert any(json.loads(x[len(progress.PREFIX):])["event"] == "heartbeat" for x in lines)


# -- log parsing -----------------------------------------------------------------------------------

def _ev(seq, event="op_done", **kw):
    return progress.PREFIX + json.dumps({"v": 1, "seq": seq, "event": event, **kw})


def test_parser_handles_split_garbage_duplicate_and_json_array_logs():
    p = kaggle_cli.EventParser()
    line = _ev(2, done=1, total=4)
    p.feed(["setup output", _ev(1, "run_start", total=4), line[:40], line[40:], progress.PREFIX + "{not json",
            "more", _ev(1, "run_start", total=4), "Traceback (most recent call last):", "RuntimeError: boom"])
    p.close()
    assert [e["seq"] for e in p.events] == [1, 2] and p.malformed == 1
    assert p.errors[-1] == "RuntimeError: boom"
    stored = json.dumps([{"stream_name": "stdout", "time": 1.0, "data": _ev(3, done=2) + "\n"},
                         {"stream_name": "stdout", "time": 2.0, "data": "plain\n" + _ev(4, "notebook_done") + "\n"}])
    p.feed(kaggle_cli.log_lines(stored))
    assert [e["seq"] for e in p.events] == [1, 2, 3, 4] and p.last_of("notebook_done")["seq"] == 4


def test_status_and_push_parsing():
    assert kaggle_cli.parse_status('a/b has status "KernelWorkerStatus.RUNNING"') == ("RUNNING", None)
    assert kaggle_cli.parse_status('a/b has status "complete"') == ("COMPLETE", None)
    s, f = kaggle_cli.parse_status('a/b has status "KernelWorkerStatus.ERROR"\nFailure message: "OOM"')
    assert (s, f) == ("ERROR", "OOM")
    assert kaggle_cli.parse_status("404 Client Error") == (None, None)
    assert kaggle_cli.parse_push("Kernel version 3 successfully pushed.  Please check progress at x") == 3


def test_render_never_invents_progress():
    mon = _script("monitor_kaggle")
    assert "no progress events yet" in mon.render(None, "RUNNING")
    line = mon.render({"event": "op_done", "phase": "translating", "translator": "madlad-3b", "model": "m",
                       "dataset": "xsum", "done": 5, "total": 20, "pct": 25.0, "valid": 5, "invalid": 0, "failed": 0,
                       "dropped": 0, "elapsed_s": 300, "load_s": 120, "eta_s": None, "eta_text": "ETA calculating"},
                      "RUNNING", 4)
    assert "5/20 ops 25.0%" in line and "ETA calculating" in line and "model load 2m00s" in line


# -- monitor state machine -----------------------------------------------------------------------------

class FakeKaggle:
    def __init__(self, statuses, log_text=""):
        self.statuses, self.log_text, self.calls = list(statuses), log_text, []

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("kernels", "status"):
            s = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return 0, s
        if args[:2] == ("kernels", "logs"):
            return 0, self.log_text
        return 1, "unexpected"


def _monitor(fake, **kw):
    mon = _script("monitor_kaggle")
    out, clock = [], Clock()

    def sleep(s):
        clock.t += s
    m = mon.Monitor("u/k", fake, follow_cmd=None, interval=10, out=out.append, clock=clock, sleep=sleep, **kw)
    return m, out


ST = 'u/k has status "KernelWorkerStatus.{}"'


def test_monitor_completed_with_notebook_done():
    log = "\n".join([_ev(1, "run_start", total=2, done=0, phase="loading_model"),
                     _ev(2, done=1, total=2, item_id="i", stage="S2", status="valid", phase="translating"),
                     _ev(3, "notebook_done", status="completed")])
    m, out = _monitor(FakeKaggle([ST.format("QUEUED"), ST.format("RUNNING"), ST.format("COMPLETE")], log))
    assert m.run() == 0
    text = "\n".join(out)
    assert "COMPLETE: notebook_done" in text and "1/2 ops" in text


def test_monitor_error_reports_failure_and_tracebacks():
    log = "Traceback (most recent call last):\nRuntimeError: No CUDA GPU in this session"
    m, out = _monitor(FakeKaggle([ST.format("RUNNING"), ST.format("ERROR") + '\nFailure message: "cell failed"'], log))
    assert m.run() == 1
    text = "\n".join(out)
    assert 'failure message "cell failed"' in text and "No CUDA GPU" in text


def test_monitor_complete_without_notebook_done_and_cancelled():
    m, _ = _monitor(FakeKaggle([ST.format("COMPLETE")], "no events here"))
    assert m.run() == 3
    m, _ = _monitor(FakeKaggle([ST.format("CANCEL_ACKNOWLEDGED")], ""))
    assert m.run() == 2


def test_monitor_unreadable_status_and_max_wait():
    m, out = _monitor(FakeKaggle(["404 not found"]))
    assert m.run() == 5 and any("could not read status" in o for o in out)
    m, out = _monitor(FakeKaggle([ST.format("RUNNING")], ""), max_wait_min=1)
    assert m.run() == 4 and any("no log lines received yet" in o for o in out)


def test_monitor_follows_a_live_stream(tmp_path):
    mon = _script("monitor_kaggle")
    lines = [_ev(1, "run_start", total=1, done=0), _ev(2, done=1, total=1, item_id="i", stage="S2", status="valid"),
             _ev(3, "notebook_done", status="completed")]
    src = tmp_path / "stream.py"
    src.write_text("import time\n" + "".join(f"print({x!r}, flush=True); time.sleep(0.05)\n" for x in lines))
    fake = FakeKaggle([ST.format("RUNNING")] * 3 + [ST.format("COMPLETE")], "")
    out = []
    m = mon.Monitor("u/k", fake, follow_cmd=[sys.executable, str(src)], interval=0.3, out=out.append)
    assert m.run() == 0 and len(m.parser.events) == 3            # stored log is empty: events came from the stream


# -- deployment ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def deployed(tmp_path_factory, staged):
    prep = _script("prepare_kaggle_deploy")
    root = tmp_path_factory.mktemp("deploy")
    d = prep.prepare("samahmed17", "sam-sot-mt-madlad-smoke5", root / "deploy", prep.SMOKE["overrides"], 5400,
                     build_out=root / "build")
    return prep, root, d


def test_deployment_metadata_and_notebook_overrides(deployed):
    prep, root, d = deployed
    meta = json.loads((root / "deploy" / "kernel-metadata.json").read_text())
    assert meta["id"] == "samahmed17/sam-sot-mt-madlad-smoke5" and meta["code_file"] == "00_generate_translations.ipynb"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["kernel_type"] == "notebook"
    assert meta["dataset_sources"] == ["samahmed17/sam-sot-mt-code", "samahmed17/sam-sot-mt-staged"]
    assert prep.validate(root / "deploy", meta) == []
    nb = json.loads((root / "deploy" / "00_generate_translations.ipynb").read_text())
    cfg = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    for line in ['TRANSLATOR = "madlad-3b"', "MAX_ITEMS = 5", "MOCK = False", "ALLOW_MODEL_DOWNLOAD = True",
                 "DRIFT_PROBE = True", 'ITEM_IDS_FILE = "data/manifests/stage3b_madlad_smoke5.csv"',
                 f'EXPECTED_CODE_SHA = "{d["code_digest"]}"']:
        assert line in cfg, line
    assert verify_code_tree(root / "build" / "sam-sot-mt-code" / "Sam", d["code_digest"]) == d["code_digest"]


@pytest.mark.parametrize("change,problem", [
    ({"is_private": False}, "private"), ({"machine_shape": "NvidiaTeslaP100"}, "retired"),
    ({"title": "other-title"}, "resolve"), ({"enable_internet": False}, "Internet"),
    ({"dataset_sources": ["no-slash"]}, "dataset source"), ({"id": "a/b/3"}, "version")])
def test_validation_rejects_bad_metadata(deployed, change, problem):
    prep, root, _ = deployed
    meta = {**json.loads((root / "deploy" / "kernel-metadata.json").read_text()), **change}
    assert any(problem in e for e in prep.validate(root / "deploy", meta))


def test_stale_code_or_edited_notebook_blocks_launch(deployed, tmp_path):
    prep, root, d = deployed
    launch = _script("launch_kaggle")
    dep = tmp_path / "deploy"
    shutil.copytree(root / "deploy", dep)
    assert launch.check(dep)["kernel"] == d["kernel"]
    code = tmp_path / "code"
    shutil.copytree(root / "build" / "sam-sot-mt-code", code)
    (code / "Sam" / "configs" / "experiment.yaml").write_text("changed: true\n")
    with pytest.raises(RuntimeError, match="differs"):
        verify_code_tree(code / "Sam")
    with pytest.raises(RuntimeError, match="expects"):
        verify_code_tree(root / "build" / "sam-sot-mt-code" / "Sam", "0" * 16)
    nb = dep / "00_generate_translations.ipynb"
    nb.write_text(nb.read_text().replace("DRIFT_PROBE = True", "DRIFT_PROBE = False"))
    with pytest.raises(SystemExit, match="notebook changed"):
        launch.check(dep)


def test_missing_config_key_is_an_error():
    prep = _script("prepare_kaggle_deploy")
    nb = {"cells": [{"cell_type": "code", "source": ["A = 1\n"]}]}
    with pytest.raises(KeyError):
        prep.override_config(nb, {"B": "2"})


# -- deployed notebook (MOCK) -> fake download -> local filing -> resume ------------------------------

def _run_deployed(deployed, tmp_path):
    prep, root, d = deployed
    inp = tmp_path / "input"
    shutil.copytree(root / "build" / "sam-sot-mt-code", inp / "sam-sot-mt-code")
    shutil.copytree(root / "build" / "sam-sot-mt-staged", inp / "sam-sot-mt-staged")
    nb = json.loads((root / "deploy" / "00_generate_translations.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    work = tmp_path / "kaggle" / "working"
    script = tmp_path / "nb.py"
    script.write_text("\n\n".join(["display = print", cells[0],
                                   f"MOCK = True\nOUTPUT_DIR = {str(work / 'sam_outputs')!r}", *cells[1:]]))
    res = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, cwd=tmp_path,
                         env={"SAM_KAGGLE_INPUT": str(inp), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return work, res.stdout


def test_deployed_notebook_emits_progress_and_files_locally(deployed, tmp_path, staged):
    _, _, d = deployed
    work, stdout = _run_deployed(deployed, tmp_path)
    assert f"code digest: {d['code_digest']}" in stdout
    p = kaggle_cli.EventParser()
    p.feed(stdout.splitlines())
    names = [e["event"] for e in p.events]
    assert names[0] == "notebook_start" and names[-1] == "notebook_done" and "export_done" in names
    assert p.last_of("run_end")["total"] == 20 and p.last_of("run_end")["pct"] == 100.0

    dl = _script("download_kaggle_outputs")
    log = json.dumps([{"stream_name": "stdout", "time": 0, "data": line + "\n"} for line in stdout.splitlines()])

    def fake_cli(*args):
        if args[:2] == ("kernels", "status"):
            return 0, 'k has status "KernelWorkerStatus.COMPLETE"'
        if args[:2] == ("kernels", "output"):
            dest = Path(args[args.index("-p") + 1])
            shutil.copytree(work, dest, dirs_exist_ok=True)
            (dest / "sam-sot-mt-madlad-smoke5.log").write_text(log)
            return 0, "Output file downloaded"
        return 1, ""
    raw = tmp_path / "results" / "raw" / "v1"
    dl.download(fake_cli, d["kernel"], raw, version=1)
    raw_listing = json.loads((raw / "DOWNLOAD.json").read_text())["files"]
    gen_root, results = tmp_path / "generated", tmp_path / "results"
    out = dl.organize(raw, paths.staged_dir(), gen_root, results, "madlad-3b", "mt-default", allow_mock=True)
    gen = gen_root / "xsum" / "madlad-3b"
    assert out["generated_dir"] == gen
    for name in ["generations_merged.parquet", "generations_delta.parquet", "runs.parquet", "RESUME.json",
                 "PROVENANCE.json"]:
        assert (gen / name).is_file(), name
    for name in ["validation_report.csv", "attrition.csv", "new_translations.csv", "madlad-3b__mt-default.zip",
                 "sam-sot-mt-madlad-smoke5.log", "progress_events.jsonl", "summary.md", "PROVENANCE.json"]:
        assert (results / name).is_file(), name
    assert out["new_records"] == 20 and out["merged_records"] == 420 and out["stage2_records_verified"] == 400
    assert "| S2 | 5 | 0 |" in (results / "summary.md").read_text()
    assert len((results / "progress_events.jsonl").read_text().splitlines()) == len(p.events)
    assert json.loads((raw / "DOWNLOAD.json").read_text())["files"] == raw_listing     # raw untouched
    with pytest.raises(SystemExit, match="not empty"):
        dl.download(fake_cli, d["kernel"], raw)

    # idempotent re-filing; a differing local file is never silently overwritten
    assert set(dl.organize(raw, paths.staged_dir(), gen_root, results, "madlad-3b", "mt-default",
                           allow_mock=True)["placed"].values()) == {"unchanged"}
    (results / "attrition.csv").write_text("edited\n")
    with pytest.raises(dl.ProvenanceError, match="different content"):
        dl.organize(raw, paths.staged_dir(), gen_root, results, "madlad-3b", "mt-default", allow_mock=True)
    out = dl.organize(raw, paths.staged_dir(), gen_root, results, "madlad-3b", "mt-default", allow_mock=True,
                      replace=True)
    assert out["superseded"] and Path(out["superseded"][0]).read_text() == "edited\n"
    with pytest.raises(dl.ProvenanceError, match="MOCK"):
        dl.organize(raw, paths.staged_dir(), tmp_path / "g2", tmp_path / "r2", "madlad-3b", "mt-default")

    # the filed table resumes: a later session with it as base has nothing left to do for these items
    manifest, _ = staged
    base = runner.load_tables([gen / "generations_merged.parquet"], manifest)
    r = runner.Runner(runner.RunSettings("madlad-3b", "mt-default", item_ids=smoke.read_item_ids(SMOKE_FILE)),
                      T.MockTranslator(TCFG, "madlad-3b", "mt-default"), manifest, base, tmp_path / "next",
                      load_yaml("experiment.yaml"), TCFG, log=lambda *a: None)
    assert r.run()["operations"] == 0


def test_tampered_export_is_rejected(deployed, tmp_path):
    work, _ = _run_deployed(deployed, tmp_path)
    dl = _script("download_kaggle_outputs")
    exp = work / "sam_outputs" / "export" / "madlad-3b__mt-default"
    (exp / "attrition.csv").write_text("tampered\n")
    with pytest.raises(dl.ProvenanceError, match="RESUME.json"):
        dl.organize(work, paths.staged_dir(), tmp_path / "g", tmp_path / "r", "madlad-3b", "mt-default",
                    allow_mock=True)


def test_download_refuses_unfinished_runs(tmp_path):
    dl = _script("download_kaggle_outputs")
    with pytest.raises(SystemExit, match="RUNNING"):
        dl.download(lambda *a: (0, 'k has status "KernelWorkerStatus.RUNNING"'), "u/k", tmp_path / "raw")
    assert not (tmp_path / "raw").exists()
