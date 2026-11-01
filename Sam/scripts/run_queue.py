"""Launch, monitor and download several Kaggle deployments with one command (the existing launch_kaggle.py,
monitor_kaggle.py and download_kaggle_outputs.py, in order). Dry run unless --yes.

The monitor's exit code is never trusted on its own: a crash of the local monitor also exits non-zero. After
every monitor run the queue asks Kaggle for the kernel's real status (`kernels status`, retried on API or
authentication errors) and decides from that:
  COMPLETE                 -> download (normal), continue
  ERROR / CANCEL_*         -> remote failure: raw download (--raw-only) for diagnosis; stop unless --keep-going
  RUNNING / QUEUED         -> the job is alive and only monitoring failed: re-attach the monitor (up to
                              --reattach times), never relaunch; then stop with the attach command
  unknown (API/auth error) -> stop; nothing is relaunched, the job (if any) keeps running

Adoption: a deployment that already has last_push.json was launched before. It is never pushed again (that
would start a new GPU version); the queue reads its remote status and continues from there: monitor a running
kernel, download a finished one (skipped when its export is already downloaded), report a failed one.

Sequential (default): one GPU session at a time. --parallel N launches up to N at once; use it only after
confirming on kaggle.com that the account may run N GPU sessions together (not visible through the CLI). The
code dataset is uploaded once, by the first new launch; later launches verify the remote listing equals the
local build (--skip-code-upload).
Items are deployment folders, or `full:MODEL`: prepare that model's full run from its downloaded preflight
(prepare_kaggle_deploy.py --preset full; skipped when the folder already exists), run it as above, and while the
downloaded export still has pending operations (a clean TIME_LIMIT stop) and the attempt made progress, prepare
and run the next attempt on that export (full_r2, full_r3, ... up to --max-attempts). A model whose preflight
is not downloaded (or failed) cannot be prepared: reported as prepare_failed.

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/run_queue.py Sam/kaggle/nllb/preflight Sam/kaggle/qwen3-4b/preflight          # dry run
  Sam/.venv/bin/python Sam/scripts/run_queue.py full:nllb Sam/kaggle/phi4-mini/preflight full:qwen3-4b \
      full:phi4-mini --keep-going --yes
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import kaggle_cli, paths  # noqa: E402

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent
ALIVE = {"RUNNING", "QUEUED", "NEW", "PENDING"}
FAILED = {"ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"}
AUTH_HINTS = ("401", "403", "unauthorized", "forbidden", "credentials", "authenticate")


def step(cmd: list, dry: bool, out=print) -> int:
    out("$ " + " ".join(cmd))
    return 0 if dry else subprocess.run(cmd).returncode


def remote_status(kernel: str, cli_run, tries: int = 4, wait_s: float = 15, sleep=time.sleep) -> tuple:
    """(status, kind, detail): kind is 'ok', 'auth' or 'api'. Retries transient API problems."""
    detail = ""
    for i in range(tries):
        try:
            rc, text = cli_run("kernels", "status", kernel)
        except Exception as e:  # noqa: BLE001  timeouts, missing CLI
            rc, text = -1, f"{type(e).__name__}: {e}"
        status, _ = kaggle_cli.parse_status(text)
        if status:
            return status, "ok", text.strip()[-300:]
        detail = text.strip()[-300:]
        if any(h in detail.lower() for h in AUTH_HINTS):
            return None, "auth", detail
        if i < tries - 1:
            sleep(wait_s)
    return None, "api", detail


def _deploy(d: str) -> dict:
    return json.loads((Path(d) / "deploy.json").read_text())


def downloaded(d: str) -> bool:
    dep = _deploy(d)
    return (paths.SAM_ROOT / dep["generated_root"] / "xsum" / dep["translator"] / "RESUME.json").exists()


def queue(dirs: list, yes: bool, parallel: int = 1, keep_going: bool = False, run=step, out=print,
          status=None, reattach: int = 2, state=None) -> dict:
    """Returns {deploy_dir: {"outcome": ..., ...}}. Outcomes: downloaded, already_downloaded, remote_failed,
    still_running (monitoring failed, job alive), status_unknown, launch_failed, dry_run."""
    status = status or (lambda k: remote_status(k, lambda *x: kaggle_cli.run(kaggle_cli.find_cli(), *x)))
    launch = lambda d, skip: [PY, str(SCRIPTS / "launch_kaggle.py"), "--deploy-dir", d, "--yes"] + (  # noqa: E731
        ["--skip-code-upload"] if skip else [])
    monitor = lambda d: [PY, str(SCRIPTS / "monitor_kaggle.py"), "--deploy-dir", d, "--interval", "60"]  # noqa: E731
    download = lambda d, ok: [PY, str(SCRIPTS / "download_kaggle_outputs.py"), "--deploy-dir", d] + (  # noqa: E731
        [] if ok else ["--raw-only"])
    state = state if state is not None else {}
    results, uploaded = {}, state.get("uploaded", False)       # one code upload per queue process
    for k in range(0, len(dirs), parallel):
        wave, active = dirs[k:k + parallel], []
        for d in wave:
            kernel = _deploy(d)["kernel"]
            if (Path(d) / "last_push.json").exists():           # launched before: adopt, never push again
                st, kind, detail = (status(kernel) if yes else ("(not checked: dry run)", "ok", ""))
                out(f"adopting {kernel} (already launched): remote status {st or kind}")
                results[d] = {"adopted": True, "remote_status": st}
                if not yes:
                    results[d]["outcome"] = "dry_run"
                elif st is None:
                    results[d].update(outcome="status_unknown", detail=f"{kind}: {detail}")
                elif st == "COMPLETE" and downloaded(d):
                    results[d]["outcome"] = "already_downloaded"
                else:
                    active.append(d)
                continue
            rc = run(launch(d, uploaded), not yes, out)
            uploaded = uploaded or rc == 0
            state["uploaded"] = uploaded
            results[d] = {"adopted": False, "launch": rc}
            if rc == 0:
                active.append(d)
            else:
                results[d]["outcome"] = "launch_failed"
        for d in active:
            kernel = _deploy(d)["kernel"]
            for attempt in range(reattach + 1):
                st = status(kernel)[0] if yes else None
                if yes and st == "COMPLETE":
                    break                                        # nothing to follow
                rc = run(monitor(d), not yes, out)
                results[d].setdefault("monitor", []).append(rc)
                if not yes:
                    break
                st, kind, detail = status(kernel)                # the truth, whatever the monitor said
                if st not in ALIVE:
                    break
                out(f"{kernel} is still {st} after the monitor exited ({rc}): local monitoring problem, "
                    f"re-attaching ({attempt + 1}/{reattach})" if attempt < reattach else
                    f"{kernel} is still {st}; not relaunched. Re-attach with: Sam/.venv/bin/python "
                    f"Sam/scripts/monitor_kaggle.py --deploy-dir {d}")
            if not yes:
                results[d]["download"] = run(download(d, True), True, out)
                results[d]["outcome"] = "dry_run"
                continue
            results[d]["remote_status"] = st
            if st == "COMPLETE":
                rc = run(download(d, True), False, out)
                results[d].update(download=rc, outcome="downloaded" if rc == 0 else "download_failed")
            elif st in FAILED:
                results[d].update(download=run(download(d, False), False, out), outcome="remote_failed")
            elif st in ALIVE:
                results[d]["outcome"] = "still_running"
            else:
                results[d].update(outcome="status_unknown", detail=f"{kind}: {detail}")
        bad = [d for d in wave if results[d]["outcome"] not in {"downloaded", "already_downloaded", "dry_run"}]
        if bad and not (keep_going and all(results[d]["outcome"] in {"remote_failed", "launch_failed"} for d in bad)):
            out(f"stopping: {[(d, results[d]['outcome']) for d in bad]}; later deployments not launched")
            break
    return results


OK = {"downloaded", "already_downloaded", "dry_run"}
CONTINUABLE = {"remote_failed", "launch_failed", "prepare_failed"}   # --keep-going may pass these, nothing else


def full_dir(model: str, attempt: int) -> Path:
    sys.path.insert(0, str(SCRIPTS))
    import model_runs as mr
    return paths.SAM_ROOT / "kaggle" / mr.MODELS[model][0] / mr.tag("full", attempt)


def remaining_ops(d: str) -> tuple:
    """(pending + not attempted operations, records made by that session) from its downloaded RESUME.json."""
    dep = _deploy(d)
    res = json.loads((paths.SAM_ROOT / dep["generated_root"] / "xsum" / dep["translator"] / "RESUME.json").read_text())
    left = sum(int(r.get("pending", 0)) + int(r.get("not_attempted", 0)) for r in res.get("attrition", []))
    return left, int(res.get("records_this_work_dir", 0))


def prepare_full(model: str, attempt: int, dry: bool, out=print, run=step) -> int:
    return run([PY, str(SCRIPTS / "prepare_kaggle_deploy.py"), "--preset", "full", "--model", model,
                "--resume-attempt", str(attempt)], dry, out)


def plan(items: list, yes: bool, keep_going: bool = False, run=step, out=print, status=None, reattach: int = 2,
         max_attempts: int = 4, prepare=None, remaining=remaining_ops) -> dict:
    """Run items in order, one GPU session at a time; `full:MODEL` items prepare and continue their own attempts."""
    prepare = prepare or (lambda m, k: prepare_full(m, k, not yes, out, run))
    state, results = {}, {}
    for item in items:
        if not item.startswith("full:"):
            r = queue([item], yes, 1, keep_going, run, out, status, reattach, state)
            results.update(r)
            outcomes = [x["outcome"] for x in r.values()]
        else:
            model, outcomes = item.split(":", 1)[1], []
            for attempt in range(1, max_attempts + 1):
                d = full_dir(model, attempt)
                if not (d / "deploy.json").exists():
                    if prepare(model, attempt) != 0 or (yes and not (d / "deploy.json").exists()):
                        results[f"{item}#{attempt}"] = {"outcome": "prepare_failed"}
                        outcomes.append("prepare_failed")
                        break
                    if not yes:
                        results[f"{item}#{attempt}"] = {"outcome": "dry_run", "note": "prepared at run time"}
                        outcomes.append("dry_run")
                        out(f"  (then: attempt {attempt + 1}, ... while operations remain, up to {max_attempts})")
                        break
                r = queue([str(d)], yes, 1, keep_going, run, out, status, reattach, state)
                results.update(r)
                outcome = r[str(d)]["outcome"]
                outcomes.append(outcome)
                if outcome not in OK or not yes:
                    break
                left, made = remaining(str(d))
                out(f"{model} attempt {attempt}: {left} operation(s) still pending")
                if left == 0:
                    break
                if made == 0:
                    out(f"{model}: attempt {attempt} made no progress; not continuing automatically")
                    outcomes.append("no_progress")
                    break
        bad = [o for o in outcomes if o not in OK]
        if bad and not (keep_going and all(o in CONTINUABLE for o in bad)):
            out(f"stopping after {item}: {bad}; later items not started")
            break
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("deploy_dirs", nargs="+", help="deployment folders and/or full:MODEL items, in order")
    ap.add_argument("--yes", action="store_true", help="really launch (GPU)")
    ap.add_argument("--parallel", type=int, default=1, help="GPU sessions at once (verify the account allows it)")
    ap.add_argument("--keep-going", action="store_true", help="continue after a REMOTE failure (never after an "
                                                               "unknown status or a still-running job)")
    ap.add_argument("--reattach", type=int, default=2, help="monitor re-attachments while the job is still alive")
    ap.add_argument("--max-attempts", type=int, default=4, help="full:MODEL sessions at most")
    a = ap.parse_args()
    for d in a.deploy_dirs:
        if not d.startswith("full:") and not (Path(d) / "deploy.json").exists():
            raise SystemExit(f"{d}: no deploy.json (prepare it first)")
    if not a.yes:
        print("DRY RUN (nothing launched, no status calls, nothing prepared). With --yes:")
    if a.parallel > 1:
        if any(d.startswith("full:") for d in a.deploy_dirs):
            raise SystemExit("--parallel takes deployment folders only")
        res = queue(a.deploy_dirs, a.yes, a.parallel, a.keep_going, reattach=a.reattach)
    else:
        res = plan(a.deploy_dirs, a.yes, a.keep_going, reattach=a.reattach, max_attempts=a.max_attempts)
    print(json.dumps(res, indent=1, default=str))
    ok = all(r["outcome"] in OK for r in res.values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
