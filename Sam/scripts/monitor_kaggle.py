"""Monitor a running Kaggle notebook from the terminal: status, live log, progress bar, ETA, errors.

Sources (Kaggle CLI 2.2.5, nothing else):
  * `kaggle kernels status K`     polled every --interval seconds; QUEUED / RUNNING / COMPLETE / ERROR / CANCEL_*
  * `kaggle kernels logs -f K`    the live log stream, read line by line in a background thread
  * `kaggle kernels logs K`       the stored log, used when the live stream is unavailable or has ended
Progress is shown only from SOTMT_PROGRESS events the notebook printed; with none yet, it says so.

Exit codes: 0 notebook completed (and reported notebook_done), 1 Kaggle status ERROR, 2 cancelled,
3 completed without a notebook_done event (check the log), 4 gave up waiting (--max-wait-min), 5 CLI problem.

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/monitor_kaggle.py                          # kernel from the smoke deployment
  Sam/.venv/bin/python Sam/scripts/monitor_kaggle.py --kernel samahmed17/sam-sot-mt-madlad-smoke5 --interval 20
  Sam/.venv/bin/python Sam/scripts/monitor_kaggle.py --log-file Sam/results/madlad_smoke5/raw/<stamp>/x.log
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import kaggle_cli, paths  # noqa: E402
from sot_mt.progress import fmt_seconds  # noqa: E402

DEFAULT_DEPLOY = paths.SAM_ROOT / "kaggle" / "madlad_smoke5"
BAR = 30


def bar(done, total) -> str:
    if not total:
        return "[" + "?" * BAR + "]"
    n = int(round(BAR * min(done, total) / total))
    return "[" + "#" * n + "-" * (BAR - n) + "]"


def render(ev, status=None, age_s=None) -> str:
    """One readable line from the latest progress event (or the plain status before any event)."""
    st = f"[{status or '?'}]"
    if ev is None:
        return f"{st} no progress events yet (setup, dependency install or queueing)"
    parts = [st, f"{ev.get('translator', '?')} ({ev.get('model', '?')}) {ev.get('dataset', '')}".strip(),
             f"phase {ev.get('phase')}"]
    if ev.get("total"):
        parts.append(f"{bar(ev['done'], ev['total'])} {ev['done']}/{ev['total']} ops {ev.get('pct', 0):.1f}%")
        parts.append(f"valid {ev.get('valid', 0)} invalid {ev.get('invalid', 0)} failed {ev.get('failed', 0)}"
                     f" dropped {ev.get('dropped', 0)}")
    if ev.get("current_item_id"):
        cur = f"now {ev['current_item_id']} {ev.get('current_stage')}"
        if ev.get("current_elapsed_s") is not None:
            cur += f" ({fmt_seconds(ev['current_elapsed_s'])})"
        parts.append(cur)
    parts.append(f"elapsed {fmt_seconds(ev.get('elapsed_s'))}")
    if ev.get("load_s") is not None:
        parts.append(f"model load {fmt_seconds(ev['load_s'])}")
    if ev.get("total"):
        parts.append(ev.get("eta_text") or "ETA calculating")
    if ev.get("ops_per_min") is not None:
        parts.append(f"{ev['ops_per_min']} ops/min, {ev.get('words_per_s')} words/s")
    if age_s is not None:
        parts.append(f"last event {fmt_seconds(age_s)} ago")
    return " | ".join(parts)


class Monitor:
    def __init__(self, kernel, cli_run, follow_cmd=None, interval=30, max_wait_min=None, out=print,
                 clock=time.monotonic, sleep=time.sleep):
        self.kernel, self.cli_run, self.follow_cmd = kernel, cli_run, follow_cmd
        self.interval, self.max_wait_s = interval, (max_wait_min * 60 if max_wait_min else None)
        self.out, self.clock, self.sleep = out, clock, sleep
        self.parser = kaggle_cli.EventParser()
        self.q: "queue.Queue" = queue.Queue()
        self.follow_alive = False
        self.follow_lines = 0
        self.status = None
        self.failure = None
        self.last_line = None
        self.last_event_at = None
        self.notes = set()

    # -- sources -----------------------------------------------------------------------------------
    def start_follow(self):
        if not self.follow_cmd:
            return
        try:
            proc = subprocess.Popen(self.follow_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
        except OSError as e:
            self.note(f"live log stream unavailable ({e}); polling stored logs")
            return
        self.follow_alive = True

        def reader():
            for line in proc.stdout:
                self.q.put(line.rstrip("\n"))
            proc.wait()
            self.q.put(None)                      # end of stream
        threading.Thread(target=reader, daemon=True).start()

    def poll_status(self):
        rc, text = self.cli_run("kernels", "status", self.kernel)
        status, failure = kaggle_cli.parse_status(text)
        if status is None:
            self.note(f"could not read status (rc {rc}): {text.strip()[:200]}")
        else:
            self.status, self.failure = status, failure or self.failure
        return self.status

    def poll_logs(self):
        rc, text = self.cli_run("kernels", "logs", self.kernel)
        if rc == 0:
            self.feed(kaggle_cli.log_lines(text))
        else:
            self.note(f"stored log not available yet (rc {rc})")

    def drain(self):
        lines = []
        while True:
            try:
                line = self.q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.follow_alive = False
                if self.status not in kaggle_cli.TERMINAL:
                    self.note("live log stream ended before the session finished; polling stored logs")
            else:
                lines.append(line)
        self.follow_lines += len(lines)
        self.feed(lines)

    def feed(self, lines):
        new = self.parser.feed(lines)
        if new:
            self.last_event_at = self.clock()
            for ev in new:
                if ev.get("event") in {"op_done", "phase", "model_loaded", "run_start", "run_end", "parity_done",
                                       "probe_done", "probe_model_loaded", "export_done", "notebook_done",
                                       "notebook_start", "quality_gate", "gpu_memory"}:
                    self.show(ev)

    # -- output ----------------------------------------------------------------------------------
    def note(self, msg):
        if msg not in self.notes:
            self.notes.add(msg)
            self.out(f"note: {msg}")

    def show(self, ev=None):
        ev = ev or self.parser.last_of("op_done", "op_start", "heartbeat", "phase", "model_loaded", "run_start",
                                       "quality_gate",
                                       "run_end", "export_done", "notebook_done", "parity_done", "probe_done",
                                       "notebook_start", "probe_model_loaded")
        age = None if self.last_event_at is None else self.clock() - self.last_event_at
        try:
            line = render(ev, self.status, age)
        except Exception as e:  # noqa: BLE001  a malformed event must not stop the monitor
            line = f"[{self.status or '?'}] event {ev.get('event') if isinstance(ev, dict) else ev!r} " \
                   f"not renderable ({type(e).__name__}: {e})"
        if ev is not None:
            detail = describe(ev)
            if detail:
                line += "\n" + detail
        if line != self.last_line:
            self.out(line)
            self.last_line = line

    # -- main loop -------------------------------------------------------------------------------
    def run(self) -> int:
        t0 = self.clock()
        if self.poll_status() is None and self.status is None:
            return 5
        self.out(f"monitoring {self.kernel}: status {self.status}")
        self.start_follow()
        next_poll = self.clock() + self.interval
        while True:
            self.drain()
            if self.clock() >= next_poll:
                next_poll = self.clock() + self.interval
                self.poll_status()
                if not self.follow_alive and self.status not in kaggle_cli.TERMINAL:
                    self.poll_logs()
                if self.status == "RUNNING" and self.parser.last is None and self.follow_lines == 0 \
                        and self.clock() - t0 > 3 * self.interval:
                    self.note("no log lines received yet: Kaggle may expose the log only with a delay")
                self.show()
            if self.status in kaggle_cli.TERMINAL:
                return self.finish()
            if self.max_wait_s and self.clock() - t0 > self.max_wait_s:
                self.out("gave up waiting (--max-wait-min); the notebook keeps running on Kaggle")
                return 4
            self.sleep(1)

    def finish(self) -> int:
        deadline = self.clock() + 20
        while self.follow_alive and self.clock() < deadline:
            self.drain()
            self.sleep(1)
        self.poll_logs()                          # the stored log is complete once the session has ended
        self.parser.close()
        self.show()
        if self.parser.malformed:
            self.out(f"note: {self.parser.malformed} malformed progress line(s) ignored")
        if self.parser.errors:
            self.out("errors / tracebacks seen in the log (last 8):")
            for e in self.parser.errors[-8:]:
                self.out("  " + e)
        if self.status == "COMPLETE":
            done = self.parser.last_of("notebook_done")
            if done is None:
                self.out("Kaggle reports COMPLETE, but the notebook never reported notebook_done: inspect the log")
                return 3
            self.out(f"COMPLETE: notebook_done (run status {done.get('status')}). Download with "
                     "Sam/.venv/bin/python Sam/scripts/download_kaggle_outputs.py")
            return 0
        self.out(f"{self.status}: " + (f'failure message "{self.failure}"' if self.failure else "no failure message"))
        return kaggle_cli.TERMINAL[self.status]


def probe_text(e: dict) -> str:
    """probe_done comes in three kinds: MADLAD's drift probe and wiring diagnostic (compared with the stored
    round-1 S1: identical / similarity) and the other translators' sanity probe (fixed sentences: ok, words
    found). Events from before the `probe` field are recognised by their fields."""
    kind = e.get("probe") or ("sanity" if str(e.get("item_id", "")).startswith("sanity")
                              else "diagnostic" if "variant" in e else "drift")
    secs = f" ({e['op_seconds']}s)" if e.get("op_seconds") is not None else ""
    err = f" error {e['error']}" if e.get("error") else ""
    if kind == "sanity":
        found = f", expected words {e['expected_words_found']}" if e.get("expected_words_found") else ""
        block = f", blocking {e['blocking']}" if e.get("blocking") else ""
        return f"  -> sanity probe {e.get('direction') or e.get('item_id')}: ok={e.get('ok')}{found}{block}{secs}{err}"
    ref = f"identical={e.get('identical')}" + (f" similarity={e['similarity']}" if e.get("similarity") is not None else "")
    variant = f" [{e['variant']}]" if e.get("variant") else ""
    return f"  -> {kind} probe {e.get('item_id')}{variant}: {ref} ok={e.get('ok')}{secs}{err}"


DESCRIBE = {
    "op_done": lambda e: f"  -> {e.get('item_id')} {e.get('stage')}: {e.get('status')}"
    f" in {e.get('op_seconds')}s, {e.get('n_segments')} segment(s)"
    + (f", {e['reason']}" if e.get("reason") else "")
    + (f", {e['dropped_after']} later stage(s) dropped" if e.get("dropped_after") else ""),
    "parity_done": lambda e: f"  -> tokenizer parity {e.get('matched')}/{e.get('checked')}",
    "probe_done": probe_text,
    "probe_model_loaded": lambda e: f"  -> model loaded in {e.get('probe_load_s')}s"
    + (f", weights {e['load_report'].get('weights_id')}" if isinstance(e.get("load_report"), dict)
       and e["load_report"].get("weights_id") else ""),
    "gpu_memory": lambda e: "  -> peak GPU memory " + ", ".join(
        f"GPU{g.get('device')} {g.get('max_allocated_gb')} GB" for g in (e.get("gpu") or []) if isinstance(g, dict)),
    "run_end": lambda e: f"  -> run finished: {e.get('status')}" + (f" ({e['error']})" if e.get("error") else ""),
    "quality_gate": lambda e: f"  -> QUALITY GATE stopped the run ({e.get('source', 'runner')}): "
    f"{e.get('degenerate', e.get('failed'))} degenerate output(s)",
    "export_done": lambda e: f"  -> exported {e.get('files')} files to {e.get('export_dir')}",
}


def describe(ev: dict):
    """The detail line for an event, or None. A malformed or unexpected event never stops the monitor: it is
    shown with its raw fields instead (errors in the notebook itself are reported from the log and run_end)."""
    f = DESCRIBE.get(ev.get("event"))
    if f is None:
        return None
    try:
        return f(ev)
    except Exception as e:  # noqa: BLE001
        raw = {k: v for k, v in ev.items() if k not in {"v", "seq", "ts"}}
        return f"  -> {ev.get('event')} (unexpected fields, {type(e).__name__}: {e}): {json.dumps(raw, default=str)[:300]}"


def from_log_file(path: Path, out=print) -> int:
    p = kaggle_cli.EventParser()
    p.feed(kaggle_cli.log_lines(Path(path).read_text(errors="replace")))
    p.close()
    out(render(p.last, "LOG FILE"))
    for ev in p.events:
        if ev.get("event") == "op_done":
            out(f"  {ev.get('item_id')} {ev.get('stage')}: {ev.get('status')} {ev.get('op_seconds')}s")
    out(f"{len(p.events)} events, {p.malformed} malformed")
    return 0 if p.last_of("notebook_done") else 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", help="<owner>/<slug>; default: from the deployment's deploy.json")
    ap.add_argument("--deploy-dir", default=str(DEFAULT_DEPLOY))
    ap.add_argument("--kaggle", help="path to the Kaggle CLI")
    ap.add_argument("--interval", type=float, default=30, help="seconds between status polls")
    ap.add_argument("--no-follow", action="store_true", help="poll the stored log instead of streaming")
    ap.add_argument("--max-wait-min", type=float)
    ap.add_argument("--log-file", help="parse a downloaded log instead of contacting Kaggle")
    a = ap.parse_args()
    if a.log_file:
        sys.exit(from_log_file(Path(a.log_file)))
    kernel = a.kernel or json.loads((Path(a.deploy_dir) / "deploy.json").read_text())["kernel"]
    try:
        cli = kaggle_cli.find_cli(a.kaggle)
    except FileNotFoundError as e:
        print(e)
        sys.exit(5)
    m = Monitor(kernel, lambda *args: kaggle_cli.run(cli, *args),
                follow_cmd=None if a.no_follow else [cli, "kernels", "logs", "-f", kernel],
                interval=a.interval, max_wait_min=a.max_wait_min)
    try:
        sys.exit(m.run())
    except KeyboardInterrupt:
        print("\nstopped monitoring; the notebook keeps running on Kaggle")
        sys.exit(130)


if __name__ == "__main__":
    main()
