"""Thin wrappers around the Kaggle CLI (2.2.5) plus parsers for its output. Standard library only.

Verified against the installed CLI source (kaggle/api/kaggle_api_extended.py, kagglesdk 0.1.38):
  * `kaggle kernels status K`   prints `K has status "KernelWorkerStatus.RUNNING"` (+ `Failure message: "..."`);
    states QUEUED, RUNNING, COMPLETE, ERROR, CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED, NEW_SCRIPT.
  * `kaggle kernels logs -f K`  streams the live session log (server-sent events; replays from the start on
    reconnect; for a finished session it returns the stored log).
  * `kaggle kernels logs K`     the stored log, which can be a JSON array of {stream_name, time, data}.
  * `kaggle kernels output K -p DIR`  downloads the output files of the latest run, plus `<slug>.log`.
  * `kaggle kernels push -p DIR [--accelerator A] [-t SECONDS]` saves a version and runs it; prints
    "Kernel version N successfully pushed.  Please check progress at URL".
Nothing here is invented progress: events come only from SOTMT_PROGRESS lines the notebook printed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .progress import PREFIX

TERMINAL = {"COMPLETE": 0, "ERROR": 1, "CANCEL_ACKNOWLEDGED": 2, "CANCEL_REQUESTED": 2}
STATUS_RX = re.compile(r'has status "([^"]*)"')
FAILURE_RX = re.compile(r'Failure message: "(.*)"', re.S)
PUSH_RX = re.compile(r"Kernel version (\d+) successfully pushed")


def find_cli(explicit: Optional[str] = None) -> str:
    for c in [explicit, os.environ.get("KAGGLE_CLI"), str(Path.home() / ".venvs/kaggle-cli/bin/kaggle"),
              shutil.which("kaggle")]:
        if c and Path(c).expanduser().exists():
            return str(Path(c).expanduser())
    raise FileNotFoundError("Kaggle CLI not found: pass --kaggle or set KAGGLE_CLI")


def run(cli: str, *args: str, timeout: float = 120) -> Tuple[int, str]:
    p = subprocess.run([cli, *args], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def parse_status(text: str) -> Tuple[Optional[str], Optional[str]]:
    """('RUNNING' | 'COMPLETE' | ... | None if unparseable, failure message or None)."""
    m = STATUS_RX.search(text or "")
    if not m:
        return None, None
    status = m.group(1).split(".")[-1].strip().upper()
    f = FAILURE_RX.search(text)
    return status, (f.group(1) if f else None)


def parse_push(text: str) -> Optional[int]:
    m = PUSH_RX.search(text or "")
    return int(m.group(1)) if m else None


def log_lines(text: str) -> List[str]:
    """Plain lines from either log format (JSON array of {data} events, or already-printed text)."""
    text = text or ""
    s = text.lstrip()
    if s.startswith("["):
        try:
            events = json.loads(s)
            return "".join(e.get("data", "") for e in events if isinstance(e, dict)).splitlines()
        except ValueError:
            pass
    return text.splitlines()


class EventParser:
    """Feed log lines; collects progress events (deduplicated by seq), malformed counts and error lines.
    A progress line split across two log chunks is re-joined with the next line once."""

    def __init__(self):
        self.events: List[dict] = []
        self.seen = set()
        self.malformed = 0
        self.errors: List[str] = []
        self._pending: Optional[str] = None

    def feed(self, lines: Iterable[str]) -> List[dict]:
        new = []
        for line in lines:
            if self._pending is not None:
                joined, self._pending = self._pending + line, None
                ev = self._parse(joined)
                if ev is None:
                    self.malformed += 1
                else:
                    new += self._add(ev)
                    continue
            if PREFIX.strip() in line:
                ev = self._parse(line)
                if ev is None:
                    self._pending = line
                else:
                    new += self._add(ev)
            elif re.search(r"Traceback|Error\b|Exception\b|CUDA out of memory|Killed", line):
                self.errors.append(line.strip()[:300])
        return new

    def close(self) -> None:
        if self._pending is not None:
            self.malformed += 1
            self._pending = None

    @staticmethod
    def _parse(line: str) -> Optional[dict]:
        i = line.find(PREFIX.strip())
        try:
            ev = json.loads(line[i + len(PREFIX.strip()):].strip())
        except ValueError:
            return None
        return ev if isinstance(ev, dict) and "event" in ev and "seq" in ev else None

    def _add(self, ev: dict) -> List[dict]:
        key = (ev.get("run_id"), ev.get("seq"), ev.get("event"))
        if key in self.seen:
            return []
        self.seen.add(key)
        self.events.append(ev)
        return [ev]

    @property
    def last(self) -> Optional[dict]:
        return self.events[-1] if self.events else None

    def last_of(self, *names: str) -> Optional[dict]:
        for ev in reversed(self.events):
            if ev.get("event") in names:
                return ev
        return None
