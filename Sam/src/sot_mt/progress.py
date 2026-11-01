"""Live progress events for the translation runner: one JSON object per stdout line, flushed immediately.

  SOTMT_PROGRESS {"v": 1, "seq": 7, "event": "op_done", "phase": "translating", "done": 3, "total": 20, ...}

scripts/monitor_kaggle.py parses these lines out of the Kaggle log. Every number is measured or counted,
never assumed:

  * total     operations this run expects to perform, planned per item: from its next missing stage to
              the last selected stage. Items whose chain is already blocked (invalid or missing
              predecessor, retries exhausted) are not planned at all.
  * done      completed operations (valid, invalid or failed) + planned operations dropped because an
              earlier stage of the same item came out invalid or failed in this run. So progress reaches
              100% even when chains stop early.
  * ETA       measured translation seconds per input word over this run's completed operations, times the
              estimated input words still to go (an item's next input is its latest output, so later stages
              are estimated from the most recent known text). Words track the number of segments a document
              needs, so long multi-segment documents weigh more. Until `min_samples` operations have
              finished, eta_s is null and eta_text is "ETA calculating". Model loading (including any
              download on first use) is timed separately (load_s) and never enters the ETA.

A heartbeat thread repeats the current state every `heartbeat_s` seconds while nothing else is emitted
(e.g. during a long model download or a long document), so a silent log means a stalled session.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

PREFIX = "SOTMT_PROGRESS "
VERSION = 1


def words(text) -> int:
    return len(str(text).split()) if isinstance(text, str) else 0


def fmt_seconds(s: Optional[float]) -> str:
    if s is None:
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s" if m else f"{sec}s"


class Progress:
    def __init__(self, emit: Optional[Callable[[str], None]] = None, clock=time.monotonic, heartbeat_s: float = 30,
                 min_samples: int = 2):
        self.emit_line, self.clock, self.heartbeat_s, self.min_samples = emit, clock, heartbeat_s, min_samples
        self.context: Dict = {}
        self.phase = "idle"
        self.seq = 0
        self.t0 = clock()
        self.last_emit = self.t0
        self.load_s: Optional[float] = None
        self.translate_t0: Optional[float] = None
        self.plan: Dict[str, List[str]] = {}          # item -> remaining planned stages, in order
        self.next_words: Dict[str, int] = {}          # item -> words of its next input
        self.total = 0
        self.counts = {"valid": 0, "invalid": 0, "failed": 0, "dropped": 0}
        self.op_seconds = 0.0
        self.op_words = 0
        self.segments = 0
        self.current: Dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def stdout(cls, **kw) -> "Progress":
        return cls(emit=lambda line: (sys.stdout.write(line + "\n"), sys.stdout.flush()), **kw)

    # -- state -----------------------------------------------------------------------------------
    @property
    def completed(self) -> int:
        return self.counts["valid"] + self.counts["invalid"] + self.counts["failed"]

    @property
    def done(self) -> int:
        return self.completed + self.counts["dropped"]

    def remaining_words(self) -> int:
        return sum(self.next_words.get(i, 0) * len(st) for i, st in self.plan.items())

    def eta(self) -> Optional[float]:
        if self.completed < self.min_samples or self.op_words <= 0:
            return None
        return self.op_seconds / self.op_words * self.remaining_words()

    def snapshot(self) -> dict:
        now = self.clock()
        eta = self.eta()
        tr = None if self.translate_t0 is None else now - self.translate_t0
        snap = {"phase": self.phase, **self.context, "done": self.done, "total": self.total,
                "pct": round(100.0 * self.done / self.total, 1) if self.total else (100.0 if self.phase == "finished" else 0.0),
                **self.counts, "elapsed_s": round(now - self.t0, 1), "load_s": self.load_s,
                "translate_elapsed_s": None if tr is None else round(tr, 1),
                "eta_s": None if eta is None else round(eta, 1),
                "eta_text": "ETA calculating" if eta is None else f"ETA {fmt_seconds(eta)}",
                "segments_done": self.segments}
        if self.op_seconds > 0:
            snap["ops_per_min"] = round(60.0 * self.completed / self.op_seconds, 2)
            snap["words_per_s"] = round(self.op_words / self.op_seconds, 1)
        if self.current:
            snap.update({f"current_{k}": v for k, v in self.current.items()})
            if "started" in self.current:
                snap["current_elapsed_s"] = round(now - self.current["started"], 1)
                snap.pop("current_started")
        return snap

    def event(self, event: str, **fields) -> dict:
        with self._lock:
            self.seq += 1
            rec = {"v": VERSION, "seq": self.seq, "event": event,
                   "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **self.snapshot(), **fields}
            self.last_emit = self.clock()
            if self.emit_line:
                self.emit_line(PREFIX + json.dumps(rec, default=str, separators=(",", ":")))
            return rec

    # -- lifecycle used by the runner ------------------------------------------------------------
    def start(self, context: dict, plan: Dict[str, List[str]], next_words: Dict[str, int]) -> None:
        self.stop_heartbeat()                         # a re-run starts from clean counters; seq keeps growing
        self._stop = threading.Event()
        self.t0 = self.last_emit = self.clock()
        self.load_s = self.translate_t0 = None
        self.counts = {"valid": 0, "invalid": 0, "failed": 0, "dropped": 0}
        self.op_seconds, self.op_words, self.segments, self.current = 0.0, 0, 0, {}
        self.context = dict(context)
        self.plan = {i: list(st) for i, st in plan.items() if st}
        self.next_words = dict(next_words)
        self.total = sum(len(st) for st in self.plan.values())
        self.event("run_start", planned_items=len(self.plan))
        if self.emit_line and self.heartbeat_s:
            self._thread = threading.Thread(target=self._beat, daemon=True)
            self._thread.start()

    def set_phase(self, phase: str, emit: bool = True, **fields) -> None:
        self.phase = phase
        if emit:
            self.event("phase", **fields)

    def loaded(self, seconds: float, already_loaded: bool = False) -> None:
        self.load_s = round(seconds, 1)
        self.translate_t0 = self.clock()
        self.set_phase("translating", emit=False)
        self.event("model_loaded", already_loaded=already_loaded)

    def op_start(self, item_id: str, stage: str, input_text: str) -> None:
        self.phase = "translating"
        self.current = {"item_id": item_id, "stage": stage, "input_words": words(input_text), "started": self.clock()}
        self.event("op_start")

    def op_done(self, item_id: str, stage: str, status: str, seconds: float, input_text: str,
                output_text: Optional[str], n_segments: Optional[int] = None, reason: Optional[str] = None) -> None:
        """status: "valid" | "invalid" | "failed". Anything but valid drops the rest of the item's plan."""
        with self._lock:
            self.counts[status] += 1
            self.op_seconds += max(0.0, seconds)
            self.op_words += max(1, words(input_text))
            self.segments += int(n_segments or 0)
            st = self.plan.get(item_id, [])
            if stage in st:
                st.remove(stage)
            dropped = 0
            if status == "valid":
                self.next_words[item_id] = words(output_text) or self.next_words.get(item_id, 0)
            else:
                dropped = len(st)
                self.counts["dropped"] += dropped
                st.clear()
            if not st:
                self.plan.pop(item_id, None)
            self.current = {}
        self.event("op_done", item_id=item_id, stage=stage, status=status, op_seconds=round(seconds, 2),
                   n_segments=n_segments, reason=reason, dropped_after=dropped)

    def finish(self, status: str, **fields) -> None:
        self.stop_heartbeat()
        self.phase = "finished" if status == "completed" else status
        self.event("run_end", status=status, **fields)

    # -- heartbeat ---------------------------------------------------------------------------------
    def _beat(self) -> None:
        while not self._stop.wait(min(self.heartbeat_s, 5.0)):
            if self.clock() - self.last_emit >= self.heartbeat_s:
                self.event("heartbeat")

    def stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
