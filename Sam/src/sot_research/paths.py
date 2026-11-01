"""Locations of the research inputs, local (text-bearing, gitignored) tables and committed (text-free) results."""
from __future__ import annotations

import os
from pathlib import Path

SAM_ROOT = Path(__file__).resolve().parents[2]
GENERATED = SAM_ROOT / "data" / "generated"           # finalized translation datasets (gitignored, contain text)
STAGED = SAM_ROOT / "data" / "staged"                 # Stage 2 manifest (gitignored, contains T0 text)
LOCAL = Path(os.environ.get("SOT_RESEARCH_LOCAL", SAM_ROOT / "data" / "research"))   # text tables + caches
CACHE = LOCAL / "cache"
RESULTS = Path(os.environ.get("SOT_RESEARCH_RESULTS", SAM_ROOT / "results" / "research"))   # committed, text-free
CONFIG = SAM_ROOT / "configs"

SUBDIRS = ["data_coverage", "shared_features", "metrics", "rq1", "rq2", "rq3", "figures", "run_manifests"]


def results(sub: str) -> Path:
    p = RESULTS / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def local(name: str) -> Path:
    LOCAL.mkdir(parents=True, exist_ok=True)
    return LOCAL / name


def cache(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name
