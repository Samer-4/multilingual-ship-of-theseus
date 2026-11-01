"""Path resolution for local runs and Kaggle. No absolute path to a personal machine appears anywhere.

Lookup order for each input root: environment variable, then the Kaggle input mount, then the Sam default.
  corpus   SOT_CORPUS_DIR   /kaggle/input/sot-corpus-4423d8c     Sam/data/raw/sot_corpus
  staged   SAM_STAGED_DIR   /kaggle/input/sam-sot-mt-staged      Sam/data/staged
Writable outputs go to /kaggle/working/sam_outputs on Kaggle (inputs are read-only there), else Sam/.
"""
from __future__ import annotations

import os
from pathlib import Path

SAM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SAM_ROOT.parent
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_SLUGS = {"corpus": "sot-corpus-4423d8c", "staged": "sam-sot-mt-staged"}


def on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or KAGGLE_INPUT.is_dir()


def _resolve(env: str, kaggle_slug: str, default: Path) -> Path:
    if os.environ.get(env):
        return Path(os.environ[env])
    if on_kaggle():
        return KAGGLE_INPUT / kaggle_slug
    return default


def corpus_dir() -> Path:
    return _resolve("SOT_CORPUS_DIR", KAGGLE_SLUGS["corpus"], SAM_ROOT / "data" / "raw" / "sot_corpus")


def staged_dir() -> Path:
    return _resolve("SAM_STAGED_DIR", KAGGLE_SLUGS["staged"], SAM_ROOT / "data" / "staged")


def output_root() -> Path:
    """Where new files are written: Sam/ locally, /kaggle/working/sam_outputs on Kaggle."""
    if os.environ.get("SAM_OUTPUT_DIR"):
        return Path(os.environ["SAM_OUTPUT_DIR"])
    return KAGGLE_WORKING / "sam_outputs" if on_kaggle() else SAM_ROOT


def repo_file(rel: str) -> Path:
    """A team file in this repository (read-only). Only available where the repo is checked out."""
    return REPO_ROOT / rel


def config_path(name: str) -> Path:
    return SAM_ROOT / "configs" / name
