"""Build the canonical research corpus and compute/cache every shared feature and pairwise metric.

Usage (from the repo root, research environment):
  Sam/.venv-research/bin/python Sam/scripts/research/compute_features.py              # everything (resumes)
  Sam/.venv-research/bin/python Sam/scripts/research/compute_features.py --corpus-only
  Sam/.venv-research/bin/python Sam/scripts/research/compute_features.py --skip-neural
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sot_research import pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-only", action="store_true")
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.corpus_only:
        pipeline.build_corpus()
    else:
        pipeline.run(skip_neural=a.skip_neural, workers=a.workers)
