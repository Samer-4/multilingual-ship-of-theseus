"""Stage and load the upstream Ship of Theseus corpus (read-only)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .files import sha256_file

STAGED_INFO = "STAGED.json"


def stage_corpus(src: Path, dst: Path, files: list, expected_commit: str) -> dict:
    """Copy the listed corpus files from a local upstream clone into `dst` (read-only), recording the commit
    and file hashes in dst/STAGED.json. Already-staged identical files are left alone."""
    src, dst = Path(src), Path(dst)
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != expected_commit:
        raise RuntimeError(f"corpus clone is at {commit}, config expects {expected_commit}")
    info = {"repo_commit": commit, "files": {}}
    for rel in files:
        s, d = src / rel, dst / rel
        if d.exists() and sha256_file(d) == sha256_file(s):
            pass
        elif d.exists():
            raise RuntimeError(f"{d} exists and differs from {s}; refusing to overwrite a staged corpus file")
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            d.chmod(0o444)
        info["files"][rel] = sha256_file(d)
    (dst / STAGED_INFO).write_text(json.dumps(info, indent=1))
    return info


def staged_info(corpus_dir: Path) -> dict:
    p = Path(corpus_dir) / STAGED_INFO
    return json.loads(p.read_text()) if p.exists() else {}


def paraphrased_file(dataset: str) -> str:
    return f"paraphrased_datasets/{dataset}_paraphrased.csv"


def load_paraphrased(corpus_dir: Path, dataset: str) -> pd.DataFrame:
    """The raw paraphrased CSV with `source_row_index` = 0-based data-row position in the file."""
    d = pd.read_csv(Path(corpus_dir) / paraphrased_file(dataset))
    return d.rename_axis("source_row_index").reset_index()
