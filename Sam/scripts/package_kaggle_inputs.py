"""Assemble the two folders to upload as PRIVATE Kaggle datasets. Builds them locally; uploads nothing.

  Sam/build/kaggle/sam-sot-mt-code/     Sam/src, Sam/configs, Sam/scripts, Sam/notebooks, Sam/data/manifests
  Sam/build/kaggle/sam-sot-mt-staged/   t0_manifest.parquet, generations.parquet, STAGED_FILES.json

Both get a dataset-metadata.json. The licence is declared "unknown": the upstream corpus repository has no
LICENSE file, and this script does not guess one. Every packaged file is scanned for credentials (Hugging
Face / Kaggle / cloud keys, private keys, e-mail addresses, credential files); a hit aborts the build.

Usage (from the repo root):
  Sam/.venv/bin/python Sam/scripts/package_kaggle_inputs.py --kaggle-user YOUR_KAGGLE_USERNAME
Then upload each folder yourself (the Kaggle CLI creates PRIVATE datasets unless --public is given):
  kaggle datasets create -p Sam/build/kaggle/sam-sot-mt-code --dir-mode zip
  kaggle datasets create -p Sam/build/kaggle/sam-sot-mt-staged
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot_mt import paths, session  # noqa: E402
from sot_mt.files import CODE_MANIFEST, code_digest, code_manifest  # noqa: E402

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints", ".DS_Store")
CODE_PARTS = ["src", "configs", "scripts", "notebooks", "data/manifests"]
SECRET_PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "kaggle_key": re.compile(rb"[\"']key[\"']\s*:\s*[\"'][0-9a-f]{32}[\"']"),
    "aws_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "api_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "email": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}"),
}
CREDENTIAL_FILES = {"kaggle.json", ".env", ".netrc", "token", ".git-credentials", "credentials.json"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".ipynb", ".md", ".txt", ".csv", ".cfg", ".toml"}


class SecretFoundError(RuntimeError):
    pass


def metadata(user: str, slug: str, title: str, description: str) -> dict:
    return {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "unknown"}], "description": description}


CODE_DESCRIPTION = ("Code and configuration for a CS6120 course project (round-trip translation of the Ship of "
                    "Theseus XSum subset). Private working copy; not for redistribution.")
STAGED_DESCRIPTION = ("Stage 2 tables for a CS6120 course project: 200 XSum texts from the Ship of Theseus corpus "
                      "(tripto03/Ship_of_theseus_paraphrased_copus @ 4423d8c) and their MADLAD round-1 translations. "
                      "The upstream repository has no licence file, so the licence is unknown. Private working "
                      "copy; not for redistribution.")


def scan_for_secrets(root: Path) -> list:
    """[(relative path, finding)] for credential-looking content or file names under root."""
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if p.name in CREDENTIAL_FILES:
            hits.append((str(rel), "credential file name"))
        if p.suffix in TEXT_SUFFIXES:
            data = p.read_bytes()
            for name, rx in SECRET_PATTERNS.items():
                if rx.search(data):
                    hits.append((str(rel), name))
    return hits


def build(out: Path, user: str, staged_dir: Path) -> dict:
    code = out / "sam-sot-mt-code"
    if code.exists():
        shutil.rmtree(code)
    for sub in CODE_PARTS:
        shutil.copytree(paths.SAM_ROOT / sub, code / "Sam" / sub, ignore=IGNORE)
    shutil.copy2(paths.SAM_ROOT / "requirements.txt", code / "Sam" / "requirements.txt")
    manifest = code_manifest(code / "Sam")
    (code / "Sam" / CODE_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (code / "dataset-metadata.json").write_text(json.dumps(
        metadata(user, "sam-sot-mt-code", "sam-sot-mt-code", CODE_DESCRIPTION), indent=1))

    staged = out / "sam-sot-mt-staged"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    info = session.verify_staged(staged_dir)          # refuse to package modified inputs
    for name in [*info["files"], "STAGED_FILES.json"]:
        shutil.copy2(staged_dir / name, staged / name)
    session.verify_staged(staged)                     # and check the copies
    (staged / "dataset-metadata.json").write_text(json.dumps(
        metadata(user, "sam-sot-mt-staged", "sam-sot-mt-staged", STAGED_DESCRIPTION), indent=1))

    hits = scan_for_secrets(code) + scan_for_secrets(staged)
    if hits:
        shutil.rmtree(code), shutil.rmtree(staged)
        raise SecretFoundError(f"possible credentials, nothing packaged: {hits}")
    return {"code": code, "staged": staged, "code_digest": code_digest(manifest)}


def listing(root: Path) -> list:
    return [(str(p.relative_to(root)), p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-user", default="YOUR_KAGGLE_USERNAME")
    ap.add_argument("--out", default=str(paths.SAM_ROOT / "build" / "kaggle"))
    a = ap.parse_args()
    built = build(Path(a.out), a.kaggle_user, paths.staged_dir())
    for name, root in [(k, built[k]) for k in ("code", "staged")]:
        files = listing(root)
        print(f"\n{root}  ({len(files)} files, {sum(s for _, s in files):,d} bytes)")
        for rel, size in files:
            print(f"  {size:>10,d}  {rel}")
    print(f"\ncode digest {built['code_digest']}; staged hashes verified; no credentials found. "
          "Upload the folders yourself as PRIVATE datasets.")


if __name__ == "__main__":
    main()
