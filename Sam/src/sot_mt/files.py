"""Hashing, config loading and write-once file helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from . import paths


class ImmutableFileError(RuntimeError):
    """Raised when a write would change an existing immutable file."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_id(obj) -> str:
    """Stable id for a JSON-serializable config: sha256 of canonical JSON, first 12 hex."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")))[:12]


CODE_MANIFEST = "CODE_MANIFEST.json"


def code_manifest(root: Path) -> dict:
    """{relative path: sha256} for every file under root (the packaged Sam/ folder), except the manifest."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != CODE_MANIFEST and "__pycache__" not in p.parts}


def code_digest(manifest: dict) -> str:
    """One id for a packaged code tree: sha256 of its canonical manifest, first 16 hex."""
    return sha256_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))[:16]


def verify_code_tree(root: Path, expected_digest=None) -> str:
    """Check the files under root against root/CODE_MANIFEST.json (and the digest, if given); return the digest."""
    root = Path(root)
    listed = json.loads((root / CODE_MANIFEST).read_text())
    found = code_manifest(root)
    if found != listed:
        bad = sorted(k for k in set(found) | set(listed) if found.get(k) != listed.get(k))
        raise RuntimeError(f"code tree differs from {CODE_MANIFEST}: {bad[:10]}")
    digest = code_digest(listed)
    if expected_digest and digest != expected_digest:
        raise RuntimeError(f"attached code dataset is version {digest}, this notebook expects {expected_digest}: "
                           "upload the current build (scripts/launch_kaggle.py does this) and re-run")
    return digest


def load_yaml(name: str) -> dict:
    with open(paths.config_path(name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_immutable_parquet(df: pd.DataFrame, path: Path) -> str:
    """Write once. Re-writing identical content is a no-op ("unchanged"); different content raises."""
    path = Path(path)
    if path.exists():
        old = pd.read_parquet(path)
        if old.equals(df.reset_index(drop=True)):
            return "unchanged"
        raise ImmutableFileError(f"{path} exists with different content; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)
    return "written"


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write to a temporary file in the same directory, fsync, then rename over the target. A crash leaves
    either the old file or the new one, never a partial file (readers ignore leftover .tmp files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.reset_index(drop=True).to_parquet(tmp, index=False)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


def atomic_write_text(text: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
