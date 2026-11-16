#!/usr/bin/env python3
"""Verify the frozen source files shipped in this handoff.

This check says nothing about scientific correctness or full-run reproduction.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "BUNDLE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[Path]:
    paths = list((ROOT / "methods").rglob("*"))
    paths += [ROOT / "run_all.py", ROOT / "run_stage.py", ROOT / "tools/preflight.py"]
    return sorted(path for path in paths if path.is_file()
                  and "__pycache__" not in path.parts
                  and path.suffix != ".pyc" and path.name != ".DS_Store")


def build() -> None:
    entries = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                "bytes": path.stat().st_size} for path in source_files()]
    MANIFEST.write_text(json.dumps({"purpose": "Frozen method-source integrity",
                                    "files": entries}, indent=2) + "\n")
    print(f"Recorded {len(entries)} source files")


def verify() -> None:
    payload = json.loads(MANIFEST.read_text())
    expected = {item["path"]: item for item in payload["files"]}
    actual = {str(path.relative_to(ROOT)) for path in source_files()}
    if actual != set(expected):
        raise ValueError(f"Source inventory differs: {actual ^ set(expected)}")
    for relative, item in expected.items():
        path = ROOT / relative
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Source changed: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Verified {len(expected)} source hashes and Python syntax")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    verify()
