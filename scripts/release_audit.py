#!/usr/bin/env python3
"""Fail closed on obvious private/large assets in a proposed Git release."""
from __future__ import annotations

import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".h5ad", ".h5", ".hdf5", ".parquet", ".npy", ".npz",
                      ".tif", ".tiff", ".svs", ".ndpi", ".pth", ".pt", ".ckpt",
                      ".safetensors", ".pem", ".key"}
FORBIDDEN_PARTS = {"data", "outputs", "runs", "runs_integrated", "results",
                   "checkpoints", "__pycache__", ".venv"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-only", action="store_true", help="Audit assets without approving publication")
    args = parser.parse_args()
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, check=True, capture_output=True)
    files = [Path(x.decode()) for x in result.stdout.split(b"\0") if x]
    problems = []
    for relative in files:
        file = ROOT / relative
        if not file.is_file():
            continue
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES or FORBIDDEN_PARTS & set(relative.parts):
            problems.append(f"Private or generated asset: {relative}")
        if file.stat().st_size > 50 * 1024 * 1024:
            problems.append(f"File exceeds 50 MiB: {relative}")
        if "patient" in relative.name.lower() or "credential" in relative.name.lower():
            problems.append(f"Review sensitive filename: {relative}")
    report = {"files_checked": len(files), "obvious_asset_problems": problems,
              "license_review": "BLOCKED: see THIRD_PARTY.md and RELEASE_BLOCKERS.md",
              "public_push_approved": False}
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)
    if not args.files_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
