#!/usr/bin/env python3
"""Run molecular labels -> HEST patches -> CellViT dataset -> train/evaluate."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .config import load, run_name, jsonable
from .stages import molecular, convert, validate, assemble
from .audit import inspect_sample


def check_assembled(dataset: Path, patch_size: int) -> None:
    """Reject an interrupted or stale assembly before any CellViT training."""
    for required in ("dataset_config.yaml", "weight_config.yaml", "split_manifest.json"):
        if not (dataset / required).is_file():
            raise ValueError(f"Incomplete CellViT dataset: missing {required} in {dataset}")
    for fold in (0, 1):
        inspect_sample(dataset / f"fold{fold}", patch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "preflight", "molecular", "convert", "check", "assemble", "train", "evaluate", "smoke"), default="all")
    args = parser.parse_args()
    config = load(args.config)
    if args.stage == "all" and not config.get("cellvit", {}).get("external_test_dataset"):
        raise ValueError("--stage all requires cellvit.external_test_dataset; run individual stages first")
    if args.stage in ("preflight", "all"):
        preflight = [sys.executable, str(Path(__file__).resolve().parent / "check_environment.py"),
                     "--config", str(args.config), "--deep", "--strict"]
        if args.stage == "all":
            preflight.append("--require-train-val")
        subprocess.run(preflight, check=True)
        if args.stage == "preflight":
            return
    run_dir = config["output_root"] / run_name(args.config, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = run_dir / "resolved_config.json"
    resolved = jsonable(config)
    if config_copy.exists() and json.loads(config_copy.read_text()) != resolved:
        raise ValueError("Existing run has different resolved configuration")
    config_copy.write_text(json.dumps(resolved, indent=2) + "\n")
    shutil.copy2(args.config, run_dir / "source_config.yaml")

    labels, patch_dirs = {}, {}
    if args.stage in ("all", "molecular", "convert", "check", "assemble"):
        for sample in config["samples"]:
            labels[sample["sample_id"]] = molecular(config, sample, run_dir)
        if args.stage == "molecular":
            print(json.dumps({k: str(v) for k, v in labels.items()}, indent=2)); return
        for sample in config["samples"]:
            sid = sample["sample_id"]
            patch_dirs[sid] = convert(config, sample, labels[sid], run_dir)
        if args.stage == "convert":
            print(json.dumps({k: str(v) for k, v in patch_dirs.items()}, indent=2)); return
        summary = validate(config, patch_dirs, run_dir)
        print("Patch audit:", json.dumps(summary, indent=2))
        if args.stage == "check":
            return
        dataset = run_dir / "cellvit_dataset"
        if not dataset.exists():
            dataset = assemble(config, patch_dirs, run_dir)
        check_assembled(dataset, config.get("patch_size", 256))
        if args.stage == "assemble":
            print(dataset); return
    else:
        dataset = run_dir / "cellvit_dataset"
        if not dataset.exists():
            raise FileNotFoundError("Run assemble stage first: " + str(dataset))
        check_assembled(dataset, config.get("patch_size", 256))

    if args.stage in ("all", "smoke"):
        subprocess.run([str(config["cellvit_python"]), str(Path(__file__).resolve().parent / "train.py"),
                        "--resolved-config", str(config_copy), "--run-dir", str(run_dir),
                        "--stage", "smoke"], check=True)
    if args.stage in ("all", "train"):
        subprocess.run([str(config["cellvit_python"]), str(Path(__file__).resolve().parent / "train.py"),
                        "--resolved-config", str(config_copy), "--run-dir", str(run_dir),
                        "--stage", "train"], check=True)
    if args.stage in ("all", "evaluate"):
        subprocess.run([str(config["cellvit_python"]), str(Path(__file__).resolve().parent / "train.py"),
                        "--resolved-config", str(config_copy), "--run-dir", str(run_dir),
                        "--stage", "evaluate"], check=True)


if __name__ == "__main__":
    main()
