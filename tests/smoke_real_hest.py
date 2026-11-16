#!/usr/bin/env python3
"""Real-data training smoke test; train/val may be same-slide and are NOT study results."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integrated_pipeline.audit import inspect_sample
from integrated_pipeline.train import train


def copy_patch(source: Path, fold: Path, name: str) -> None:
    (fold / "images").mkdir(parents=True, exist_ok=True)
    (fold / "labels").mkdir(exist_ok=True)
    images = sorted((source / "images").glob("*.png"))
    if len(images) != 1:
        raise ValueError("Smoke test expects exactly one patch per input directory")
    image = images[0]
    shutil.copy2(image, fold / "images" / f"{name}.png")
    shutil.copy2(source / "labels" / f"{image.stem}.npy", fold / "labels" / f"{name}.npy")
    with (source / "types.csv").open(newline="") as stream:
        tissue = next(csv.DictReader(stream))["type"]
    with (fold / "types.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["img", "type"])
        writer.writeheader(); writer.writerow({"img": f"{name}.png", "type": tissue})
    with (source / "cell_count.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    row["Image"] = f"{name}.png"
    with (fold / "cell_count.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-patch-dir", type=Path, required=True)
    parser.add_argument("--val-patch-dir", type=Path, required=True)
    parser.add_argument("--cdge-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    a, b = inspect_sample(args.train_patch_dir), inspect_sample(args.val_patch_dir)
    dataset = args.out_dir / "cellvit_dataset"
    dataset.mkdir(parents=True)
    source_config = args.cdge_root / "CellViT/configs/datasets/PanNuke/dataset_config.yaml"
    shutil.copy2(source_config, dataset / "dataset_config.yaml")
    copy_patch(args.train_patch_dir, dataset / "fold0", "TENX142_train_smoke")
    copy_patch(args.val_patch_dir, dataset / "fold1", "TENX142_val_smoke")
    (args.out_dir / "SMOKE_ONLY.txt").write_text(
        "Both folds may come from one HEST slide. This checks code execution only; "
        "it is not an independent validation split or a performance result.\n"
    )
    config = {"cdge_root": args.cdge_root, "cellvit": {"device": args.device,
               "batch_size": 1, "learning_rate": 3e-4}}
    result = train(config, dataset, args.out_dir, smoke=True)
    (args.out_dir / "real_data_smoke_summary.json").write_text(json.dumps({
        "input_audit": {"train": a, "val": b}, "training": result,
        "scope": "TENX142 teacher-label interface demo; not proposed-label result",
    }, indent=2) + "\n")
    print("PASSED", args.out_dir / "real_data_smoke_summary.json")


if __name__ == "__main__":
    main()
