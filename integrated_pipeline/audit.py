"""Validate the actual PanNukeDataset files consumed by CellViT."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

KNOWN = {1, 2, 3, 5}
UNKNOWN = 11


def inspect_pair(image: Path, label: Path, patch_size: int = 256) -> dict:
    with Image.open(image) as im:
        # The teacher writer stores opaque RGBA PNGs. Its PanNukeDataset
        # converts them to RGB when loading; mirror that exact behavior.
        if im.mode == "RGBA" and im.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"Transparent H&E pixels would change on RGB loading: {image}")
        if im.mode not in ("RGB", "RGBA"):
            raise ValueError(f"Unsupported H&E PNG mode {im.mode}: {image}")
        rgb = np.asarray(im.convert("RGB"))
    payload = np.load(label, allow_pickle=True)[()]
    if set(payload) != {"inst_map", "type_map"}:
        raise ValueError(f"Unexpected mask keys: {label}")
    inst, typ = payload["inst_map"], payload["type_map"]
    if rgb.shape != (patch_size, patch_size, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Invalid image dimensions or dtype: {image}")
    if inst.shape != rgb.shape[:2] or typ.shape != inst.shape:
        raise ValueError(f"Mask dimensions differ from image: {label}")
    if inst.dtype.kind not in "iu" or typ.dtype.kind not in "iu":
        raise ValueError(f"Masks must contain integer IDs: {label}")
    if (inst < 0).any() or (typ < 0).any():
        raise ValueError(f"Negative instance or type ID: {label}")
    if np.any((inst == 0) & (typ != 0)) or np.any((inst > 0) & (typ == 0)):
        raise ValueError(f"Instance/background mismatch: {label}")
    labels = set(np.unique(typ).tolist())
    if not labels <= ({0} | KNOWN | {UNKNOWN}):
        raise ValueError(f"Unexpected type IDs {labels}: {label}")
    instance_ids = np.unique(inst[inst > 0])
    for instance_id in instance_ids:
        values = set(np.unique(typ[inst == instance_id]).tolist())
        if len(values) != 1:
            raise ValueError(f"One nucleus has multiple type IDs: {label}, instance {instance_id}")
    return {
        "image": str(image), "label": str(label),
        "instances": int(len(instance_ids)),
        "known_pixels": int(np.isin(typ, list(KNOWN)).sum()),
        "unknown_pixels": int((typ == UNKNOWN).sum()),
        "type_ids": sorted(labels),
    }


def inspect_sample(sample_dir: Path, patch_size: int = 256) -> dict:
    images = sorted((sample_dir / "images").glob("*.png"))
    labels = sorted((sample_dir / "labels").glob("*.npy"))
    if not images or {p.stem for p in images} != {p.stem for p in labels}:
        raise ValueError(f"PNG and NPY do not match in {sample_dir}")
    reports = [inspect_pair(image, sample_dir / "labels" / (image.stem + ".npy"), patch_size) for image in images]
    counts = {"patches": len(reports), "instances": sum(r["instances"] for r in reports),
              "known_pixels": sum(r["known_pixels"] for r in reports),
              "unknown_pixels": sum(r["unknown_pixels"] for r in reports)}
    types_csv = sample_dir / "types.csv"
    cell_count_csv = sample_dir / "cell_count.csv"
    if not types_csv.exists() or not cell_count_csv.exists():
        raise ValueError(f"Missing CellViT metadata CSV in {sample_dir}")
    with types_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(images) or {r["img"] for r in rows} != {p.name for p in images}:
        raise ValueError("types.csv does not cover exactly the image files")
    if any(not r["type"] for r in rows):
        raise ValueError("types.csv contains an empty tissue type")
    with cell_count_csv.open(newline="") as stream:
        counts_rows = list(csv.DictReader(stream))
    if len(counts_rows) != len(images) or {r["Image"] for r in counts_rows} != {p.name for p in images}:
        raise ValueError("cell_count.csv does not cover exactly the image files")
    return counts


def write_report(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
