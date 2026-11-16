#!/usr/bin/env python3
"""Attach one sample's cell-label CSV to HEST nucleus polygons and make CellViT patches.

This adapter reuses the CDGE polygon/patch writer, with one compatibility fix:
instance IDs start at 1 (CellViT reserves 0 for background). Unknown remains
11 and needs an explicit ignore policy in the training code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd


# IDs from CDGE preprocessing_hest/preproce_utils/common.py.
CLASS_TO_ID = {
    "Neoplastic": 1,
    "Inflammatory": 2,
    "Connective_tissue": 3,
    "Epithelial": 5,
    "Unknown": 11,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_id(labels_csv: Path, digest: str, run_name: str | None, patch_index: int | None) -> str:
    """Give each input file version its own deterministic output directory."""
    name = run_name or labels_csv.stem
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")[:48]
    if not name:
        raise ValueError("Run name/CSV stem must contain letters or digits")
    scope = "all" if patch_index is None else f"patch{patch_index:05d}"
    return f"{name}-{digest[:12]}-{scope}"


def read_labels(path: Path, sample_id: str) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"sample_id": str, "cell_id": str})
    required = {"sample_id", "cell_id", "predicted_class"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(labels.columns))}")
    if labels.empty or labels["sample_id"].isna().any() or set(labels["sample_id"]) != {sample_id}:
        raise ValueError("CSV must contain only the requested sample_id")
    if labels["cell_id"].isna().any() or labels["cell_id"].eq("").any():
        raise ValueError("CSV contains an empty cell_id")
    if labels["cell_id"].duplicated().any():
        raise ValueError("CSV contains duplicate cell_id values")
    if labels["predicted_class"].isna().any():
        raise ValueError("CSV contains a missing predicted_class")
    unknown_names = set(labels["predicted_class"]) - set(CLASS_TO_ID)
    if unknown_names:
        raise ValueError(f"Unexpected predicted_class values: {sorted(unknown_names)}")
    return labels


def assign_labels(seg, labels: pd.DataFrame):
    """Return a copy of a GeoDataFrame with CDGE's cell_type_id column.

    The segmentation index and CSV cell_id both name the same Xenium cell.
    Segmentation entries without a CSV label remain Unlabeled/Unknown (11).
    """
    seg = seg.copy()
    seg.index = pd.Index(seg.index.map(str), name="cell_id")
    if not seg.index.is_unique:
        raise ValueError("Segmentation contains duplicate cell IDs")
    lookup = labels.set_index("cell_id")["predicted_class"].map(CLASS_TO_ID)
    seg["cell_type_id"] = lookup.reindex(seg.index).fillna(11).astype("int32")
    stats = {
        "csv_cells": int(len(labels)),
        "segmentation_nuclei": int(len(seg)),
        "matched_ids": int(seg.index.isin(lookup.index).sum()),
        "csv_ids_without_nucleus": int((~lookup.index.isin(seg.index)).sum()),
        "nuclei_without_csv_label": int((~seg.index.isin(lookup.index)).sum()),
        "nuclei_by_class_id": {str(k): int(v) for k, v in seg["cell_type_id"].value_counts().sort_index().items()},
    }
    return seg, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hest-dir", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, help="Override HEST nucleus polygon parquet")
    parser.add_argument("--coordinates", type=Path, help="Override H&E patch coordinate HDF5")
    parser.add_argument("--wsi", type=Path, help="Override registered H&E WSI")
    parser.add_argument("--metadata", type=Path, help="Override HEST sample metadata JSON")
    parser.add_argument("--cdge-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", help="Readable label-source name, e.g. bottom_up or ours")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-index", type=int, help="Write only this zero-based HEST patch for a smoke test")
    parser.add_argument(
        "--min-csv-match-fraction", type=float, default=0.9,
        help="Stop if too few CSV cell IDs occur in the selected HEST nucleus segmentation",
    )
    args = parser.parse_args()
    if args.patch_size <= 0:
        raise ValueError("patch-size must be positive")
    if not 0 <= args.min_csv_match_fraction <= 1:
        raise ValueError("min-csv-match-fraction must be between 0 and 1")
    labels = read_labels(args.labels_csv, args.sample_id)
    labels_sha = file_sha256(args.labels_csv)
    run_id = make_run_id(args.labels_csv, labels_sha, args.run_name, args.patch_index)
    sample_out = args.out_dir / run_id / args.sample_id
    if sample_out.exists():
        raise FileExistsError(f"Output already exists for this CSV and patch scope: {sample_out}")

    # Load the original CDGE writer directly, without importing its full
    # preprocessing package (which would also pull in Scanpy and AnnData).
    common_path = args.cdge_root / "preprocessing_hest" / "preproce_utils" / "common.py"
    if not common_path.is_file():
        raise FileNotFoundError(common_path)
    spec = importlib.util.spec_from_file_location("cdge_patch_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(common_path)
    cdge_common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cdge_common)
    original_build_shapes = cdge_common._build_shapes

    def build_shapes_with_positive_instance_ids(gdf_roi, x0, y0):
        """Preserve CDGE polygons and classes, reserving instance ID 0 for background."""
        semantic, instances, counts = original_build_shapes(gdf_roi, x0, y0)
        return semantic, [(geom, instance_id + 1) for geom, instance_id in instances], counts

    cdge_common._build_shapes = build_shapes_with_positive_instance_ids
    import geopandas as gpd
    import h5py
    import openslide
    generate_paired_sample = cdge_common.generate_paired_sample
    process_patch = cdge_common.process_patch

    seg_path = args.segmentation or args.hest_dir / "xenium_seg" / f"{args.sample_id}_xenium_nucleus_seg.parquet"
    coords_path = args.coordinates or args.hest_dir / "coordinates" / "patches" / f"{args.sample_id}_patches.h5"
    wsi_path = args.wsi or args.hest_dir / "wsis" / f"{args.sample_id}.tif"
    metadata_path = args.metadata or args.hest_dir / "metadata" / f"{args.sample_id}.json"
    for path in (seg_path, coords_path, wsi_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    seg, stats = assign_labels(gpd.read_parquet(seg_path), labels)
    if stats["matched_ids"] / stats["csv_cells"] < args.min_csv_match_fraction:
        raise ValueError(
            "Too few CSV cell IDs match this HEST segmentation: "
            f"{stats['matched_ids']}/{stats['csv_cells']}"
        )
    seg["center_x"] = seg.geometry.centroid.x
    seg["center_y"] = seg.geometry.centroid.y
    sindex = seg.sindex
    with h5py.File(coords_path, "r") as handle:
        coords = handle["coords"][:]
    if args.patch_index is not None and not 0 <= args.patch_index < len(coords):
        raise ValueError(f"patch-index must be in [0, {len(coords) - 1}]")
    metadata = json.loads(metadata_path.read_text())
    organ_raw = metadata.get("tissue", metadata.get("organ", "Unknown"))
    organ = cdge_common.NORM_TISSUE_NAMES.get(organ_raw, organ_raw)
    (sample_out / "images").mkdir(parents=True, exist_ok=True)
    (sample_out / "labels").mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        if args.patch_index is None:
            generate_paired_sample(
                None, seg, sindex, slide, coords, sample_out, organ,
                patch_size=args.patch_size, vis_dir=None, enable_vis=False,
            )
            stats["patches_requested"] = int(len(coords))
        else:
            idx = args.patch_index
            x0, y0 = coords[idx]
            process_patch(
                idx, x0, y0, args.patch_size, seg, sindex, slide,
                sample_out, organ, {}, {}, {}, vis_dir=None, enable_vis=False,
            )
            stats["patches_requested"] = 1
            stats["patch_index"] = idx
    finally:
        slide.close()

    if file_sha256(args.labels_csv) != labels_sha:
        raise RuntimeError("Input CSV changed while masks were being generated")
    stats.update({
        "run_id": run_id,
        "labels_csv": str(args.labels_csv.resolve()),
        "labels_csv_sha256": labels_sha,
        "cdge_patch_writer_sha256": file_sha256(common_path),
        "image_files": len(list((sample_out / "images").glob("*.png"))),
        "label_files": len(list((sample_out / "labels").glob("*.npy"))),
        "unknown_class_id": 11,
        "instance_id_policy": "positive IDs; 0 is background",
        "original_cellvit_trainer_compatible": False,
        "required_training_adapter": "integrated_pipeline.train masks class ID 11 in the type loss",
    })
    audit_path = sample_out / "csv_to_cellvit_audit.json"
    audit_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
