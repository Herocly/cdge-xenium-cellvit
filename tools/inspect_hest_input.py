#!/usr/bin/env python3
"""Read-only check of the supported HEST image, nucleus, and cell-ID contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import openslide


def inspect(source: Path | None, segmentation: Path, coordinates: Path,
            wsi: Path, metadata: Path, minimum_match: float) -> dict:
    for path in (segmentation, coordinates, wsi, metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = json.loads(metadata.read_text())
    if not isinstance(meta, dict) or not (meta.get("tissue") or meta.get("organ")):
        raise ValueError("Metadata JSON needs tissue or organ")
    nuclei = gpd.read_parquet(segmentation)
    if nuclei.empty or not nuclei.index.is_unique or nuclei.geometry.isna().any():
        raise ValueError("Nucleus polygons are empty, duplicated, or missing")
    invalid_nuclei = int((~nuclei.geometry.is_valid).sum())
    invalid_fraction = invalid_nuclei / len(nuclei)
    # Real HEST segmentations can contain a small number of self-intersecting
    # polygons. Report them; the original raster writer handles the geometry.
    # Stop only when the source is predominantly malformed.
    if invalid_fraction > 0.01:
        raise ValueError(f"Too many invalid nucleus polygons: {invalid_nuclei}/{len(nuclei)}")
    with h5py.File(coordinates) as handle:
        if "coords" not in handle:
            raise ValueError("Patch coordinate HDF5 lacks /coords")
        coords = np.asarray(handle["coords"][:])
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0 or not np.isfinite(coords).all():
        raise ValueError("Patch coordinates must be a nonempty finite Nx2 array")
    slide = openslide.OpenSlide(str(wsi))
    try:
        width, height = slide.dimensions
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
    finally:
        slide.close()
    if not (0 < mpp_x < 10 and 0 < mpp_y < 10):
        raise ValueError("WSI has missing or implausible microns-per-pixel")
    if np.any(coords < 0) or np.any(coords[:, 0] >= width) or np.any(coords[:, 1] >= height):
        raise ValueError("Patch coordinates fall outside the WSI")
    centroids = nuclei.geometry.centroid
    if (centroids.x < 0).any() or (centroids.y < 0).any() or (centroids.x >= width).any() or (centroids.y >= height).any():
        raise ValueError("Nucleus centroids fall outside the WSI")
    result = {"nuclei": int(len(nuclei)), "patches": int(len(coords)),
              "invalid_nuclei": invalid_nuclei, "invalid_fraction": invalid_fraction,
              "wsi_size": [int(width), int(height)], "mpp": [mpp_x, mpp_y],
              "tissue": meta.get("tissue", meta.get("organ"))}
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(source)
        with h5py.File(source) as handle:
            if "obs" not in handle or "var" not in handle:
                raise ValueError("Source h5ad needs /obs and /var")
            name = handle["obs"].attrs["_index"]
            name = name.decode() if isinstance(name, bytes) else name
            ids = handle["obs"][name][:]
            if "layers" not in handle or "counts" not in handle["layers"]:
                raise ValueError("Source h5ad needs raw /layers/counts")
        as_text = lambda x: x.decode() if isinstance(x, bytes) else str(x)
        ids = {as_text(x) for x in ids}
        if len(ids) == 0:
            raise ValueError("Source h5ad has no cell IDs")
        matched = len(ids & set(nuclei.index.map(str)))
        result["cell_id_match"] = {"matched": matched, "source_cells": len(ids),
                                   "fraction": matched / len(ids)}
        if matched / len(ids) < minimum_match:
            raise ValueError(f"Too few source cell IDs match nuclei: {matched}/{len(ids)}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    for flag in ("segmentation", "coordinates", "wsi", "metadata"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--minimum-match", type=float, default=0.9)
    args = p.parse_args()
    print(json.dumps(inspect(args.source, args.segmentation, args.coordinates,
                             args.wsi, args.metadata, args.minimum_match), indent=2))


if __name__ == "__main__":
    main()
