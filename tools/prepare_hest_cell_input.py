#!/usr/bin/env python3
"""Build label-free per-cell AnnData from HEST cell counts and registered nuclei.

The HEST `st/` h5ad contains spot IDs. This tool reads `proc_adata` cell IDs
and raw counts, drops UNASSIGNED, and uses the matched H&E nucleus polygon
centroids as spatial positions. It copies NO precomputed cluster or class label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import openslide
from scipy import sparse


def decode(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_raw_counts(node, row_indices: np.ndarray, shape: tuple[int, int]):
    """Read a dense or AnnData CSR/CSC raw-count layer without changing counts."""
    if isinstance(node, h5py.Dataset):
        if node.shape != shape:
            raise ValueError("Counts matrix shape does not match cell/gene names")
        counts = node[row_indices, :]
    elif isinstance(node, h5py.Group):
        encoding = decode(node.attrs.get("encoding-type", ""))
        stored_shape = tuple(int(x) for x in node.attrs.get("shape", ()))
        if stored_shape != shape or encoding not in {"csr_matrix", "csc_matrix"}:
            raise ValueError(f"Unsupported raw-count encoding/shape: {encoding}, {stored_shape}")
        constructor = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
        matrix = constructor((node["data"][:], node["indices"][:], node["indptr"][:]), shape=shape)
        counts = matrix[row_indices, :].tocsr()
    else:
        raise ValueError("Unsupported raw-count object")
    values = counts.data if sparse.issparse(counts) else counts
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("Raw counts must be nonnegative integers")
    if values.size and np.max(values) > np.iinfo(np.int32).max:
        raise ValueError("Counts exceed int32; refusing lossy cast")
    return counts.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--wsi", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-match-fraction", type=float, default=0.9)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    seg = gpd.read_parquet(args.segmentation)
    if not seg.index.is_unique:
        raise ValueError("Duplicate nucleus cell IDs")
    seg.index = seg.index.astype(str)
    centroids = seg.geometry.centroid
    xy_by_id = pd.DataFrame({"x": centroids.x.to_numpy(), "y": centroids.y.to_numpy()}, index=seg.index)
    with h5py.File(args.source) as h5:
        obs = h5["obs"]
        ids = pd.Index([decode(x) for x in obs[decode(obs.attrs["_index"])][:]])
        genes = pd.Index([decode(x) for x in h5["var"][decode(h5["var"].attrs["_index"])][:]])
        if not ids.is_unique or not genes.is_unique:
            raise ValueError("Duplicate cell or gene IDs in source")
        eligible = ids.isin(xy_by_id.index)
        matched = int(eligible.sum())
        fraction = matched / len(ids)
        if fraction < args.min_match_fraction:
            raise ValueError(f"Only {matched}/{len(ids)} cell IDs match registered nuclei")
        if "counts" not in h5["layers"]:
            raise ValueError("Source lacks raw counts layer")
        indices = np.flatnonzero(eligible)
        # The 10x/HEST source can contain an UNASSIGNED aggregate row; exclude
        # it because there is no corresponding single nucleus.
        counts = selected_raw_counts(h5["layers/counts"], indices, (len(ids), len(genes)))
    selected = ids[eligible]
    spatial_px = xy_by_id.loc[selected, ["x", "y"]].to_numpy(dtype=np.float64)
    slide = openslide.OpenSlide(str(args.wsi))
    try:
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        width, height = slide.dimensions
    finally:
        slide.close()
    if not (0 < mpp_x < 10 and 0 < mpp_y < 10):
        raise ValueError("Missing or implausible H&E microns-per-pixel metadata")
    if (spatial_px[:, 0] < 0).any() or (spatial_px[:, 0] >= width).any() or (spatial_px[:, 1] < 0).any() or (spatial_px[:, 1] >= height).any():
        raise ValueError("Nucleus centroid lies outside the registered WSI")
    spatial = spatial_px * np.array([mpp_x, mpp_y])
    if not np.isfinite(spatial).all():
        raise ValueError("Nonfinite registered nucleus centroid")
    data = ad.AnnData(X=counts, obs=pd.DataFrame(index=selected), var=pd.DataFrame(index=genes))
    data.layers["counts"] = counts
    data.obsm["spatial"] = spatial
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(args.out)
    report = {"source": str(args.source.resolve()), "source_sha256": sha(args.source),
              "segmentation": str(args.segmentation.resolve()), "segmentation_sha256": sha(args.segmentation),
              "output": str(args.out.resolve()), "cells": int(data.n_obs), "genes": int(data.n_vars),
              "source_rows": len(ids), "id_match_fraction": fraction,
              "spatial_coordinates": "registered H&E nucleus polygon centroids converted from WSI pixels to microns",
              "wsi_mpp_x": mpp_x, "wsi_mpp_y": mpp_y,
              "annotation_columns_copied": [], "precomputed_clusters_copied": False}
    args.out.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
