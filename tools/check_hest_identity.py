#!/usr/bin/env python3
"""Reject spot-level h5ad before running cell-level molecular labeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import h5py


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--min-overlap", type=float, default=0.9)
    args = parser.parse_args()
    with h5py.File(args.adata) as handle:
        obs = handle["obs"]
        ids = {decode(x) for x in obs[decode(obs.attrs["_index"])][:]}
    segmentation = gpd.read_parquet(args.segmentation)
    seg_ids = set(segmentation.index.astype(str))
    overlap = len(ids & seg_ids)
    fraction = overlap / len(ids) if ids else 0
    report = {"h5ad_rows": len(ids), "nucleus_segmentation_rows": len(seg_ids),
              "matching_cell_ids": overlap, "fraction_of_h5ad_rows_matching": fraction,
              "minimum_required": args.min_overlap}
    print(json.dumps(report, indent=2))
    if fraction < args.min_overlap:
        raise ValueError("Input AnnData IDs do not match Xenium nucleus IDs; this may be spot-level ST data")


if __name__ == "__main__":
    main()
