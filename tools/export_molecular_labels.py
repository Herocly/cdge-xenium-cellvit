#!/usr/bin/env python3
"""Export molecular cell labels and source coordinates for later image matching.

The output is not a CellViT instance mask or verified image registration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CLASSES = {"Connective_tissue", "Inflammatory", "Epithelial", "Neoplastic", "Unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    predictions = pd.read_csv(args.predictions, dtype={"cell_id": str})
    if predictions.cell_id.duplicated().any() or predictions.cell_id.isna().any():
        raise ValueError("Predictions need unique nonempty cell_id values")
    if not set(predictions.predicted_class.astype(str)) <= CLASSES:
        raise ValueError("Unexpected prediction class")
    data = ad.read_h5ad(args.adata, backed="r")
    try:
        ids = pd.Index(data.obs_names.astype(str))
        if not ids.is_unique or "spatial" not in data.obsm:
            raise ValueError("AnnData needs unique cell IDs and spatial coordinates")
        xy = np.asarray(data.obsm["spatial"])
        if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
            raise ValueError("Expected finite two-dimensional spatial coordinates")
        positions = pd.DataFrame({"cell_id": ids, "x_source": xy[:, 0], "y_source": xy[:, 1]})
    finally:
        data.file.close()
    result = positions.merge(predictions, on="cell_id", validate="one_to_one", how="left", indicator=True)
    if not result["_merge"].eq("both").all() or len(result) != len(predictions):
        raise ValueError("Prediction cell IDs must match the input AnnData exactly")
    result.insert(0, "sample_id", args.sample_id)
    result["is_unknown"] = result.predicted_class.eq("Unknown")
    columns = ["sample_id", "cell_id", "x_source", "y_source", "predicted_class", "is_unknown"]
    for extra in ("confidence", "rejected_by_preprocessing"):
        if extra in result:
            columns.append(extra)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result[columns].to_csv(args.out, index=False)
    print(f"Exported {len(result)} molecular labels; Unknown: {int(result.is_unknown.sum())}")


if __name__ == "__main__":
    main()
