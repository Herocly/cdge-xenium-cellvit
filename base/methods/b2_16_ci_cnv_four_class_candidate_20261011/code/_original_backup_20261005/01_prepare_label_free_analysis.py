#!/usr/bin/env python3
"""Prepare a Xenium slide for label-free pseudo-label selection.

This script intentionally never reads annotation files.  It transforms the
gene-count matrix, computes a PCA expression representation, builds the
Leiden neighbourhood graph, and preserves spatial coordinates for later
evidence checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prepared-input",
        type=Path,
        help=(
            "Reuse an existing label-free AnnData and attach raw counts from --input "
            "after strict cell/gene alignment, without recomputing PCA or Leiden."
        ),
    )
    parser.add_argument(
        "--expression-layer",
        default=None,
        help="Use this AnnData layer as the expression matrix before normalization (for example: counts).",
    )
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--leiden-resolution", type=float, default=2.0)
    parser.add_argument("--min-counts", type=int, default=10)
    parser.add_argument("--min-genes", type=int, default=5)
    return parser.parse_args()


def scale_sparse_equivalent(
    matrix: sparse.spmatrix,
    *,
    max_value: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Represent Scanpy's centred scaling without materialising a dense matrix.

    The returned matrix stores ``scaled + mean / std``. PCA centres its input,
    and gene-wise rank tests are invariant to this column-wise offset, so both
    downstream operations are equivalent to using the dense centred matrix.
    """
    scaled = matrix.tocsr(copy=True)
    if not np.issubdtype(scaled.dtype, np.floating):
        scaled = scaled.astype(np.float64)

    mean, population_variance = mean_variance_axis(scaled, axis=0)
    variance = population_variance * scaled.shape[0] / (scaled.shape[0] - 1)
    std = np.sqrt(variance)
    std[std == 0] = 1
    shift = mean / std
    if np.any(shift > max_value):
        genes = np.flatnonzero(shift > max_value)
        raise RuntimeError(
            "Exact sparse scaling cannot represent the lower clipping bound for "
            f"{len(genes)} genes. Use a larger-memory host for dense scaling."
        )

    data = scaled.data
    columns = scaled.indices
    block_size = 1_000_000
    upper = shift + max_value
    for start in range(0, data.size, block_size):
        stop = min(start + block_size, data.size)
        block_columns = columns[start:stop]
        data[start:stop] /= std[block_columns]
        np.minimum(data[start:stop], upper[block_columns], out=data[start:stop])
    scaled.eliminate_zeros()
    return scaled, mean, std


def scale_for_pca(adata: sc.AnnData, *, max_value: float) -> str:
    """Scale expression while keeping sparse slides sparse when mathematically safe."""
    if not sparse.issparse(adata.X):
        sc.pp.scale(adata, max_value=max_value)
        return "dense_centered"

    scaled, mean, std = scale_sparse_equivalent(adata.X, max_value=max_value)
    adata.X = scaled
    adata.var["mean"] = mean
    adata.var["std"] = std
    adata.var["sparse_centering_shift"] = mean / std
    return "sparse_column_shift_equivalent"


def prepare_label_free_analysis(
    adata: sc.AnnData,
    *,
    expression_layer: str | None = None,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    leiden_resolution: float = 2.0,
    min_counts: int = 10,
    min_genes: int = 5,
) -> tuple[sc.AnnData, dict[str, object]]:
    """Prepare one slide while preserving unnormalised counts for CNV stages."""
    from reliability.data_contracts import validate_counts

    original_cells = adata.n_obs

    if expression_layer is not None:
        if expression_layer not in adata.layers:
            raise KeyError(f"Missing AnnData layer: {expression_layer!r}")
        raw_counts = adata.layers[expression_layer].copy()
    else:
        raw_counts = adata.X.copy()
    validate_counts(raw_counts)
    adata.X = raw_counts.copy()
    adata.layers["counts"] = raw_counts

    adata.var_names = adata.var_names.astype(str).str.upper()
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell IDs and uppercase gene symbols must be unique")
    adata.obs["total_counts_raw"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["n_genes_by_counts_raw"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    keep = (adata.obs["total_counts_raw"] >= min_counts) & (
        adata.obs["n_genes_by_counts_raw"] >= min_genes
    )
    adata = adata[keep].copy()

    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    scaling_representation = scale_for_pca(adata, max_value=10)
    n_comps = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack", zero_center=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)
    cluster_key = f"leiden_res_{leiden_resolution:.2f}"
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added=cluster_key)

    provenance: dict[str, object] = {
        "official_labels_policy": "not_loaded",
        "input_cells": original_cells,
        "retained_cells": adata.n_obs,
        "min_counts": min_counts,
        "min_genes": min_genes,
        "normalization": "normalize_total(target_sum=10000), log1p, scale(max_value=10)",
        "scaling_representation": scaling_representation,
        "scaling_equivalence": (
            "sparse X stores scaled + mean/std; PCA centering and gene-wise ranks "
            "are equivalent to dense centered scaling"
        ),
        "expression_source": expression_layer or "X",
        "raw_counts_layer": "counts",
        "representation": "X_pca",
        "n_pcs": n_comps,
        "n_neighbors": n_neighbors,
        "leiden_key": cluster_key,
        "leiden_resolution": leiden_resolution,
    }
    adata.uns["label_free_preprocessing"] = provenance
    return adata, provenance


def attach_raw_counts(
    prepared: sc.AnnData,
    raw: sc.AnnData,
    *,
    expression_layer: str | None = None,
    output_layer: str = "counts",
) -> tuple[sc.AnnData, dict[str, object]]:
    """Attach raw counts to an existing analysis after explicit ID alignment."""
    if not prepared.obs_names.is_unique or not raw.obs_names.is_unique:
        raise ValueError("Both AnnData objects must have unique cell IDs.")
    prepared.var_names = prepared.var_names.astype(str).str.upper()
    raw.var_names = raw.var_names.astype(str).str.upper()
    if not prepared.var_names.is_unique or not raw.var_names.is_unique:
        raise ValueError("Gene names must remain unique after upper-casing.")
    missing_cells = prepared.obs_names.difference(raw.obs_names)
    missing_genes = prepared.var_names.difference(raw.var_names)
    if len(missing_cells) or len(missing_genes):
        raise ValueError(
            f"Raw input misses {len(missing_cells)} prepared cells and "
            f"{len(missing_genes)} prepared genes."
        )
    aligned = raw[prepared.obs_names, prepared.var_names]
    if expression_layer is not None:
        if expression_layer not in aligned.layers:
            raise KeyError(f"Missing AnnData layer: {expression_layer!r}")
        counts = aligned.layers[expression_layer].copy()
    else:
        counts = aligned.X.copy()
    from reliability.data_contracts import validate_counts

    validate_counts(counts)
    prepared.layers[output_layer] = counts
    provenance: dict[str, object] = {
        "official_labels_policy": "not_loaded",
        "alignment": "cell_id_and_uppercase_gene_name",
        "source_expression": expression_layer or "X",
        "output_layer": output_layer,
        "cell_count": prepared.n_obs,
        "gene_count": prepared.n_vars,
    }
    prepared.uns["raw_counts_provenance"] = provenance
    return prepared, provenance


def main() -> None:
    args = parse_args()
    if args.prepared_input is not None:
        adata, counts_provenance = attach_raw_counts(
            sc.read_h5ad(args.prepared_input),
            sc.read_h5ad(args.input),
            expression_layer=args.expression_layer,
        )
        provenance = dict(adata.uns.get("label_free_preprocessing", {}))
        print(json.dumps(counts_provenance, indent=2))
    else:
        adata, provenance = prepare_label_free_analysis(
            sc.read_h5ad(args.input),
            expression_layer=args.expression_layer,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            leiden_resolution=args.leiden_resolution,
            min_counts=args.min_counts,
            min_genes=args.min_genes,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out, compression="gzip")
    print(json.dumps(provenance, indent=2))
    print(
        f"wrote {args.out}: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{adata.obs[str(provenance['leiden_key'])].nunique()} Leiden clusters"
    )


if __name__ == "__main__":
    main()
