#!/usr/bin/env python3
"""Check scientific environments and input contracts before the complete run."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["analysis", "cnv"], required=True)
    parser.add_argument("--adata", type=Path)
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--preparation-mode", choices=["recompute", "existing"], default="recompute")
    parser.add_argument("--organ")
    args = parser.parse_args()
    libraries = ["anndata", "scanpy", "numpy", "pandas", "scipy", "sklearn", "leidenalg", "igraph"]
    libraries += ["diptest", "pronto", "fastobo"] if args.kind == "analysis" else ["insitucnv", "infercnvpy"]
    versions = {}
    for library in libraries:
        importlib.import_module(library)
        distribution = "scikit-learn" if library == "sklearn" else library
        try:
            versions[library] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[library] = "importable; distribution version unavailable"
    report = {"python": sys.version, "executable": sys.executable, "kind": args.kind, "libraries": versions}
    if args.adata:
        import scanpy as sc
        import numpy as np
        from scipy import sparse
        code = Path(__file__).resolve().parents[1] / "methods/current/code"
        sys.path.insert(0, str(code))
        from reliability.data_contracts import validate_counts, usable_panel_coordinates
        from reliability.ontology_tree import CellOntologyMapper
        import pandas as pd
        data = sc.read_h5ad(args.adata, backed="r")
        try:
            if not data.obs_names.is_unique or not data.var_names.is_unique:
                raise ValueError("Cell IDs and gene IDs must be unique")
            if any(any(token in str(column).lower() for token in (
                "annotation", "ground_truth", "official_label", "manual_label", "cell_type", "truth"
            )) for column in data.obs.columns):
                raise ValueError("Use a label-free input; annotation-like obs column found")
            if "spatial" not in data.obsm or np.asarray(data.obsm["spatial"]).shape != (data.n_obs, 2):
                raise ValueError("Need finite n_cells by 2 spatial coordinates")
            if not np.isfinite(data.obsm["spatial"]).all():
                raise ValueError("Nonfinite spatial coordinates")
            layer = args.counts_layer
            if layer != "X" and layer not in data.layers:
                raise ValueError("Missing raw counts layer: " + layer)
            raw = data.X if layer == "X" else data.layers[layer]
            zero = 0
            for start in range(0, data.n_obs, 10000):
                block = raw[start:start + 10000]
                validate_counts(block)
                zero += int((np.asarray(block.sum(axis=1)).ravel() <= 0).sum())
            if data.n_vars < 31 or data.n_obs < 31:
                raise ValueError("This fixed method requires 30 PCA dimensions")
            if args.preparation_mode == "existing":
                if "counts" not in data.layers:
                    raise ValueError("Existing preparation requires raw counts in layers['counts']")
                if "X_pca" not in data.obsm or data.obsm["X_pca"].shape[1] < 30:
                    raise ValueError("Existing preparation requires at least 30 PCs")
                if "leiden_res_2.00" not in data.obs or "connectivities" not in data.obsp:
                    raise ValueError("Existing preparation requires expression clusters and graph")
                if zero:
                    raise ValueError("Existing preparation contains zero-count cells")
            resource = code.parent / "resources"
            coordinates = pd.read_csv(resource / "ensembl_hg38_coordinates.tsv", sep="\t").rename(columns={"gene": "gene_name"})
            covered = usable_panel_coordinates(coordinates, data.var_names.str.upper())
            if len(covered) < 30:
                raise ValueError("Fewer than the fixed minimum of 30 mapped panel genes")
            mapper = CellOntologyMapper.from_files(organ=args.organ)
            report["input"] = {"path": str(args.adata), "cells": data.n_obs, "genes": data.n_vars,
                               "counts_layer": layer, "zero_count_cells": zero,
                               "mapped_panel_genes": len(covered), "organ": args.organ,
                               "ontology_loaded": mapper is not None}
        finally:
            data.file.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
