"""Build label-free Neoplastic evidence and A/B/C/D candidate variants.

CNV is deliberately one feature among several.  No official annotations are
loaded here; the output consists of evidence columns and fixed candidate rules
that a separate evaluation script may assess afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.cluster_evidence_gate import strong_cluster_label_map  # noqa: E402
from reliability.data_contracts import validate_counts, positive_quantile_evidence  # noqa: E402


def to_dense(matrix: object) -> np.ndarray:
    """Convert a sparse AnnData matrix only when the 20k local sample is read."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)


def mean_abs_cnv(profile: np.ndarray) -> np.ndarray:
    """One CNV-deviation value per cell, relative to normal references."""
    return np.mean(np.abs(profile), axis=1)


def marker_gene_score(adata: sc.AnnData, genes: set[str]) -> tuple[np.ndarray, list[str]]:
    """Use full-panel, unsmoothed raw counts, never CNV's coordinate subset."""
    overlap = sorted(genes & set(adata.var_names.astype(str)))
    if not overlap:
        raise RuntimeError("No neoplastic marker genes overlap the Xenium panel.")
    if "counts" not in adata.layers:
        raise KeyError("Gene evidence requires an explicit raw counts layer")
    validate_counts(adata.layers["counts"])
    totals = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Gene evidence cannot normalize zero-count cells")
    matrix = sparse.csr_matrix(adata[:, overlap].layers["counts"], dtype=float)
    matrix = matrix.multiply((10_000 / totals)[:, None]).tocsr()
    np.log1p(matrix.data, out=matrix.data)
    return np.asarray(matrix.mean(axis=1)).ravel(), overlap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnv-adata", type=Path, required=True)
    parser.add_argument("--expression-adata", type=Path, required=True)
    parser.add_argument(
        "--cluster-table",
        type=Path,
        default=ROOT / "results/01_cluster_reliability/cluster_reliability.csv",
    )
    parser.add_argument(
        "--cnv-audit",
        type=Path,
        default=ROOT / "results/02_insitucnv/cnv_candidate_audit_label_free.csv",
    )
    parser.add_argument(
        "--neoplastic-genes",
        type=Path,
        default=ROOT / "resources/marker_databases/neoplastic_genes.csv",
    )
    parser.add_argument("--expression-cluster-column", default="leiden_res_2.00")
    parser.add_argument("--cnv-cluster-column", default="cnv_leiden_res0.2")
    parser.add_argument("--gene-evidence-quantile", type=float, default=0.90)
    parser.add_argument("--cnv-reference-quantile", type=float, default=0.99)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/07_neoplastic_evidence")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.cnv_adata)
    expression = sc.read_h5ad(args.expression_adata)
    if not adata.obs_names.is_unique or not expression.obs_names.is_unique:
        raise ValueError("Expression and CNV cell IDs must be unique")
    if set(adata.obs_names) != set(expression.obs_names):
        raise ValueError("Expression and CNV evidence use different cell axes")
    expression = expression[adata.obs_names].copy()
    expression_clusters = adata.obs[args.expression_cluster_column].astype(str).to_numpy()
    if not np.array_equal(expression.obs[args.expression_cluster_column].astype(str), expression_clusters):
        raise ValueError("Expression and CNV expression-cluster assignments differ")
    cnv_clusters = adata.obs[args.cnv_cluster_column].astype(str).to_numpy()
    cluster_table = pd.read_csv(args.cluster_table, dtype={"cluster": str})
    cnv_audit = pd.read_csv(args.cnv_audit, dtype={"cnv_cluster": str})
    if not cluster_table["cluster"].is_unique or not cnv_audit["cnv_cluster"].is_unique:
        raise ValueError("Cluster audit IDs must be unique")
    if not set(expression_clusters).issubset(set(cluster_table["cluster"])) or not set(cnv_clusters).issubset(set(cnv_audit["cnv_cluster"])):
        raise ValueError("Cluster evidence is missing clusters from the analysis")

    strong_normal_labels = strong_cluster_label_map(cluster_table)
    normal_evidence = np.asarray(
        [strong_normal_labels.get(cluster, "") for cluster in expression_clusters], dtype=object
    )
    has_strong_normal_evidence = np.isin(normal_evidence, ["Connective_tissue", "Inflammatory"])
    structural_score = dict(
        zip(cluster_table["cluster"].astype(str), cluster_table["structural_reliability_score"])
    )
    cell_structural_score = np.asarray(
        [structural_score.get(cluster, np.nan) for cluster in expression_clusters], dtype=float
    )

    cnv_profile = to_dense(adata.obsm["X_cnv"])
    cnv_amplitude = mean_abs_cnv(cnv_profile)
    reference = adata.obs["cnv_reference_category"].astype(str).to_numpy() != "Observation"
    if not reference.any() or not np.isfinite(cnv_amplitude).all():
        raise ValueError("CNV evidence requires finite profiles and normal references")
    cnv_cutoff = float(np.quantile(cnv_amplitude[reference], args.cnv_reference_quantile))
    cnv_abnormal = cnv_amplitude > cnv_cutoff
    audit_by_cluster = cnv_audit.set_index("cnv_cluster")
    cell_cnv_coherence = np.asarray(
        [audit_by_cluster.loc[cluster, "cnv_profile_coherence"] for cluster in cnv_clusters], dtype=float
    )
    cell_spatial_consistency = np.asarray(
        [audit_by_cluster.loc[cluster, "spatial_neighbor_consistency"] for cluster in cnv_clusters], dtype=float
    )
    if not all(np.isfinite(values).all() for values in (cell_cnv_coherence, cell_spatial_consistency, cell_structural_score)):
        raise ValueError("CNV, spatial and structural support must be finite")

    genes = set(pd.read_csv(args.neoplastic_genes)["gene"].astype(str))
    tumor_score, gene_overlap = marker_gene_score(expression, genes)
    tumor_evidence, tumor_cutoff = positive_quantile_evidence(tumor_score, args.gene_evidence_quantile)

    # A/B/C/D are fixed before official-label evaluation.
    # A: tumor-gene evidence only, excluding a strong normal explanation.
    variant_a = tumor_evidence & ~has_strong_normal_evidence & ~reference
    # B: CNV is a self-contained diagnostic only, not a final label.
    variant_b = cnv_abnormal & (cell_cnv_coherence >= 0.70)
    # C: distinct gene and CNV computations agree; statistical independence is not assumed.
    variant_c = variant_a & variant_b
    # D: require coherent spatial organization and a reasonably stable
    # expression cluster as additional safeguards.
    variant_d = (
        variant_c
        & (cell_spatial_consistency >= 0.50)
        & (cell_structural_score >= 0.50)
    )

    table = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "expression_cluster": expression_clusters,
            "cnv_cluster": cnv_clusters,
            "strong_normal_evidence_label": normal_evidence,
            "has_strong_normal_evidence": has_strong_normal_evidence,
            "epithelial_lineage_evidence": normal_evidence == "Epithelial",
            "is_cnv_reference": reference,
            "expression_cluster_structural_score": cell_structural_score,
            "cnv_amplitude": cnv_amplitude,
            "cnv_abnormal_relative_to_normal_reference": cnv_abnormal,
            "cnv_cluster_profile_coherence": cell_cnv_coherence,
            "cnv_cluster_spatial_consistency": cell_spatial_consistency,
            "tumor_gene_score": tumor_score,
            "tumor_gene_evidence": tumor_evidence,
            "variant_A_gene_only": variant_a,
            "variant_B_cnv_only": variant_b,
            "variant_C_gene_plus_cnv": variant_c,
            "variant_D_all_evidence": variant_d,
        }
    )
    table.to_csv(args.out_dir / "neoplastic_evidence_table_no_official_labels.csv", index=False)
    summary = pd.DataFrame(
        {
            "variant": ["A_gene_only", "B_cnv_only", "C_gene_plus_cnv", "D_all_evidence"],
            "candidate_cells": [int(variant_a.sum()), int(variant_b.sum()), int(variant_c.sum()), int(variant_d.sum())],
        }
    )
    summary.to_csv(args.out_dir / "ablation_candidate_counts_no_official_labels.csv", index=False)
    (args.out_dir / "thresholds_no_official_labels.json").write_text(
        pd.Series(
            {
                "official_labels_policy": "not_loaded",
                "tumor_gene_overlap_count": len(gene_overlap),
                "tumor_gene_overlap": gene_overlap,
                "gene_expression_source": str(args.expression_adata),
                "gene_expression_policy": "full_panel_raw_counts_normalize10000_log1p_no_neighbor_smoothing",
                "normal_exclusion_policy": "reference_cells_and_strong_connective_inflammatory_not_epithelial_lineage",
                "tumor_gene_evidence_quantile": args.gene_evidence_quantile,
                "tumor_gene_score_cutoff": tumor_cutoff,
                "cnv_reference_quantile": args.cnv_reference_quantile,
                "cnv_amplitude_cutoff": cnv_cutoff,
                "cnv_coherence_minimum": 0.70,
                "spatial_consistency_minimum": 0.50,
                "structural_score_minimum": 0.50,
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
