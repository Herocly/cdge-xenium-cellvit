"""Rank InSituCNV clusters for manual Neoplastic-seed review.

This is deliberately an audit, not an automatic tumour caller.  The authors'
InSituCNV workflow produces CNV clusters and then expects a human to review
the CNV heatmap and spatial distribution.  Official labels are never loaded.

【pipeline 位置】第 6 步:输入 05 的 adata_cnv.h5ad,输出每个 CNV cluster 一行的
审计表 cnv_candidate_audit_label_free.csv。它回答"哪些 CNV cluster 值得怀疑是肿瘤",
但本身不产生任何种子;逐细胞的 Neoplastic 种子由 07 结合 marker/结构/空间证据生成。
【cluster/cell 边界】本脚本全部是 CNV-cluster 级判断(一个 CNV cluster 一行)。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def to_dense(matrix: object) -> np.ndarray:
    """Convert InSituCNV's sparse or dense embedding to a NumPy array."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)


def mean_profile_coherence(profiles: np.ndarray) -> float:
    """Measure whether cells in one CNV cluster share one CNV pattern."""
    centre = profiles.mean(axis=0)
    denominator = np.linalg.norm(profiles, axis=1) * max(np.linalg.norm(centre), 1e-12)
    return float(np.mean((profiles @ centre) / np.maximum(denominator, 1e-12)))


def percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Make metrics with different scales comparable without fitting a label."""
    score = values.rank(method="average", pct=True)
    return score if higher_is_better else 1.0 - score


def spatial_neighbor_consistency(
    coordinates: np.ndarray, cnv_clusters: np.ndarray, n_neighbors: int = 20
) -> np.ndarray:
    """Return the fraction of spatial neighbours in the same CNV cluster."""
    tree = cKDTree(coordinates)
    _, neighbours = tree.query(coordinates, k=min(n_neighbors + 1, len(coordinates)))
    return np.mean(cnv_clusters[neighbours[:, 1:]] == cnv_clusters[:, None], axis=1)


def build_cnv_audit(
    cnv_profiles: np.ndarray,
    cnv_clusters: np.ndarray,
    reference_categories: np.ndarray,
    spatial_consistency: np.ndarray,
) -> pd.DataFrame:
    """Summarise amplitude, within-cluster coherence, and reference overlap.

    A candidate is more interesting when its cells have stronger CNV deviation,
    share a consistent profile, and are not mostly the non-tumour reference
    cells.  The composite score only ranks clusters for human review; it does
    not create a Neoplastic label.
    """
    rows: list[dict[str, float | int | str]] = []
    for cluster in sorted(pd.unique(cnv_clusters), key=lambda value: int(value) if value.isdigit() else value):
        mask = cnv_clusters == cluster
        profiles = cnv_profiles[mask]
        rows.append(
            {
                "cnv_cluster": str(cluster),
                "n_cells": int(mask.sum()),
                "mean_absolute_cnv_amplitude": float(np.mean(np.abs(profiles))),
                "cnv_profile_coherence": mean_profile_coherence(profiles),
                "normal_reference_fraction": float(np.mean(reference_categories[mask] != "Observation")),
                "spatial_neighbor_consistency": float(np.mean(spatial_consistency[mask])),
            }
        )
    audit = pd.DataFrame(rows)
    # ---- CNV 候选综合分(固定权重,无标签)----
    # 0.45*CNV 幅值百分位 + 0.35*簇内剖面一致性 + 0.10*"非参考细胞占比"取反
    # + 0.10*空间邻居一致性。只用于给 cluster 排序供人工复核,不产生标签。
    audit["cnv_candidate_score"] = (
        0.45 * percentile(audit["mean_absolute_cnv_amplitude"])
        + 0.35 * percentile(audit["cnv_profile_coherence"])
        + 0.10 * percentile(audit["normal_reference_fraction"], higher_is_better=False)
        + 0.10 * percentile(audit["spatial_neighbor_consistency"])
    )
    # These are deliberately conservative, fixed gates rather than thresholds
    # fitted against official labels.  "Automatic candidate" still means a
    # high-confidence seed source, never a confirmed clinical tumour call.
    audit["automatic_high_confidence_candidate"] = (
        (audit["mean_absolute_cnv_amplitude"] >= 0.25)
        & (audit["cnv_profile_coherence"] >= 0.70)
        & (audit["normal_reference_fraction"] <= 0.05)
        & (audit["spatial_neighbor_consistency"] >= 0.50)
    )
    return audit.sort_values("cnv_candidate_score", ascending=False).reset_index(drop=True)


def save_candidate_plot(audit: pd.DataFrame, output: Path) -> None:
    """Make a compact plot to review together with the CNV heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = np.clip(np.sqrt(audit["n_cells"]) * 6, 40, 550)
    points = ax.scatter(
        audit["mean_absolute_cnv_amplitude"],
        audit["cnv_profile_coherence"],
        s=sizes,
        c=audit["spatial_neighbor_consistency"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.8,
    )
    for row in audit.itertuples():
        ax.annotate(str(row.cnv_cluster), (row.mean_absolute_cnv_amplitude, row.cnv_profile_coherence), xytext=(4, 3), textcoords="offset points")
    fig.colorbar(points, ax=ax, label="Spatial neighbour consistency")
    ax.set(
        xlabel="Mean absolute CNV amplitude",
        ylabel="Within-cluster CNV-profile coherence",
        title="CNV clusters ranked for manual review (no official labels)",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnv-adata",
        type=Path,
        default=ROOT / "results/02_insitucnv/rerun/adata_cnv.h5ad",
    )
    parser.add_argument("--cluster-column", default="cnv_leiden_res0.2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/02_insitucnv/rerun")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.cnv_adata)
    if "X_cnv" not in adata.obsm:
        raise KeyError("Missing adata.obsm['X_cnv']; run code/02 first.")
    if args.cluster_column not in adata.obs:
        raise KeyError(f"Missing CNV cluster column {args.cluster_column!r}.")
    if "cnv_reference_category" not in adata.obs:
        raise KeyError("Missing CNV reference categories from code/02.")
    if "spatial" not in adata.obsm:
        raise KeyError("Missing adata.obsm['spatial']; spatial consistency is required.")

    cnv_clusters = adata.obs[args.cluster_column].astype(str).to_numpy()

    audit = build_cnv_audit(
        to_dense(adata.obsm["X_cnv"]),
        cnv_clusters,
        adata.obs["cnv_reference_category"].astype(str).to_numpy(),
        spatial_neighbor_consistency(np.asarray(adata.obsm["spatial"], dtype=float), cnv_clusters),
    )
    audit.to_csv(args.out_dir / "cnv_candidate_audit_label_free.csv", index=False)
    save_candidate_plot(audit, args.out_dir / "cnv_candidate_audit_label_free.png")
    print(audit.to_string(index=False))
    print("Review this table together with the CNV heatmap and spatial plot before selecting seeds.")


if __name__ == "__main__":
    main()
