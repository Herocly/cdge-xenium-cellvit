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
    """Convert InSituCNV's sparse or dense embedding to a NumPy array.

    【中文】X_cnv 稀疏/稠密统一转 NumPy 稠密数组（main 调用一次）。"""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)


def mean_profile_coherence(profiles: np.ndarray) -> float:
    """Measure whether cells in one CNV cluster share one CNV pattern.

    【中文】CNV cluster 内"剖面一致性"：每个细胞对簇中心剖面的余弦
    相似度，再取平均。高(→1) = 该 cluster 的细胞共享同一种拷贝数变异
    模式；低 = 混杂了多种模式（不可信的肿瘤候选）。
    参数: profiles (cluster 内细胞数 × 窗口数)。被 build_cnv_audit 调用。"""
    centre = profiles.mean(axis=0)
    denominator = np.linalg.norm(profiles, axis=1) * max(np.linalg.norm(centre), 1e-12)
    return float(np.mean((profiles @ centre) / np.maximum(denominator, 1e-12)))


def percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Make metrics with different scales comparable without fitting a label.

    【中文】百分位排名（0~1）：不同量纲的指标可以等权加权组合；
    higher_is_better=False 时取反（1-排名），如"参考细胞占比"越低越好。
    纯无标签变换。被 build_cnv_audit 的综合分公式调用。"""
    score = values.rank(method="average", pct=True)
    return score if higher_is_better else 1.0 - score


def spatial_neighbor_consistency(
    coordinates: np.ndarray, cnv_clusters: np.ndarray, n_neighbors: int = 20
) -> np.ndarray:
    """Return the fraction of spatial neighbours in the same CNV cluster.

    【中文】逐细胞的"空间邻居一致性"：cKDTree 找空间最近的 20 个邻居
    （不含自己），算邻居中与自己同 CNV cluster 的比例。肿瘤在组织里
    通常空间成片，一致性高的 cluster 更值得怀疑是肿瘤。
    参数: coordinates(XRrun 的 obsm["spatial"]), cnv_clusters(每细胞 CNV 簇)
    返回: 长度=细胞数的数组。被 main 调用后按 cluster 取均值。"""
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

    【中文】构建 CNV cluster 审计表（本脚本核心函数，main 调用）。
    ------------------------------------------------------------------
    参数:
        cnv_profiles         X_cnv 稠密矩阵（细胞 × 窗口）
        cnv_clusters         每细胞的 CNV cluster 标签
        reference_categories 每细胞的 CNV 参考类别（"Observation" 或种子类）
        spatial_consistency  spatial_neighbor_consistency 的逐细胞输出
    每个 cluster 一行，四个原始指标：
        mean_absolute_cnv_amplitude  平均 |CNV|（偏离正常参考的强度）
        cnv_profile_coherence        簇内剖面一致性（mean_profile_coherence）
        normal_reference_fraction    簇内参考细胞占比（越高越"不是肿瘤候选"）
        spatial_neighbor_consistency 空间邻居一致性均值
    返回: 按综合分降序的 DataFrame（含 cnv_candidate_score 列和
    automatic_high_confidence_candidate 布尔列，公式见行内注释）。
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
    """Make a compact plot to review together with the CNV heatmap.

    【中文】画审计散点图（横轴 CNV 幅值、纵轴簇内一致性、点面积=细胞数、
    颜色=空间一致性、每个点标注 cluster 号），保存 180dpi PNG，
    供人工复核时与 CNV 热图对照。被 main 调用。"""
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
    """【中文】主流程：读 05 的 adata_cnv.h5ad → 校验四样必需输入
    (X_cnv / CNV cluster 列 / cnv_reference_category / spatial) →
    算空间邻居一致性 → build_cnv_audit 生成审计表 → 写 CSV + PNG →
    打印表格并提醒"须配合热图与空间图人工复核后再选种子"。
    参数:
      --cnv-adata      05 输出（默认 results/02_insitucnv/rerun/adata_cnv.h5ad）
      --cluster-column 用哪个 CNV 聚类列（默认 cnv_leiden_res0.2 主分辨率）
      --out-dir        输出目录
    """
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
