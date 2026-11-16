"""Build PCA features and measure label-free cluster structure."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse


def expression_features(
    adata: sc.AnnData, n_components: int, representation_key: str = "X_pca"
) -> np.ndarray:
    # 取每个细胞的表达特征向量（默认 = 前 30 个 PCA 坐标）。
    # 三条路径：
    #   1) X_pca：CDGE 预处理时算好存在 obsm 里的 PCA，直接截前 30 维 ——
    #      注意这是对全部细胞拟合的无标签压缩（transductive），常规做法；
    #   2) 其他表示（如 scGPT embedding，第 4 步实验传入 "X_scGPT"）：
    #      先中心化再做 SVD 压到 30 维，保证不同表示在相同维度数下公平对比；
    #   3) 都没有时：直接对 280 维表达矩阵标准化后 SVD 现算 PCA。
    if representation_key in adata.obsm and adata.obsm[representation_key].shape[1] >= 2:
        matrix = np.asarray(adata.obsm[representation_key], dtype=np.float32)
        if representation_key == "X_pca":
            return matrix[:, :n_components]
        components = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        return (left[:, :components] * singular_values[:components]).astype(np.float32)
    if representation_key != "X_pca":
        raise KeyError(
            f"Representation {representation_key!r} is unavailable. "
            f"Available obsm keys: {list(adata.obsm)}"
        )

    matrix = adata.X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = (matrix - matrix.mean(axis=0)) / np.maximum(matrix.std(axis=0), 1e-6)
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    _, _, right = np.linalg.svd(matrix, full_matrices=False)
    return matrix @ right[:n_components].T


def centroid_silhouette(features: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Fast cohesion/separation proxy without the O(n^2) silhouette cost."""
    # 逐细胞的"轮廓分数"近似（真 silhouette 要 O(n^2) 两两距离，2 万细胞算不动）：
    #   对每个细胞算 (到最近异 cluster 中心的距离 - 到自己 cluster 中心的距离)
    #               / max(两个距离)
    #   > 0 表示这个细胞离"自家中心"更近 → cluster 内部紧密、边界清楚。
    # 再对 cluster 内取平均，就是表格里的 mean_silhouette（结构可靠性成分之一）。
    unique = np.unique(clusters)
    centroids = np.vstack([features[clusters == cluster].mean(axis=0) for cluster in unique])
    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    cluster_index = {cluster: index for index, cluster in enumerate(unique)}
    own_index = np.array([cluster_index[cluster] for cluster in clusters])
    own_distance = distances[np.arange(len(features)), own_index]
    distances[np.arange(len(features)), own_index] = np.inf
    other_distance = distances.min(axis=1)
    return (other_distance - own_distance) / np.maximum(np.maximum(own_distance, other_distance), 1e-8)


def cluster_multimodality_dip(
    cluster_features: np.ndarray, max_components: int = 3
) -> tuple[float, float, int]:
    """Test whether a cluster has evidence of multiple internal expression modes.

    Hartigan's Dip Statistic is defined for a one-dimensional distribution,
    whereas a cell has many PCA coordinates.  We therefore make a *local* PCA
    within the cluster and test its first few independent variation directions.
    The returned p-value is Bonferroni-adjusted across those directions.

    This is deliberately a diagnostic only at first: it does not use official
    labels and it is not silently folded into the reliability score.
    """
    # The project keeps this small optional dependency separate for the local
    # Python 3.11 CDGE environment; it does not alter the original CDGE venv.
    vendor = Path(__file__).parent / ".vendor_py311"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        from diptest import diptest  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "Hartigan Dip test requires the local diptest package. "
            "Install it in outputs/cdge_teacher_followup_project/.vendor_py311."
        ) from error

    if len(cluster_features) < 20:
        return math.nan, math.nan, 0
    centered = cluster_features - cluster_features.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    usable = int(np.sum(singular_values > 1e-8))
    n_components = min(max_components, usable)
    if n_components == 0:
        return 0.0, 1.0, 0

    local_scores = centered @ right_vectors[:n_components].T
    statistics: list[float] = []
    p_values: list[float] = []
    for component in range(n_components):
        statistic, p_value = diptest(local_scores[:, component], boot_pval=False)
        statistics.append(float(statistic))
        p_values.append(float(p_value))
    # A cluster is checked along several directions, so adjust the smallest
    # p-value rather than treating any one direction as a separate discovery.
    adjusted_p = min(1.0, min(p_values) * n_components)
    return max(statistics), adjusted_p, n_components
