"""Prevent classifier expansion from contradicting strong cluster evidence.

本模块是"cluster 级证据"在"cell 级扩展"上的否决器:
即使分类器对某个细胞很自信,若其所属 cluster 有强证据支持别的标签,
该预测也会被 gate ④(allowed)拦下。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_CLASSES = {"Connective_tissue", "Inflammatory", "Epithelial", "Neoplastic"}


def epithelial_marker_conflict_mask(clusters: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Reuse the frozen two-database epithelial conflict rule at cell level."""
    required = {"cluster", "panglao_label", "cellmarker_label"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing conflict-gate columns: {sorted(required - set(table.columns))}")
    keyed = table.assign(cluster=table["cluster"].astype(str)).set_index("cluster")
    if not keyed.index.is_unique:
        raise ValueError("Conflict-gate cluster IDs must be unique")
    aligned = keyed.reindex(np.asarray(clusters).astype(str))
    if aligned[["panglao_label", "cellmarker_label"]].isna().any().any():
        raise ValueError("Conflict gate cannot align all expression clusters")
    return (aligned["panglao_label"].eq("Epithelial") & aligned["cellmarker_label"].eq("Epithelial")).to_numpy()


def strong_cluster_label_map(
    cluster_table: pd.DataFrame,
    min_structural_score: float = 2 / 3,
    min_annotation_score: float = 2 / 3,
) -> dict[str, str]:
    """Return labels only for clusters with strong label-free evidence.

    A cluster needs both structural coherence and a clear marker-voting signal.
    Clusters without both forms of evidence remain unrestricted, so the
    classifier can still improve uncertain regions.

    只有"结构分≥2/3 且投票分≥2/3 且 margin>0 且有独立库支持"的 cluster 才进入
    返回的 {cluster: label} 映射 —— 弱证据 cluster 不锁标签,分类器仍可修正它。
    与 03 的 Tier A 条件完全一致(Tier A 选种子,gate ④ 用同一条件做否决)。
    """
    strong = cluster_table.loc[
        cluster_table["pseudo_label"].isin(CORE_CLASSES)
        & (cluster_table["structural_reliability_score"] >= min_structural_score)
        & (cluster_table["annotation_confidence_score"] >= min_annotation_score)
        & (cluster_table["vote_margin"] > 0)
        & (cluster_table["source_support_count"] > 0)
    ]
    return dict(zip(strong["cluster"].astype(str), strong["pseudo_label"].astype(str)))


def prediction_allowed_by_cluster_evidence(
    clusters: np.ndarray, predicted: np.ndarray, strong_labels: dict[str, str]
) -> np.ndarray:
    """Allow a prediction unless it contradicts a strong existing cluster label."""
    evidence = np.asarray([strong_labels.get(str(cluster), "") for cluster in clusters], dtype=object)
    # Neoplastic predictions still need C and structural support downstream.
    return (evidence == "") | (evidence == predicted) | ((evidence == "Epithelial") & (predicted == "Neoplastic"))
