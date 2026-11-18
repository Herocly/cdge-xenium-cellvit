"""Label-free gates shared by iterative pseudo-labeling experiments.

【中文详解 —— 迭代伪标签的"无标签 gate"模块】
====================================================================
被 11 调用（11 的主循环每轮都用本模块的 gate 组合）。五个函数分工：
  neoplastic_support_from_evidence    Neoplastic 的 C 证据对齐与解析
  spatial_quadrants                   坐标中位数切 2x2 空间四象限
  supporting_cluster_evidence         gate ⑤：cluster 证据支持预测
  accepted_candidate_mask             五重 gate 汇总（谁能当本轮候选）
  select_spatially_balanced_additions 排序+每类每象限预算内选细胞
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cluster_evidence_gate import epithelial_marker_conflict_mask


def neoplastic_support_from_evidence(
    cell_ids: np.ndarray, clusters: np.ndarray, table: pd.DataFrame, evidence: pd.DataFrame,
) -> np.ndarray:
    """C supports expansion; D remains the stricter initial-seed definition.

    【中文】把 09 证据表的 variant_C 布尔列严格对齐到分析细胞轴
    （11 的 main 启动时调用一次，结果喂给 gate ⑤）。
    校验：证据表必须有 cell_id/expression_cluster/variant_C 三列；
    细胞集合与分析轴完全一致；expression_cluster 必须与 adata 的
    cluster 列逐行相同（防错位）；布尔解析用严格规则。
    参数: cell_ids(分析轴), clusters(每细胞 cluster), table(02 表,保留
          接口), evidence(09 证据表)
    返回: 每细胞 bool——"该细胞有基因+CNV 双证据支持 Neoplastic"。"""
    required = {"cell_id", "expression_cluster", "variant_C_gene_plus_cnv"}
    if not required.issubset(evidence.columns):
        raise ValueError(f"Missing Neoplastic support columns: {sorted(required - set(evidence.columns))}")
    keyed = evidence.assign(cell_id=evidence["cell_id"].astype(str)).set_index("cell_id")
    ids = pd.Index(np.asarray(cell_ids).astype(str))
    if not keyed.index.is_unique or not ids.is_unique or set(keyed.index) != set(ids):
        raise ValueError("Neoplastic support must contain exactly one row per analysis cell")
    aligned = keyed.reindex(ids)
    if not np.array_equal(aligned["expression_cluster"].astype(str).to_numpy(), np.asarray(clusters).astype(str)):
        raise ValueError("Neoplastic support uses different expression clusters")
    raw = aligned["variant_C_gene_plus_cnv"]
    if raw.isna().any():
        raise ValueError("Missing C-evidence flags")
    normalized = raw.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Invalid C-evidence flags")
    candidate = normalized.isin({"true", "1"}).to_numpy()
    # Epithelial lineage does not imply a benign state. C still requires A+B.
    return candidate


def spatial_quadrants(coordinates: np.ndarray) -> np.ndarray:
    """Split a tissue into four fixed spatial regions without annotations.

    按坐标中位数把切片切成 2x2 四象限(0..3),用途有二:
    ① 12/13 的四折 spatial holdout(每折留一个象限做纯预测);
    ② 11 的每轮扩展按象限均衡取样(per_region = cap // 4)。
    纯几何切分,无任何标签。
    """
    x_high = coordinates[:, 0] >= np.median(coordinates[:, 0])
    y_high = coordinates[:, 1] >= np.median(coordinates[:, 1])
    return (x_high.astype(int) + 2 * y_high.astype(int)).astype(int)


def supporting_cluster_evidence(
    clusters: np.ndarray,
    predicted: np.ndarray,
    cluster_table: pd.DataFrame,
    min_structural_score: float,
    min_annotation_score: float,
    neoplastic_support: np.ndarray | None = None,
) -> np.ndarray:
    """Require matching marker evidence and a structurally reliable cluster.

    【gate ⑤ supported】逐细胞判定:该细胞所属 Leiden cluster 的 coarse
    marker vote 必须与分类器预测一致,且该 cluster 结构分≥2/3、投票分≥2/3、
    margin>0、有独立数据库支持。它是 cluster 级证据表在 cell-level gate 上的
    直接投影 —— 表达聚类认为"这个 cluster 是 X"的细胞,不允许被分类器改成 Y。
    """
    table = cluster_table.set_index("cluster")
    supported = np.zeros(len(clusters), dtype=bool)
    if len(predicted) != len(clusters):
        raise ValueError("Predictions and clusters must have matching lengths")
    if neoplastic_support is not None and np.asarray(neoplastic_support).shape != (len(clusters),):
        raise ValueError("Neoplastic support must match the analysis cell axis")
    if np.any(predicted == "Neoplastic") and neoplastic_support is None:
        raise ValueError("Neoplastic expansion requires explicit gene+CNV evidence")
    for position, (cluster, label) in enumerate(zip(clusters, predicted)):
        if cluster not in table.index:
            continue
        row = table.loc[cluster]
        if label == "Neoplastic":
            supported[position] = bool(neoplastic_support[position] and row["structural_reliability_score"] >= min_structural_score)
            continue
        supported[position] = bool(
            row["pseudo_label"] == label
            and row["structural_reliability_score"] >= min_structural_score
            and row["annotation_confidence_score"] >= min_annotation_score
            and row["vote_margin"] > 0
            and row["source_support_count"] > 0
        )
        if label == "Epithelial" and neoplastic_support is not None and neoplastic_support[position]:
            supported[position] = False
    return supported


def accepted_candidate_mask(
    unlabelled: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    neighbour_agreement: np.ndarray,
    allowed_by_cluster_evidence: np.ndarray,
    supported_by_cluster_evidence: np.ndarray,
    min_confidence: float,
    min_margin: float,
    min_neighbour_agreement: float,
) -> np.ndarray:
    """Return cells eligible to become pseudo labels in this iteration.

    【五重 gate 汇总处】一个 Unknown 细胞当轮能否成为 pseudo label,必须同时满足:
      unlabelled       仍是 Unknown(seed_round<0)
      confidence       ≥ min_confidence(0.80) 分类器最大类概率
      margin           ≥ min_margin(0.20)   前两名概率差
      neighbour_agreement ≥ 0.70            kNN 邻居同预测比例
      allowed          未与强证据 cluster 标签矛盾(gate ④)
      supported        所属 cluster 证据支持该预测(gate ⑤)
    任何一个不满足 → 该细胞当轮保持 Unknown(下一轮可再竞争)。
    """
    return (
        unlabelled
        & (confidence >= min_confidence)
        & (margin >= min_margin)
        & (neighbour_agreement >= min_neighbour_agreement)
        & allowed_by_cluster_evidence
        & supported_by_cluster_evidence
    )


def select_spatially_balanced_additions(
    eligible: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    spatial_region: np.ndarray,
    classes: tuple[str, ...],
    max_per_class: int,
    rank_score: np.ndarray | None = None,
) -> np.ndarray:
    """Pick high-scoring candidates while balancing class and tissue region.

    ``rank_score`` is optional so existing experiments retain their original
    confidence-first ordering. The formal iterative pipeline supplies a
    label-free combined score after all hard reliability gates have passed.
    """
    additions: list[int] = []
    per_region = max(1, max_per_class // 4)
    for label in classes:
        for region in range(4):
            positions = np.flatnonzero(
                eligible & (predicted == label) & (spatial_region == region)
            )
            if rank_score is None:
                # Confidence is primary; margin breaks close confidence ties.
                order = np.lexsort((-margin[positions], -confidence[positions]))
            else:
                # All candidates have passed fixed gates. Rank the remaining
                # cells by the pre-registered combined, label-free score.
                order = np.lexsort(
                    (-margin[positions], -confidence[positions], -rank_score[positions])
                )
            additions.extend(positions[order[:per_region]].tolist())
    return np.asarray(additions, dtype=int)
