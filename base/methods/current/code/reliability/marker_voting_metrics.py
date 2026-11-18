"""Turn the existing CDGE marker-voting result into confidence measures.

【中文详解 —— 投票结果量化模块】
====================================================================
只有一个函数 vote_metrics：把 02 里 CDGE 原版 classify_cluster 的
投票结果提炼成 4 个无标签指标（vote_confidence / vote_margin /
vote_total / marker_coverage），作为 annotation_confidence_score 的
原材料。详细公式与实例见函数内中文注释。
"""

from __future__ import annotations

import math

import numpy as np


def vote_metrics(result: dict, top_genes: list[str], gene_votes: dict) -> dict[str, float]:
    # 把 marker voting 的结果提炼成 4 个无标签指标（第 1 步"投票可信度"的量化）：
    #   vote_confidence  赢家类别票数占总票数比例（票是否集中）
    #   vote_margin      第一名与第二名的票差占比（赢得是否干脆）
    #   vote_total       总票数
    #   marker_coverage  cluster top 基因里有多少能在数据库查到
    # 实例：贴对的 cluster 4 → confidence 0.84 / coverage 0.80；
    #       贴错的 cluster 23（缺氧肿瘤被投成 Connective）→ coverage 只有 0.25，
    #       且 CellMarker 直接弃权 —— 这些差异就是可靠性判据的原材料。
    if not result or not gene_votes:
        return {
            "vote_confidence": math.nan,
            "vote_margin": math.nan,
            "vote_total": math.nan,
            "marker_coverage": math.nan,
        }
    votes = np.array(list(result.get("total_votes", {}).values()), dtype=float)
    vote_sum = float(votes.sum())
    ordered = np.sort(votes)[::-1] if votes.size else np.array([])
    top1 = float(ordered[0]) if ordered.size else 0.0
    top2 = float(ordered[1]) if ordered.size > 1 else 0.0
    covered = sum(gene in gene_votes for gene in top_genes)
    return {
        "vote_confidence": float(result.get("confidence", 0.0)),
        "vote_margin": (top1 - top2) / vote_sum if vote_sum else 0.0,
        "vote_total": vote_sum,
        "marker_coverage": covered / len(top_genes) if top_genes else 0.0,
    }
