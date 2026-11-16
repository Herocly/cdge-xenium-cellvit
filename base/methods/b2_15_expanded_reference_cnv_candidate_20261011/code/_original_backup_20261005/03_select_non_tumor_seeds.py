#!/usr/bin/env python3
"""Choose label-free Tier A / Tier B seeds with an explicit fallback policy.

Tier A is the strict original seed rule.  Tier B is used only when a marker
class has no Tier A cluster and still requires independently supported,
non-ambiguous evidence.  No truth labels are loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


CORE_CLASSES = ("Connective_tissue", "Inflammatory", "Epithelial", "Neoplastic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--cluster-evidence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cluster-key", default="leiden_res_2.00")
    parser.add_argument("--core-fraction", type=float, default=.30)
    parser.add_argument("--seed-policy", choices=("single", "multi"), default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.adata)
    clusters = adata.obs[args.cluster_key].astype(str).to_numpy()
    pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)[:, :30]
    table = pd.read_csv(args.cluster_evidence, dtype={"cluster": str})
    # 只考虑四类核心标签的 cluster(过滤掉 Others/Unknown 的投票结果)。
    table = table.loc[table["pseudo_label"].isin(CORE_CLASSES)].copy()

    # ---- cluster 级入选规则:Tier A(严格)与 Tier B(兜底)----
    # Tier A = 原始严格规则:结构分和投票分都要 >= 2/3,且投票有明确差距
    # (margin>0)且至少一个独立数据库支持(source_support>0)。
    # Tier B 只在某类(如 Neoplastic)没有任何 Tier A cluster 时启用,
    # 阈值整体放宽但仍要求独立数据库支持。两个阈值组都是预先固定的,
    # 不依据任何 ground truth 拟合(见注释原话)。
    table["tier_a"] = (
        (table["structural_reliability_score"] >= 2 / 3)
        & (table["annotation_confidence_score"] >= 2 / 3)
        & (table["vote_margin"] > 0)
        & (table["source_support_count"] > 0)
    )
    table["tier_b"] = (
        (table["structural_reliability_score"] >= .50)
        & (table["annotation_confidence_score"] >= .50)
        & (table["marker_coverage"] >= .40)
        & (table["vote_margin"] >= .25)
        & (table["source_support_count"] > 0)
    )
    labels = np.full(adata.n_obs, "Unknown", dtype=object)
    selected: list[dict[str, object]] = []
    # ---- 每类只选"最可靠的那一个 cluster",再取它的核心细胞 ----
    # 这是 cluster→cell 的关键转换点:先在 cluster 级选出该类可靠性最高的
    # 一个 cluster,然后取该 cluster 内到 PCA 中心距离最近的 core_fraction
    # (30%)细胞,把它们整体继承 cluster 的 coarse label 作为 initial seed。
    # 其余细胞(含同 cluster 的 70% 边缘细胞)保持 Unknown。
    # 注意:每类只选一个 cluster 是本脚本的保守设计;Epithelial 的多状态
    # 种子由 08/09 单独处理。
    for label in CORE_CLASSES:
        candidates = table.loc[table["pseudo_label"].eq(label)]
        tier = "Tier_A" if candidates["tier_a"].any() else "Tier_B"
        eligible = candidates.loc[candidates["tier_a"] if tier == "Tier_A" else candidates["tier_b"]]
        if eligible.empty:
            continue
        rows = eligible.sort_values(["reliability_score", "cluster"], ascending=[False, True])
        if args.seed_policy == "single":
            rows = rows.head(1)
        core_groups = []
        for row in rows.to_dict("records"):
            positions = np.flatnonzero(clusters == row["cluster"])
            centre = pca[positions].mean(axis=0)
            distance = np.linalg.norm(pca[positions] - centre, axis=1)
            core_groups.append(positions[np.argsort(distance, kind="stable")[:max(1, int(len(positions) * args.core_fraction))]])
        # Keep the original best-cluster seed budget: test diversity, not volume.
        allocation = allocate_seed_budget([len(group) for group in core_groups], len(core_groups[0]))
        for row, candidates_core, count in zip(rows.to_dict("records"), core_groups, allocation):
            core = candidates_core[:count]
            labels[core] = label
            selected.append({
                "label": label, "tier": tier, "source_leiden_cluster": row["cluster"],
                "cluster_size": int((clusters == row["cluster"]).sum()), "selected_core_cells": len(core),
                **{key: row[key] for key in ("structural_reliability_score", "annotation_confidence_score", "marker_coverage", "vote_margin", "source_support_count", "top_marker_genes")},
                "official_labels_policy": "not_loaded",
            })
    output = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str), "is_initial_seed": labels != "Unknown",
        "refined_pseudo_label": labels, "source_leiden_cluster": clusters,
    })
    output.to_csv(args.out_dir / "tiered_initial_seeds_no_official_labels.csv", index=False)
    pd.DataFrame(selected).to_csv(args.out_dir / "tiered_seed_cluster_summary_no_official_labels.csv", index=False)
    config = {
        "official_labels_policy": "not_loaded", "core_fraction": args.core_fraction,
        "seed_policy": args.seed_policy,
        "tier_a": "structural>=0.667, annotation>=0.667, margin>0, source_support>0",
        "tier_b": "only for a missing class: structural>=0.50, annotation>=0.50, coverage>=0.40, margin>=0.25, source_support>0",
        "selected_classes": [row["label"] for row in selected],
        "unlabeled_missing_classes": sorted(set(CORE_CLASSES) - {row["label"] for row in selected}),
    }
    (args.out_dir / "tiered_seed_config_no_official_labels.json").write_text(json.dumps(config, indent=2))
    print(pd.DataFrame(selected).to_string(index=False))
    print(json.dumps(config, indent=2))


def allocate_seed_budget(capacities: list[int], budget: int) -> list[int]:
    allocation = [0] * len(capacities)
    while budget > 0:
        available = [i for i, size in enumerate(capacities) if allocation[i] < size]
        if not available:
            break
        share = max(1, budget // len(available))
        for index in available:
            count = min(share, capacities[index] - allocation[index], budget)
            allocation[index] += count
            budget -= count
    return allocation


if __name__ == "__main__":
    main()
