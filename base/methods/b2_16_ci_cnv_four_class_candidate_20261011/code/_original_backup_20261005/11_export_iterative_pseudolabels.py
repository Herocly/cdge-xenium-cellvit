#!/usr/bin/env python3
"""Export final selective pseudo labels using Epithelial program classes.

No official annotation is loaded. Program labels are retained for audit and
merged to the four project output labels for downstream CellViT supervision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.cluster_evidence_gate import prediction_allowed_by_cluster_evidence, strong_cluster_label_map  # noqa: E402
from reliability.expression_space import expression_features  # noqa: E402
from reliability.iterative_expansion import accepted_candidate_mask, select_spatially_balanced_additions, spatial_quadrants, supporting_cluster_evidence, neoplastic_support_from_evidence  # noqa: E402
from reliability.seed_classifier import fit_balanced_logistic_classifier, prediction_neighbor_agreement  # noqa: E402


PROGRAM_PREFIX = "Epithelial_program_"


def core(labels: np.ndarray) -> np.ndarray:
    result = np.asarray(labels, dtype=object).copy()
    result[np.char.startswith(result.astype(str), PROGRAM_PREFIX)] = "Epithelial"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--cluster-evidence", type=Path, required=True)
    parser.add_argument("--neoplastic-evidence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=6)
    # Defaults preserve the frozen B0 behaviour.  Alternative values are
    # explicit experiment settings, not changes to the reference pipeline.
    parser.add_argument("--support-min-structural", type=float, default=2 / 3)
    parser.add_argument("--support-min-annotation", type=float, default=2 / 3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.adata)
    # ---- 读入 initial seeds,转成逐细胞标签数组 ----
    # labels:每个细胞的训练标签(seed 细承 cluster 标签,其余 Unknown);
    # seed_round:种子=0,Unknown=-1,后续每轮新增细胞记录其加入的轮次。
    # 【cluster→cell 转换在此完成】从这一步起,后续所有预测都是 per-cell 的。
    from reliability.data_contracts import aligned_cell_table, boolean_flags
    seed = aligned_cell_table(pd.read_csv(args.seed_file, dtype={"cell_id": str}), adata.obs_names.astype(str))
    labels = np.where(boolean_flags(seed["is_initial_seed"]), seed["training_label"], "Unknown").astype(object)
    if not np.isin(labels, ["Connective_tissue", "Inflammatory", "Epithelial", "Neoplastic", "Unknown"]).all():
        raise ValueError("Generic B1.8 seeds contain invalid training labels")
    seed_round = np.where(labels == "Unknown", -1, 0)
    # classifier 特征 = 前 30 个 PCA 坐标(与聚类同源的表达表示);
    # clusters 只是每细胞的 Leiden id,用于查 cluster 级 gate;
    # regions = 按坐标中位数切的四个空间象限(用于每轮按空间均衡选细胞)。
    features = expression_features(adata, 30, "X_pca")
    clusters = adata.obs["leiden_res_2.00"].astype(str).to_numpy()
    regions = spatial_quadrants(np.asarray(adata.obsm["spatial"], dtype=float))
    table = pd.read_csv(args.cluster_evidence, dtype={"cluster": str})
    neo_table = pd.read_csv(args.neoplastic_evidence, dtype={"cell_id": str, "expression_cluster": str})
    neo_support = neoplastic_support_from_evidence(adata.obs_names.astype(str).to_numpy(), clusters, table, neo_table)
    indexed = table.set_index("cluster")
    # 每个细胞继承其 cluster 的两个可靠性分,只用作"同分候选间的排序加分",
    # 不参与是否入选的硬 gate。
    structure = np.asarray([indexed.at[c, "structural_reliability_score"] if c in indexed.index else 0.0 for c in clusters])
    annotation = np.asarray([indexed.at[c, "annotation_confidence_score"] if c in indexed.index else 0.0 for c in clusters])
    # "强证据 cluster"映射:结构分≥2/3 且投票分≥2/3 且 margin>0 且有独立库支持
    # 的 cluster 才有资格"锁住"自己的标签(见 cluster_evidence_gate)。
    strong = strong_cluster_label_map(table)
    train_classes = tuple(sorted(np.unique(labels[labels != "Unknown"])))
    audit: list[pd.DataFrame] = []
    waterfalls = []

    # ---- 迭代扩展主循环 ----
    # 每一轮:用当前全部已标注细胞训练 multinomial Logistic Regression
    # (class_weight=balanced,特征=30 维 PCA),对全部细胞(含 Unknown)预测;
    # 只有同时通过 5 个 gate 的 Unknown 才有资格加入:
    #   ① confidence ≥ 0.80(预测概率的最大值)
    #   ② margin ≥ 0.20(第一名与第二名概率差)
    #   ③ neighbor agreement ≥ 0.70(kNN 邻居中预测同类比例)
    #   ④ allowed:预测不得与该 cluster 的"强证据标签"矛盾
    #   ⑤ supported:该 cluster 的 coarse vote 与预测一致且结构/投票分都 ≥2/3
    # 通过后再按 rank 分排序、每类每空间象限最多取 cap 个(program 166/类 500)。
    # 若本轮一个新增都没有 → break(因此"6 轮"只是上限,可能 1 轮就停)。
    for iteration in range(1, args.max_iterations + 1):
        train = np.flatnonzero(seed_round >= 0)
        predicted, confidence, margin, _ = fit_balanced_logistic_classifier(features, train, labels)
        # 程序标签(E 子类)先合并回 "Epithelial" 再做 cluster 级 gate 判断,
        # 因为 cluster 证据表里只有四类 coarse 标签。
        predicted_core = core(predicted)
        agreement = prediction_neighbor_agreement(adata, predicted)
        allowed = prediction_allowed_by_cluster_evidence(clusters, predicted_core, strong)
        supported = supporting_cluster_evidence(
            clusters,
            predicted_core,
            table,
            args.support_min_structural,
            args.support_min_annotation,
            neoplastic_support=neo_support,
        )
        eligible = accepted_candidate_mask(seed_round < 0, confidence, margin, agreement, allowed, supported, .80, .20, .70)
        # rank 分(无标签组合分):confidence 50% + margin 20% + 邻居一致 15%
        # + 结构分 10% + 投票分 5%。仅用于同 gate 通过者之间排序。
        rank = .50 * confidence + .20 * margin + .15 * agreement + .10 * structure + .05 * annotation
        additions: list[int] = []
        for label in train_classes:
            # 每类每轮上限:E 程序 166(三个子类合计 498 < 500,防止拆类稀释上限),
            # 其余类 500;再按四个空间象限均衡切分。
            cap = 166 if label.startswith(PROGRAM_PREFIX) else 500
            additions.extend(select_spatially_balanced_additions(eligible, predicted, confidence, margin, regions, (label,), cap, rank_score=rank).tolist())
        additions = np.asarray(sorted(set(additions)), dtype=int)
        for label in train_classes:
            remaining = (seed_round < 0) & (predicted == label)
            row = {"iteration": iteration, "class": label, "unlabelled_predicted_as_class": int(remaining.sum())}
            for name, gate in (("confidence", confidence >= .80), ("margin", margin >= .20),
                               ("neighbor", agreement >= .70), ("cluster_veto", allowed), ("evidence", supported)):
                remaining = remaining & gate
                row["after_" + name] = int(remaining.sum())
            row["selected_after_budget"] = int((predicted[additions] == label).sum())
            row["budget_truncated"] = row["after_evidence"] - row["selected_after_budget"]
            waterfalls.append(row)
        audit.append(pd.DataFrame({
            "iteration": iteration, "cell_id": adata.obs_names.astype(str), "leiden_cluster": clusters,
            "predicted_training_label": predicted, "predicted_core_label": predicted_core,
            "classifier_confidence": confidence, "classifier_margin": margin,
            "allowed_by_cluster_evidence": allowed, "neoplastic_C_evidence_supported": neo_support,
            "prediction_neighbor_agreement": agreement, "cluster_evidence_supported": supported,
            "selected_for_next_iteration": np.isin(np.arange(adata.n_obs), additions),
        }))
        if not len(additions):
            break
        labels[additions] = predicted[additions]
        seed_round[additions] = iteration

    # ---- final pseudo label 生成 ----
    # training_program_label:训练粒度标签(E 三子类分开保留,供审计);
    # refined_pseudo_label = core(labels):E 子类合回 "Epithelial" 后的
    # 四大类输出,即给下游 CellViT 用的最终 pseudo label。
    # seed_round 记录来源:0=initial seed,1..k=第 k 轮扩展,-1=始终 Unknown。
    final = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str), "seed_round": seed_round,
        "is_initial_seed": seed_round == 0, "training_program_label": labels,
        "refined_pseudo_label": core(labels), "spatial_region": regions,
    })
    final.to_csv(args.out_dir / "final_pseudolabels_no_official_labels.csv", index=False)
    pd.DataFrame(waterfalls).to_csv(args.out_dir / "gate_waterfall_no_official_labels.csv", index=False)
    pd.concat(audit, ignore_index=True).to_csv(args.out_dir / "candidate_audit_no_official_labels.csv", index=False)
    final.groupby(["refined_pseudo_label", "seed_round"], dropna=False).size().rename("cell_count").reset_index().to_csv(args.out_dir / "label_counts_no_official_labels.csv", index=False)
    pd.DataFrame({
        "max_iterations": [args.max_iterations],
        "support_min_structural": [args.support_min_structural],
        "support_min_annotation": [args.support_min_annotation],
        "official_labels_policy": ["not_loaded"],
    }).to_csv(args.out_dir / "run_config_no_official_labels.csv", index=False)


if __name__ == "__main__":
    main()
