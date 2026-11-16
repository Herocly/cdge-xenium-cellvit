"""Export the fixed D-variant candidates as Neoplastic seeds for training.

【中文详解 —— 流水线第 7 步：导出 Neoplastic（肿瘤）初始种子】
====================================================================
流水线位置：09(构建 A/B/C/D 证据表) → 【07 本脚本】 → 10(合并全部种子)

做什么：
    09 的证据表里 variant_D_all_evidence=True 的细胞（基因+CNV+空间+结构
    四重证据齐全）是最保守的肿瘤候选。本脚本把这些细胞原样导出成
    Neoplastic 种子表，不做任何新的证据计算。

    可选开关 --exclude-epithelial-marker-conflicts：
      若某细胞所在的表达 cluster 在 02 证据表里被 PanglaoDB 和 CellMarker
      【两个独立数据库同时】标为 Epithelial（epithelial_marker_conflict_mask，
      reliability/cluster_evidence_gate.py），则该细胞的肿瘤种子被否决——
      "两库都说它是正常上皮"的 cluster 不允许再当肿瘤种子（保守去冲突）。

输入 → 输出：
    --evidence-table (09 的 neoplastic_evidence_table_*.csv)
    → neoplastic_seed_candidates.csv
      （cell_id, is_neoplastic_seed, marker_conflict_excluded, pseudo_label）
    → neoplastic_conflict_gate_audit_*.csv（冲突门统计）
    输出由 10 的 --neoplastic-seeds 参数消费。

与其他模块的连接：
    reliability.data_contracts.boolean_flags —— 把 CSV 里的 "True/False/1/0"
    字符串严格解析成布尔数组（含缺失/非法值直接报错）。
    reliability.cluster_evidence_gate.epithelial_marker_conflict_mask ——
    逐细胞查其 cluster 是否"两库皆为 Epithelial"。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.cluster_evidence_gate import epithelial_marker_conflict_mask  # noqa: E402
from reliability.data_contracts import boolean_flags  # noqa: E402


def main() -> None:
    """【中文】主流程（本脚本唯一函数）：
      1. 读 09 的逐细胞证据表；若表里存了多个 variant 行，可用
         --evidence-variant 只保留指定 variant；
      2. 取出 variant_D_all_evidence 列 → boolean_flags 严格解析成布尔，
         作为 is_neoplastic_seed（cell_id 必须唯一非空）；
      3.（可选）冲突门：需要证据表带 leiden_cluster/expression_cluster 列，
         读 02 的 cluster 表(只要 cluster/panglao_label/cellmarker_label 三列)，
         merge 后用 epithelial_marker_conflict_mask 找出"两库皆上皮"的
         cluster，把这些 cluster 里的种子改判 Unknown，并在
         marker_conflict_excluded 列留痕；
      4. 写出种子 CSV + 冲突门审计 CSV，打印导出数量。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-table",
        type=Path,
        default=ROOT / "results/07_neoplastic_evidence/neoplastic_evidence_table_no_official_labels.csv",
    )
    parser.add_argument("--cluster-evidence", type=Path)
    parser.add_argument(
        "--evidence-variant",
        help="Optional row filter for historical evidence tables that store multiple variants.",
    )
    parser.add_argument(
        "--exclude-epithelial-marker-conflicts",
        action="store_true",
        help=(
            "Do not export a variant-D Neoplastic seed when both PanglaoDB and "
            "CellMarker support Epithelial for that cell's expression cluster."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/09_multievidence_neoplastic_seeds")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(args.evidence_table)
    if args.evidence_variant:
        if "variant" not in evidence.columns:
            raise ValueError("--evidence-variant was provided, but the evidence table has no 'variant' column.")
        evidence = evidence.loc[evidence["variant"].astype(str).eq(args.evidence_variant)].copy()
        if evidence.empty:
            raise ValueError(f"No evidence rows found for variant {args.evidence_variant!r}.")
    if args.exclude_epithelial_marker_conflicts and not args.cluster_evidence:
        raise ValueError("--cluster-evidence is required with --exclude-epithelial-marker-conflicts.")

    output = evidence[["cell_id", "variant_D_all_evidence"]].copy()
    if not output["cell_id"].is_unique or output["cell_id"].isna().any():
        raise ValueError("Neoplastic evidence must have unique nonmissing cell IDs")
    output["is_neoplastic_seed"] = boolean_flags(output.pop("variant_D_all_evidence"))
    output["marker_conflict_excluded"] = False
    if args.exclude_epithelial_marker_conflicts:
        cluster_column = next(
            (column for column in ("leiden_cluster", "expression_cluster") if column in evidence.columns),
            None,
        )
        if cluster_column is None:
            raise ValueError("Neoplastic evidence needs either 'leiden_cluster' or 'expression_cluster' for the conflict gate.")
        cluster_table = pd.read_csv(args.cluster_evidence, dtype={"cluster": str})
        required = {"cluster", "panglao_label", "cellmarker_label"}
        missing = required.difference(cluster_table.columns)
        if missing:
            raise ValueError(f"Cluster evidence misses conflict-gate columns: {sorted(missing)}")
        cluster_table = cluster_table[list(required)].copy()
        cluster_table["cluster"] = cluster_table["cluster"].astype(str)
        output["cluster_for_conflict_gate"] = evidence[cluster_column].astype(str)
        output = output.merge(cluster_table, left_on="cluster_for_conflict_gate", right_on="cluster", how="left")
        conflict = epithelial_marker_conflict_mask(output["cluster_for_conflict_gate"].to_numpy(), cluster_table)
        output["marker_conflict_excluded"] = output["is_neoplastic_seed"] & conflict
        output.loc[output["marker_conflict_excluded"], "is_neoplastic_seed"] = False

    output["pseudo_label"] = output["is_neoplastic_seed"].map(
        {True: "Neoplastic", False: "Unknown"}
    )
    output.to_csv(args.out_dir / "neoplastic_seed_candidates.csv", index=False)
    output.groupby("marker_conflict_excluded", dropna=False).size().rename("cell_count").reset_index().to_csv(
        args.out_dir / "neoplastic_conflict_gate_audit_no_official_labels.csv", index=False
    )
    print(f"Exported {int(output['is_neoplastic_seed'].sum())} multi-evidence Neoplastic seeds.")


if __name__ == "__main__":
    main()
