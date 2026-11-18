#!/usr/bin/env python3
"""Combine the existing label-free initial seed sources for iterative training.

The script only aligns cell IDs and resolves contradictory initial labels
conservatively. It never reads reference annotations. Existing Connective /
Inflammatory seeds take precedence over a Neoplastic candidate. Epithelial
lineage alone is not proof of benign state; C conflicts are withheld and
complete D evidence can supply a Neoplastic seed.

【中文详解 —— 流水线第 10 步：合并全部初始种子并保守消解冲突】
====================================================================
流水线位置：03(C/I/E tiered 种子) + Epithelial 程序种子(08 产物,可选)
            + 07(Neoplastic D 种子) + 09(C 证据)
            → 【10 本脚本】 → 11(迭代自训练的输入)

做什么：
    把多个来源的初始种子对齐到同一细胞轴上，合成一张
    {cell_id, is_initial_seed, training_label, seed_source} 表。
    合并优先级与冲突策略（核心逻辑，全部"保守优先"）：
      ① C/I(+E) 的 tiered cluster 核心种子先落位（seed_source=
         reliable_cluster_core）；
      ② Epithelial 程序种子(Epithelial_program_* 子类)只填补
         尚未有种子的细胞（seed_source=epithelial_expression_program）；
      ③ 【先撤后补】：与 09 的 C 证据(基因+CNV 双证据)冲突的
         Epithelial 种子先整体撤销（training_label→Unknown,
         seed_source=withheld_lineage_status_conflict）——理由写在
         docstring 里：上皮谱系≠良性，不能拿谱系证据去否认肿瘤证据；
      ④ Neoplastic D 种子最后落位，且【只允许覆盖】Unknown 或
         Epithelial 的位置——绝不覆盖 C/I 这些"非肿瘤参考"种子
         （否则 05 的 CNV 参考集会被污染）。

输入 → 输出：
    --adata(取细胞轴) + 各种子 CSV + 09 证据表
    → combined_initial_seeds_no_official_labels.csv
    → 各来源计数与冲突审计 CSV。供 11 的 --seed-file 消费。

与其他模块的连接：
    reliability.data_contracts.aligned_cell_table —— 严格按细胞 ID 对齐
    （多一行少一行都报错）；boolean_flags —— 布尔列解析。
    withhold_ambiguous_epithelial_seeds —— 上面第③步的实现。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from reliability.data_contracts import aligned_cell_table, boolean_flags


EPITHELIAL_PREFIX = "Epithelial_program_"


def withhold_ambiguous_epithelial_seeds(result: pd.DataFrame, evidence: pd.DataFrame) -> int:
    """【中文】撤销与肿瘤 C 证据冲突的 Epithelial 种子（main 第③步调用）。

    做法：用 aligned_cell_table 把 09 证据表按 cell_id 对齐到 result 的
    行序；取 variant_C_gene_plus_cnv（基因+CNV 双证据）为 True 的细胞；
    找出其中 training_label 是 "Epithelial" 或 "Epithelial_program_*" 且
    目前是种子的细胞 → 撤销种子（is_initial_seed=False, 标签改 Unknown,
    seed_source 标记 withheld_lineage_status_conflict 留痕）。
    参数: result(合并中的种子表,带 cell_id 索引), evidence(09 证据表)
    返回: 被撤销的细胞数（写进审计 CSV）。
    """
    aligned = aligned_cell_table(evidence, result.index)
    tumor_support = boolean_flags(aligned["variant_C_gene_plus_cnv"])
    epithelial = result["training_label"].eq("Epithelial") | result["training_label"].str.startswith(EPITHELIAL_PREFIX)
    conflicting = epithelial & tumor_support & result["is_initial_seed"]
    result.loc[conflicting, "is_initial_seed"] = False
    result.loc[conflicting, "training_label"] = "Unknown"
    result.loc[conflicting, "seed_source"] = "withheld_lineage_status_conflict"
    return int(conflicting.sum())


def main() -> None:
    """【中文】主流程：见文件头【中文详解】的 ①②③④ 合并顺序。
    参数:
      --adata                    表达 h5ad（只读 obs_names 当细胞轴, backed 模式）
      --tiered-seeds             03 的 tiered 种子表（C/I，可选 E）
      --epithelial-program-seeds 08 的上皮程序种子表（与 --use-tiered-epithelial
                                 二选一，同时给或都不给都报错）
      --use-tiered-epithelial    改用 03 表里通用的 Epithelial 核心种子
      --neoplastic-seeds         07 的 Neoplastic 种子表
      --neoplastic-evidence      09 的证据表（供 C 冲突撤销）
      --out-dir                  输出目录
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--tiered-seeds", type=Path, required=True)
    parser.add_argument("--epithelial-program-seeds", type=Path)
    parser.add_argument(
        "--use-tiered-epithelial",
        action="store_true",
        help="Use the existing generic Epithelial core cells from --tiered-seeds instead of breast program seeds.",
    )
    parser.add_argument("--neoplastic-seeds", type=Path, required=True)
    parser.add_argument("--neoplastic-evidence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.use_tiered_epithelial and args.epithelial_program_seeds:
        raise ValueError("Choose either --use-tiered-epithelial or --epithelial-program-seeds, not both.")
    if not args.use_tiered_epithelial and not args.epithelial_program_seeds:
        raise ValueError("Provide --epithelial-program-seeds unless --use-tiered-epithelial is selected.")

    adata = sc.read_h5ad(args.adata, backed="r")
    cells = adata.obs_names.astype(str)
    adata.file.close()
    if not cells.is_unique:
        raise ValueError("Analysis cell IDs must be unique")
    result = pd.DataFrame(
        {
            "cell_id": cells,
            "is_initial_seed": False,
            "training_label": "Unknown",
            "seed_source": "",
        }
    ).set_index("cell_id")

    # ---- ①先落位 C/I(+E) tiered 核心种子。
    # usecols 只读 --tiered-seeds 里的指定标签；tiered_labels 决定是否
    # 收入通用 Epithelial（--use-tiered-epithelial 时追加）。
    tiered = aligned_cell_table(pd.read_csv(args.tiered_seeds, dtype={"cell_id": str}), cells)
    tiered = tiered[boolean_flags(tiered["is_initial_seed"])]
    tiered_labels = ["Connective_tissue", "Inflammatory"]
    if args.use_tiered_epithelial:
        tiered_labels.append("Epithelial")
    tiered = tiered[tiered["refined_pseudo_label"].isin(tiered_labels)]
    result.loc[tiered.index, "is_initial_seed"] = True
    result.loc[tiered.index, "training_label"] = tiered["refined_pseudo_label"]
    result.loc[tiered.index, "seed_source"] = "reliable_cluster_core"

    # ---- ②Epithelial 程序种子：只补"还没有种子"的空位
    # （nonconflicting_epithelial = 程序种子 - 已是种子的位置）。
    epithelial = pd.DataFrame()
    nonconflicting_epithelial = pd.Index([], dtype=object)
    if args.epithelial_program_seeds:
        epithelial = aligned_cell_table(pd.read_csv(args.epithelial_program_seeds, dtype={"cell_id": str}), cells)
        epithelial = epithelial[boolean_flags(epithelial["is_initial_seed"])]
        epithelial = epithelial[
            epithelial["training_label"].astype(str).str.startswith(EPITHELIAL_PREFIX)
        ]
        nonconflicting_epithelial = epithelial.index[~epithelial.index.isin(result.index[result["is_initial_seed"]])]
        result.loc[nonconflicting_epithelial, "is_initial_seed"] = True
        result.loc[nonconflicting_epithelial, "training_label"] = epithelial.loc[
            nonconflicting_epithelial, "training_label"
        ]
        result.loc[nonconflicting_epithelial, "seed_source"] = "epithelial_expression_program"

    # ---- ③先撤：撤销与 C 证据冲突的 Epithelial 种子（见函数中文说明）。
    withheld_epithelial = withhold_ambiguous_epithelial_seeds(
        result, pd.read_csv(args.neoplastic_evidence, dtype={"cell_id": str})
    )
    # ---- ④后补：Neoplastic D 种子只允许落在 Unknown 或 Epithelial 位置
    # ——绝不覆盖 C/I 参考种子（保护 05 的 CNV 参考集不被肿瘤细胞污染）。
    neoplastic = aligned_cell_table(pd.read_csv(args.neoplastic_seeds, dtype={"cell_id": str}), cells)
    neoplastic = neoplastic[boolean_flags(neoplastic["is_neoplastic_seed"])]
    # D may override an epithelial lineage seed, not a non-epithelial reference.
    allowed = result["training_label"].isin(["Unknown", "Epithelial"])
    nonconflicting_neoplastic = neoplastic.index[neoplastic.index.isin(result.index[allowed])]
    result.loc[nonconflicting_neoplastic, "is_initial_seed"] = True
    result.loc[nonconflicting_neoplastic, "training_label"] = "Neoplastic"
    result.loc[nonconflicting_neoplastic, "seed_source"] = "multievidence_neoplastic"

    # ---- 收尾：最终完整性校验（细胞轴顺序不能丢）→ 写合并种子表
    # （11 的输入）、按 (标签,来源) 的计数表、合并审计表。
    if not result.index.equals(cells):
        raise RuntimeError("Seed table lost the analysis AnnData cell order.")

    output = result.reset_index()
    output.to_csv(args.out_dir / "combined_initial_seeds_no_official_labels.csv", index=False)
    summary = (
        output[output["is_initial_seed"]]
        .groupby(["training_label", "seed_source"])
        .size()
        .rename("cell_count")
        .reset_index()
    )
    summary.to_csv(args.out_dir / "combined_initial_seed_counts_no_official_labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "official_labels_policy": "not_loaded",
                "tiered_core_candidates": len(tiered),
                "epithelial_program_candidates": len(epithelial),
                "epithelial_selection_mode": "tiered_cluster_core" if args.use_tiered_epithelial else "epithelial_expression_program",
                "neoplastic_multievidence_candidates": len(neoplastic),
                "epithelial_C_conflicts_withheld_before_D": withheld_epithelial,
                "epithelial_conflicts_excluded": len(epithelial) - len(nonconflicting_epithelial),
                "neoplastic_conflicts_excluded": len(neoplastic) - len(nonconflicting_neoplastic),
            }
        ]
    ).to_csv(args.out_dir / "seed_merge_audit_no_official_labels.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
