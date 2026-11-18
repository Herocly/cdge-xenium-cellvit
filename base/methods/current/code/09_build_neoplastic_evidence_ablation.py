"""Build label-free Neoplastic evidence and A/B/C/D candidate variants.

CNV is deliberately one feature among several.  No official annotations are
loaded here; the output consists of evidence columns and fixed candidate rules
that a separate evaluation script may assess afterwards.

【中文详解 —— 流水线第 9 步：逐细胞肿瘤证据 + A/B/C/D 四档消融】
====================================================================
流水线位置：05(CNV 剖面) + 06(CNV cluster 审计) + 02(cluster 证据表)
            → 【09 本脚本】 → 07(导出 D 种子) / 10、11(C 证据做 gate)

做什么：
    把三种互相独立的证据在【逐细胞】粒度对齐并组合：
      ① 基因证据：肿瘤基因集(neoplastic_genes.csv)在完整 panel raw counts
         上的平均表达分（top 10% 分位为阳性）；
      ② CNV 证据：05 输出的 X_cnv 剖面相对正常参考的偏离幅值
         （超过参考细胞 99 分位为"CNV 异常"）+ 06 的 cluster 级
         空间一致性；剖面一致性仍记录作审计，不参与筛选；
      ③ 结构证据：细胞所属表达 cluster 的 structural_reliability_score；
      ④ 排除项：03 选出的 Inflammatory/Connective 参考细胞、以及被
         02 强证据标为 C/I 的细胞不允许再当肿瘤。

    四个候选变体是【在看任何 ground truth 之前】就固定的递进规则：
      A = 只有基因证据           （并排除强 normal 证据与参考细胞）
      B = 只有 CNV 证据          （幅值异常）
      C = A 且 B                 （两种独立计算相互印证；不假设统计独立）
      D = C 且 空间一致 ≥ 0.50 且 结构分 ≥ 0.50   （最保守，07 用它当种子）

输入 → 输出：
    --cnv-adata(05 的 h5ad) + --expression-adata(01 的 h5ad)
    + --cluster-table(02 证据表) + --cnv-audit(06 审计表)
    → neoplastic_evidence_table_*.csv（每细胞一行，含全部证据列和
      variant_A/B/C/D 布尔列）、消融计数 CSV、阈值记录 JSON。

与其他模块的连接：
    reliability.cluster_evidence_gate.strong_cluster_label_map ——
    取 02 里"强证据"cluster 的标签，用于排除强 normal 证据细胞；
    reliability.data_contracts.validate_counts / positive_quantile_evidence ——
    counts 校验与"正数分位"二值化工具。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.cluster_evidence_gate import strong_cluster_label_map  # noqa: E402
from reliability.data_contracts import validate_counts, positive_quantile_evidence  # noqa: E402


def to_dense(matrix: object) -> np.ndarray:
    """Convert a sparse AnnData matrix only when the 20k local sample is read.

    【中文】把稀疏/稠密矩阵统一转成 NumPy 稠密数组（X_cnv 每细胞一条
    染色体窗口剖面，2 万细胞规模可承受）。被 main 调用一次。"""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)


def mean_abs_cnv(profile: np.ndarray) -> np.ndarray:
    """One CNV-deviation value per cell, relative to normal references.

    【中文】每个细胞一个"CNV 偏离幅值" = 该细胞全部窗口 CNV 值绝对值的
    均值。值越大 = 相对正常参考的拷贝数变异越剧烈（肿瘤常见特征）。
    输入 profile 形状 (细胞数 × 窗口数)，返回 (细胞数,)。"""
    return np.mean(np.abs(profile), axis=1)


def marker_gene_score(adata: sc.AnnData, genes: set[str]) -> tuple[np.ndarray, list[str]]:
    """Use full-panel, unsmoothed raw counts, never CNV's coordinate subset.

    【中文】逐细胞"肿瘤基因表达分"（证据①的实现）。
    ------------------------------------------------------------------
    做法：取肿瘤基因集与 panel 的交集基因，用【raw counts 层】（不是
    CNV 里被平滑/开窗过的子集——docstring 强调这点）做
    normalize_total(10000) → log1p → 对交集基因取平均。
    参数: adata(01 的表达 h5ad，须有 layers["counts"])，
          genes(肿瘤基因符号集合，大写)
    返回: (每细胞肿瘤表达分, 交集基因名列表)
    严格性：交集为空、缺 counts 层、counts 非法、存在全 0 细胞都会报错。
    被 main 调用；分数再经 positive_quantile_evidence 二值化。
    """
    overlap = sorted(genes & set(adata.var_names.astype(str)))
    if not overlap:
        raise RuntimeError("No neoplastic marker genes overlap the Xenium panel.")
    if "counts" not in adata.layers:
        raise KeyError("Gene evidence requires an explicit raw counts layer")
    validate_counts(adata.layers["counts"])
    totals = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Gene evidence cannot normalize zero-count cells")
    matrix = sparse.csr_matrix(adata[:, overlap].layers["counts"], dtype=float)
    matrix = matrix.multiply((10_000 / totals)[:, None]).tocsr()
    np.log1p(matrix.data, out=matrix.data)
    return np.asarray(matrix.mean(axis=1)).ravel(), overlap


def main() -> None:
    """【中文】主流程：对齐两种数据的细胞轴 → 汇总三类证据 → 固定规则
    生成 A/B/C/D → 写证据表/消融计数/阈值 JSON（详见文件头中文详解）。
    参数说明（parse_args 内）:
      --cnv-adata            05 输出的 CNV h5ad（含 X_cnv、cnv cluster、参考类别）
      --expression-adata     01 输出的表达 h5ad（含 counts 层与 leiden cluster）
      --cluster-table        02 的 cluster 证据表（查强 normal 标签与结构分）
      --cnv-audit            06 的 CNV cluster 审计表（查一致性/空间一致性）
      --neoplastic-genes     肿瘤基因集 CSV
      --expression-cluster-column  表达 cluster 列名（默认 leiden_res_2.00）
      --cnv-cluster-column         CNV cluster 列名（默认 cnv_leiden_res0.2）
      --gene-evidence-quantile     基因证据阳性分位（默认 0.90）
      --cnv-reference-quantile     CNV 异常阈值分位（默认 0.99）
      --out-dir                    输出目录
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnv-adata", type=Path, required=True)
    parser.add_argument("--expression-adata", type=Path, required=True)
    parser.add_argument(
        "--cluster-table",
        type=Path,
        default=ROOT / "results/01_cluster_reliability/cluster_reliability.csv",
    )
    parser.add_argument(
        "--cnv-audit",
        type=Path,
        default=ROOT / "results/02_insitucnv/cnv_candidate_audit_label_free.csv",
    )
    parser.add_argument(
        "--neoplastic-genes",
        type=Path,
        default=ROOT / "resources/marker_databases/neoplastic_genes.csv",
    )
    parser.add_argument("--expression-cluster-column", default="leiden_res_2.00")
    parser.add_argument("--cnv-cluster-column", default="cnv_leiden_res0.2")
    parser.add_argument("--gene-evidence-quantile", type=float, default=0.90)
    parser.add_argument("--cnv-reference-quantile", type=float, default=0.99)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/07_neoplastic_evidence")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 第 1 段：严格对齐检查。CNV h5ad 与表达 h5ad 必须是同一批细胞、
    # 同一套表达 cluster（任何不一致直接报错，防止证据错位）。同时校验
    # 两张 cluster 级表(02/06)的 ID 唯一、能覆盖本数据出现的所有 cluster。
    adata = sc.read_h5ad(args.cnv_adata)
    expression = sc.read_h5ad(args.expression_adata)
    if not adata.obs_names.is_unique or not expression.obs_names.is_unique:
        raise ValueError("Expression and CNV cell IDs must be unique")
    if set(adata.obs_names) != set(expression.obs_names):
        raise ValueError("Expression and CNV evidence use different cell axes")
    expression = expression[adata.obs_names].copy()
    expression_clusters = adata.obs[args.expression_cluster_column].astype(str).to_numpy()
    if not np.array_equal(expression.obs[args.expression_cluster_column].astype(str), expression_clusters):
        raise ValueError("Expression and CNV expression-cluster assignments differ")
    cnv_clusters = adata.obs[args.cnv_cluster_column].astype(str).to_numpy()
    cluster_table = pd.read_csv(args.cluster_table, dtype={"cluster": str})
    cnv_audit = pd.read_csv(args.cnv_audit, dtype={"cnv_cluster": str})
    if not cluster_table["cluster"].is_unique or not cnv_audit["cnv_cluster"].is_unique:
        raise ValueError("Cluster audit IDs must be unique")
    if not set(expression_clusters).issubset(set(cluster_table["cluster"])) or not set(cnv_clusters).issubset(set(cnv_audit["cnv_cluster"])):
        raise ValueError("Cluster evidence is missing clusters from the analysis")

    # ---- 第 2 段：从 02 证据表取"强 normal 证据"与结构分，投影到细胞级。
    # strong_normal_labels = {cluster: coarse 标签}(仅强证据 cluster)；
    # has_strong_normal_evidence = 细胞所属 cluster 被强证据标为 C/I；
    # cell_structural_score = 细胞所属 cluster 的结构可靠性分（D 变体用）。
    strong_normal_labels = strong_cluster_label_map(cluster_table)
    normal_evidence = np.asarray(
        [strong_normal_labels.get(cluster, "") for cluster in expression_clusters], dtype=object
    )
    has_strong_normal_evidence = np.isin(normal_evidence, ["Connective_tissue", "Inflammatory"])
    structural_score = dict(
        zip(cluster_table["cluster"].astype(str), cluster_table["structural_reliability_score"])
    )
    cell_structural_score = np.asarray(
        [structural_score.get(cluster, np.nan) for cluster in expression_clusters], dtype=float
    )

    # ---- 第 3 段：CNV 证据。幅值超过"正常参考细胞 99 分位"记为 CNV 异常；
    # 再从 06 审计表按细胞所属 CNV cluster 查出剖面一致性与空间一致性。
    cnv_profile = to_dense(adata.obsm["X_cnv"])
    cnv_amplitude = mean_abs_cnv(cnv_profile)
    reference = adata.obs["cnv_reference_category"].astype(str).to_numpy() != "Observation"
    if not reference.any() or not np.isfinite(cnv_amplitude).all():
        raise ValueError("CNV evidence requires finite profiles and normal references")
    cnv_cutoff = float(np.quantile(cnv_amplitude[reference], args.cnv_reference_quantile))
    cnv_abnormal = cnv_amplitude > cnv_cutoff
    audit_by_cluster = cnv_audit.set_index("cnv_cluster")
    cell_cnv_coherence = np.asarray(
        [audit_by_cluster.loc[cluster, "cnv_profile_coherence"] for cluster in cnv_clusters], dtype=float
    )
    cell_spatial_consistency = np.asarray(
        [audit_by_cluster.loc[cluster, "spatial_neighbor_consistency"] for cluster in cnv_clusters], dtype=float
    )
    if not all(np.isfinite(values).all() for values in (cell_cnv_coherence, cell_spatial_consistency, cell_structural_score)):
        raise ValueError("CNV, spatial and structural support must be finite")

    # ---- 第 4 段：基因证据。肿瘤基因表达分 → 正分位(默认 top10%)二值化。
    genes = set(pd.read_csv(args.neoplastic_genes)["gene"].astype(str))
    tumor_score, gene_overlap = marker_gene_score(expression, genes)
    tumor_evidence, tumor_cutoff = positive_quantile_evidence(tumor_score, args.gene_evidence_quantile)

    # A/B/C/D are fixed before official-label evaluation.
    # ---- 第 5 段：四个候选变体（固定规则，先于任何 ground truth 评估）。
    # A: tumor-gene evidence only, excluding a strong normal explanation.
    #    A = 基因证据阳性 & 非强 C/I 证据 & 非 CNV 参考细胞
    variant_a = tumor_evidence & ~has_strong_normal_evidence & ~reference
    # B: CNV is a self-contained diagnostic only, not a final label.
    #    B = CNV 幅值异常；剖面一致性只作审计，不参与候选筛选。
    variant_b = cnv_abnormal
    # C: distinct gene and CNV computations agree; statistical independence is not assumed.
    #    C = A 且 B —— 基因与 CNV 两种独立计算相互印证（不假设统计独立）
    variant_c = variant_a & variant_b
    # D: require coherent spatial organization and a reasonably stable
    # expression cluster as additional safeguards.
    #    D = C + 空间组织连贯 + 表达 cluster 结构稳定（最保守，07 导出当种子）
    variant_d = (
        variant_c
        & (cell_spatial_consistency >= 0.50)
        & (cell_structural_score >= 0.50)
    )

    # ---- 第 6 段：写出三份产物 ——
    # ① 逐细胞证据表（所有原始证据列 + variant A/B/C/D 布尔列，
    #    07 的 --evidence-table 输入、11 的 --neoplastic-evidence 输入）；
    # ② 四个变体的候选细胞计数（消融对比用）；
    # ③ 全部阈值与基因交集的 JSON 记录（可复现性审计）。
    table = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "expression_cluster": expression_clusters,
            "cnv_cluster": cnv_clusters,
            "strong_normal_evidence_label": normal_evidence,
            "has_strong_normal_evidence": has_strong_normal_evidence,
            "epithelial_lineage_evidence": normal_evidence == "Epithelial",
            "is_cnv_reference": reference,
            "expression_cluster_structural_score": cell_structural_score,
            "cnv_amplitude": cnv_amplitude,
            "cnv_abnormal_relative_to_normal_reference": cnv_abnormal,
            "cnv_cluster_profile_coherence": cell_cnv_coherence,
            "cnv_cluster_spatial_consistency": cell_spatial_consistency,
            "tumor_gene_score": tumor_score,
            "tumor_gene_evidence": tumor_evidence,
            "variant_A_gene_only": variant_a,
            "variant_B_cnv_only": variant_b,
            "variant_C_gene_plus_cnv": variant_c,
            "variant_D_all_evidence": variant_d,
        }
    )
    table.to_csv(args.out_dir / "neoplastic_evidence_table_no_official_labels.csv", index=False)
    summary = pd.DataFrame(
        {
            "variant": ["A_gene_only", "B_cnv_only", "C_gene_plus_cnv", "D_all_evidence"],
            "candidate_cells": [int(variant_a.sum()), int(variant_b.sum()), int(variant_c.sum()), int(variant_d.sum())],
        }
    )
    summary.to_csv(args.out_dir / "ablation_candidate_counts_no_official_labels.csv", index=False)
    (args.out_dir / "thresholds_no_official_labels.json").write_text(
        pd.Series(
            {
                "official_labels_policy": "not_loaded",
                "tumor_gene_overlap_count": len(gene_overlap),
                "tumor_gene_overlap": gene_overlap,
                "gene_expression_source": str(args.expression_adata),
                "gene_expression_policy": "full_panel_raw_counts_normalize10000_log1p_no_neighbor_smoothing",
                "normal_exclusion_policy": "reference_cells_and_strong_connective_inflammatory_not_epithelial_lineage",
                "tumor_gene_evidence_quantile": args.gene_evidence_quantile,
                "tumor_gene_score_cutoff": tumor_cutoff,
                "cnv_reference_quantile": args.cnv_reference_quantile,
                "cnv_amplitude_cutoff": cnv_cutoff,
                "cnv_coherence_minimum": None,
                "spatial_consistency_minimum": 0.50,
                "structural_score_minimum": 0.50,
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
