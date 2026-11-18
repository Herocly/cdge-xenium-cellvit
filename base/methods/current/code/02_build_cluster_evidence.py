#!/usr/bin/env python3
"""Build the frozen cluster-evidence table without loading ground truth.

The marker-voting calls intentionally come from the original CDGE code.  This
script adds only the project reliability diagnostics and never opens an
annotation file.

【中文详解 —— 流水线第 2 步：构建"冻结"的 cluster 级证据表】
====================================================================
流水线位置：01(预处理) → 【02 本脚本】 → 03(选种子) / 09(cluster 分查询) / 11(gate)

做什么：
    对 01 输出的每个 Leiden cluster，用 marker 基因投票 + 结构统计，
    产出一张"每个 cluster 一行"的证据表，回答两个问题：
      ① 这个 cluster 最像哪类细胞（coarse 标签：Connective_tissue /
         Inflammatory / Epithelial / Others）？
      ② 这个判断有多可靠（可靠性分，供 03 选种子、11 做 gate）？

逐步流程（main）：
    1. load_original_voting_functions：从 CDGE 原仓库逐字加载 marker 投票
       函数（保证投票方程与原方法完全一致）。
    2. 读 01 的 h5ad，按 obs["leiden_res_2.00"] 分 cluster，Wilcoxon 秩和
       检验给每个 cluster 排 marker 基因，取前 --top-n(20) 个存 CSV。
    3. load_databases_compat：读 PanglaoDB / CellMarker 两个 marker 数据库
       （可选 --tissue 组织过滤），经 Cell Ontology 把数据库的细胞类型名
       映射到核心类别（reliability/ontology_tree.py），另外准备
       neoplastic_genes.csv 肿瘤基因集和 CellMarker 癌症基因集。
    4. build_gene_votes（CDGE 原函数）：把数据库整理成
       {基因: 各类别票数向量} 的"基因投票表"。
    5. 逐 cluster 循环：classify_cluster 用 top10 marker 加权投票得
       pseudo_label；check_neoplastic / check_cancer_markers 判断肿瘤基因
       提示；同时计算 4 个结构指标（silhouette 近似、空间图邻居一致性、
       表达 cohesion、簇内 dispersion）和 dip 多模态检验。
    6. 所有指标转百分位后等权组合：
       structural_reliability_score（结构 5 项均值）、
       annotation_confidence_score（投票 3 项均值）、
       reliability_score（两者均值）、reliability_band（结构分三档）。
    7. 输出 cluster_reliability_no_official_labels.csv + 运行配置 JSON。

输入 → 输出：
    --adata (01 的 h5ad) + marker 数据库 + CL 本体快照
    → cluster 证据表 CSV（03 直接消费；09/11 按 cluster 号查行）。

调用的 reliability 模块：
    common.percentile_score / pseudo_to_core
    expression_space.expression_features / centroid_silhouette / cluster_multimodality_dip
    marker_voting_metrics.vote_metrics
    ontology_tree.CellOntologyMapper / map_marker_databases / write_audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.common import percentile_score, pseudo_to_core  # noqa: E402
from reliability.expression_space import centroid_silhouette, cluster_multimodality_dip, expression_features  # noqa: E402
from reliability.marker_voting_metrics import vote_metrics  # noqa: E402
from reliability.ontology_tree import (  # noqa: E402
    CellOntologyMapper, DEFAULT_POLICY, DEFAULT_SNAPSHOT,
    map_marker_databases, write_audit,
)


def parse_args() -> argparse.Namespace:
    """【中文】命令行参数。
    --adata            01 输出的预处理 h5ad（必须含 leiden cluster 列）
    --cdge-root        CDGE 原仓库根目录（加载其原版投票函数）
    --db-dir           marker 数据库目录（PanglaoDB/CellMarker/neoplastic 基因）
    --ontology         Cell Ontology 快照(.obo)；--ontology-policy 四类映射策略 JSON
    --tissue           可选组织过滤（默认不过滤，与 CDGE 原默认一致）
    --groupby          用哪个 cluster 列，默认 "leiden_res_2.00"（01 生成）
    --top-n            每个 cluster 保留的 top marker 基因数（默认 20）
    --database-policy  original=CDGE 原版投票；
                       human_unique_source_balanced=按来源均衡的消融变体
    --out-dir          输出目录（证据表 CSV、审计文件等）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--cdge-root", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, default=ROOT / "resources" / "marker_databases")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--ontology-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--tissue",
        default=None,
        help=(
            "Optional CDGE tissue filter. By default no tissue filter is applied, "
            "matching the original CDGE database-loader default."
        ),
    )
    parser.add_argument("--groupby", default="leiden_res_2.00")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--database-policy", choices=("original", "human_unique_source_balanced"), default="original")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def neighbour_consistency(adata: sc.AnnData, clusters: np.ndarray) -> np.ndarray:
    """【中文】逐细胞的"表达图邻居一致性"（结构可靠性成分之一）。
    对每个细胞，在 01 建的 kNN 连接图(obsp["connectivities"])里找它的邻居，
    算"邻居中与自己同 cluster 的比例"。某 cluster 平均值高 = 该 cluster 在
    表达流形上是紧致连通的一块，而不是散落在图各处的杂烩。
    参数: adata(含邻居图), clusters(每细胞的 cluster 标签数组)
    返回: 长度 = 细胞数 的一维 float32 数组。
    被 main 调用；结果按 cluster 取均值写进证据表 neighbor_consistency 列。
    """
    graph = adata.obsp["connectivities"].tocsr()
    values = np.zeros(adata.n_obs, dtype=np.float32)
    for index in range(adata.n_obs):
        neighbours = graph.indices[graph.indptr[index] : graph.indptr[index + 1]]
        values[index] = np.mean(clusters[neighbours] == clusters[index]) if len(neighbours) else 0.0
    return values


def load_original_voting_functions(cdge_root: Path) -> dict[str, object]:
    """Load CDGE's marker-voting definitions without its optional image stack.

    The source module imports a geometry helper used only by unrelated image
    routines.  The replacements below remove that import line only; the
    database filters and vote equations are executed verbatim from the
    original ``bottom_up_utils.py`` source.

    【中文】加载 CDGE 原版的 marker 投票函数（不复制代码、直接执行源码）。
    ------------------------------------------------------------------
    做法：读 CDGE 仓库的 preprocessing_hest/preproce_utils/bottom_up_utils.py
    源文件文本，删掉唯一一行与本任务无关的 import（图片几何工具），然后
    exec 编译执行 —— 这样数据库过滤和投票方程 100% 是原版行为，
    避免手工移植引入偏差。
    返回: 一个 namespace 字典，含 build_gene_votes / check_cancer_markers /
    check_neoplastic / classify_cluster / VOTE_CATEGORIES / 组织过滤函数等。
    被 main 和 load_databases_compat、source_balanced_gene_votes 使用。
    """
    source_file = cdge_root / "preprocessing_hest" / "preproce_utils" / "bottom_up_utils.py"
    source = source_file.read_text(encoding="utf-8")
    source = source.replace("from .common import LABEL_MAP, map_category_to_label", "")
    namespace: dict[str, object] = {"__file__": str(source_file), "__name__": "cdge_voting_compat"}
    exec(compile(source, str(source_file), "exec"), namespace)
    return namespace


def load_databases_compat(
    db_dir: Path,
    voting: dict[str, object],
    tissue: str | None,
    ontology_path: Path,
    policy_path: Path = DEFAULT_POLICY,
) -> tuple[object, object, object, object, set[str], set[str], dict[str, object]]:
    """Load marker databases with an optional, explicit tissue restriction.

    【中文】读入并清洗三个 marker 基因库（main 第 3 步调用）。
    ------------------------------------------------------------------
    参数:
        db_dir         数据库目录；--ontology/--ontology-policy 是 CL 本体
                       快照与四类映射策略；tissue 是可选的组织过滤词
    返回七元组:
        panglao        PanglaoDB 过滤后的 DataFrame
        cellmarker     CellMarker "Normal cell" 行（Cancer 行单独返回）
        panglao_map / cellmarker_map
                       两库经 Cell Ontology 映射出的 {细胞类型: 四类 one-hot}
                       投票矩阵（map_marker_databases 生成）
        neoplastic_set 肿瘤相关基因集合（neoplastic_genes.csv，大写）
        cancer_set     CellMarker human Cancer cell 的基因符号集合
        audit          数据库规模/物种/重复记录 + 本体审计字典
    过滤链：tissue 过滤(可选,复用 CDGE 原版 _filter_* 函数)
           → CellMarker 取 species=human 的行 → 本体映射。
    """
    normalized_tissue = None if tissue is None or tissue.lower() in {"all", "none"} else tissue
    panglao = pd.read_csv(db_dir / "marker_gene_list.tsv", sep="\t")
    cellmarker_all = pd.read_csv(
        db_dir / "database_gene_functions" / "Cell_marker_All.csv",
        low_memory=False,
    )
    cellmarker = cellmarker_all.loc[cellmarker_all["cell_type"].eq("Normal cell")].copy()
    cancer = cellmarker_all.loc[cellmarker_all["cell_type"].eq("Cancer cell")].copy()

    if normalized_tissue is not None:
        supported = set(voting["PANGLAO_ORGAN_MAP"]) | set(voting["CELLMARKER_TISSUE_MAP"])
        if normalized_tissue not in supported:
            raise ValueError(
                f"Unknown CDGE tissue filter {normalized_tissue!r}; choose one of {sorted(supported)} "
                "or omit --tissue for the unfiltered default."
            )
        panglao = voting["_filter_panglao"](panglao, normalized_tissue)
        cellmarker = voting["_filter_cellmarker"](cellmarker, normalized_tissue)
        cancer = voting["_filter_cellmarker"](cancer, normalized_tissue)

    mapper = CellOntologyMapper.from_files(ontology_path, policy_path, organ=normalized_tissue)
    panglao_map, cellmarker_map, crosswalk, ontology_audit = map_marker_databases(
        panglao, cellmarker, mapper
    )
    neoplastic = pd.read_csv(db_dir / "neoplastic_genes.csv")
    neoplastic_set = {str(gene).strip().upper() for gene in neoplastic["gene"].tolist()}
    cancer_set: set[str] = set()
    human_cancer = cancer.loc[cancer["species"].fillna("").str.lower().eq("human")]
    for symbol in human_cancer["Symbol"].dropna():
        cancer_set.update(voting["_split_cellmarker_symbols"](symbol))
    audit = {
        "tissue_filter": normalized_tissue,
        "database_scope": "all_tissues" if normalized_tissue is None else "tissue_filtered",
        "panglao_rows": int(len(panglao)),
        "cellmarker_normal_rows": int(len(cellmarker)),
        "cellmarker_cancer_rows": int(len(cancer)),
        "panglao_species_rows": panglao["species"].fillna("missing").value_counts().to_dict(),
        "cellmarker_species_rows": cellmarker["species"].fillna("missing").value_counts().to_dict(),
        "panglao_duplicate_gene_type_records": int(panglao.duplicated(["official gene symbol", "cell type"]).sum()),
        "cellmarker_duplicate_symbol_type_records": int(cellmarker.duplicated(["Symbol", "cell_name"]).sum()),
        "ontology_file": str(ontology_path),
        "ontology_version": ontology_audit["policy_version"],
        "ontology_audit": ontology_audit,
    }
    audit["ontology_crosswalk"] = crosswalk
    return panglao, cellmarker, panglao_map, cellmarker_map, neoplastic_set, cancer_set, audit


def source_balanced_gene_votes(panglao, cellmarker, panglao_map, cellmarker_map, voting):
    """One unit per covered source/gene; repeated records cannot add evidence.

    【中文】--database-policy=human_unique_source_balanced 时的替代投票表
    （消融实验用）。与原版 build_gene_votes 的区别：
      ① 只保留 human 记录；
      ② Panglao 与 CellMarker 各自构建投票表，再各自做 L1 归一化
        （基因在该库中覆盖的类别上均匀分布 1 份票）；
      ③ 两库相加 —— 这样"一个库里重复收录同一基因多次"不再产生超额票，
        两库的话语权相等（原版里记录多的库天然票多）。
    返回: (合并投票表, Panglao 单库表, CellMarker 单库表)。
    被 main 在 database_policy 非 "original" 时调用。
    """
    panglao = panglao.loc[panglao["species"].fillna("").str.split().map(lambda values: "Hs" in values)]
    cellmarker = cellmarker.loc[cellmarker["species"].fillna("").str.lower().eq("human")]
    sources = [voting["build_gene_votes"](panglao, panglao_map),
               voting["build_gene_votes"](panglao.iloc[:0], panglao_map, cellmarker, cellmarker_map)]
    normalized = []
    for source in sources:
        result = {}
        for gene, vector in source.items():
            present = (vector > 0).astype(float)
            if present.sum():
                result[gene] = present / present.sum()
        normalized.append(result)
    combined = {}
    for gene in set(normalized[0]) | set(normalized[1]):
        combined[gene] = sum((source[gene] for source in normalized if gene in source),
                             np.zeros(len(voting["VOTE_CATEGORIES"])))
    return combined, *normalized


def classify_normalized_votes(genes, gene_votes, categories, top_n=10):
    """【中文】source_balanced 模式下的 classify_cluster 替身。
    与原版 classify_cluster 同样的加权投票逻辑：前 top_n 个 marker 按
    名次递减加权（第 1 名权重 top_n，第 2 名 top_n-1 …）累加各类别票。
    参数: genes(cluster 的排好序 marker 基因), gene_votes(投票表),
          categories(类别名列表), top_n(参与投票的基因数,默认 10)
    返回: {predicted_category 赢家类别, confidence 赢家票占比,
           total_votes 各类别总票}；总票为 0 时返回 Unknown/0.0。
    被 main 以 lambda 形式注入 classify_cluster 使用。
    """
    total = np.zeros(len(categories), dtype=float)
    for rank, gene in enumerate(genes[:top_n]):
        total += gene_votes.get(gene, np.zeros(len(categories))) * (top_n - rank)
    if not total.sum():
        return {"predicted_category": "Unknown", "confidence": 0.0, "total_votes": {}}
    return {"predicted_category": categories[int(total.argmax())],
            "confidence": float(total.max() / total.sum()),
            "total_votes": dict(zip(categories, total.tolist()))}


def main() -> None:
    """【中文】主流程：见文件头【中文详解】的 7 个步骤。"""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # 步骤 1：加载 CDGE 原版投票函数，并解包出本脚本要用的 4 个。
    voting = load_original_voting_functions(args.cdge_root)
    build_gene_votes = voting["build_gene_votes"]
    check_cancer_markers = voting["check_cancer_markers"]
    check_neoplastic = voting["check_neoplastic"]
    classify_cluster = voting["classify_cluster"]
    # 步骤 2：读 01 的 h5ad，校验 cluster 列存在；Wilcoxon 检验对每个
    # cluster 排基因（use_raw=False → 用缩放后的 X，秩检验对列偏移不变）。
    adata = sc.read_h5ad(args.adata)
    if args.groupby not in adata.obs:
        raise KeyError(f"Missing {args.groupby}; run label-free preprocessing first.")
    clusters = adata.obs[args.groupby].astype(str).to_numpy()
    print(f"Ranking marker genes for {adata.n_obs} cells and {pd.unique(clusters).size} clusters", flush=True)
    sc.tl.rank_genes_groups(adata, groupby=args.groupby, method="wilcoxon", use_raw=False)
    ranked = sc.get.rank_genes_groups_df(adata, group=None)
    ranked = ranked.sort_values(["group", "scores"], ascending=[True, False])
    top_markers = ranked.groupby("group", observed=True).head(args.top_n).reset_index(drop=True)
    top_markers.to_csv(args.out_dir / "cluster_top_marker_genes.csv", index=False)

    # 步骤 3-4：读 marker 数据库并构建基因投票表（原版 + 两单库版）。
    db = load_databases_compat(args.db_dir, voting, args.tissue, args.ontology, args.ontology_policy)
    panglao_db, cellmarker_db, panglao_map, cellmarker_map, neoplastic_set, cancer_set, database_audit = db
    crosswalk = database_audit.pop("ontology_crosswalk")
    write_audit(args.out_dir, crosswalk, database_audit["ontology_audit"])
    combined_votes = build_gene_votes(panglao_db, panglao_map, cellmarker_db, cellmarker_map)
    panglao_votes = build_gene_votes(panglao_db, panglao_map)
    cellmarker_votes = build_gene_votes(panglao_db.iloc[:0], panglao_map, cellmarker_db, cellmarker_map)
    if args.database_policy == "human_unique_source_balanced":
        combined_votes, panglao_votes, cellmarker_votes = source_balanced_gene_votes(
            panglao_db, cellmarker_db, panglao_map, cellmarker_map, voting)
        classify_cluster = lambda genes, votes, top_n: classify_normalized_votes(genes, votes, voting["VOTE_CATEGORIES"], top_n)

    # 步骤 5a：结构指标只在开头算一次（逐细胞的 silhouette 近似、邻居一致性）。
    features = expression_features(adata, 30, "X_pca")
    silhouette = centroid_silhouette(features, clusters)
    neighbours = neighbour_consistency(adata, clusters)
    rows: list[dict[str, object]] = []
    unique_clusters = sorted(pd.unique(clusters), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    # 步骤 5b：逐 cluster 循环 —— 先 marker 投票拿 coarse 标签，再算结构统计。
    for number, cluster in enumerate(unique_clusters, start=1):
        mask = clusters == cluster
        positions = np.flatnonzero(mask)
        genes = top_markers.loc[top_markers["group"].astype(str).eq(cluster), "names"].astype(str).tolist()
        # classify_cluster：top10 marker 加权投票 → 主标签(如 Connective_tissue)
        result = classify_cluster(genes, combined_votes, top_n=10)
        # Tumor hints must not overwrite the normal-lineage vote or its support.
        # 肿瘤相关检查只产"提示位"（hint 布尔值），不覆盖上面的正常谱系标签。
        neo = check_neoplastic(genes, neoplastic_set, top_n=20, threshold=.25)
        cancer = check_cancer_markers(genes, cancer_set, top_n=20, threshold=.25)
        # 单库各自再投一次 → 用于"独立来源是否一致"的佐证列。
        source_p = classify_cluster(genes, panglao_votes, top_n=10)
        source_c = classify_cluster(genes, cellmarker_votes, top_n=10)
        # 表达 cohesion：cluster 内每个细胞对簇中心的余弦相似度均值。
        cluster_features = features[positions]
        centre = cluster_features.mean(axis=0)
        distances = np.linalg.norm(cluster_features - centre, axis=1)
        norms = np.linalg.norm(cluster_features, axis=1) * max(np.linalg.norm(centre), 1e-8)
        cohesion = np.divide(cluster_features @ centre, norms, out=np.zeros_like(norms), where=norms > 0)
        # dip 检验：cluster 内部是否有多个表达模式（多模态 → 可能混了亚群）。
        dip, dip_p, dip_components = cluster_multimodality_dip(cluster_features)
        # pseudo_to_core：把投票标签归并到核心类别体系（reliability/common.py）。
        pseudo = pseudo_to_core(result["predicted_category"])
        panglao = pseudo_to_core(source_p["predicted_category"])
        cellmarker = pseudo_to_core(source_c["predicted_category"])
        row: dict[str, object] = {
            "cluster": cluster, "n_cells": len(positions), "pseudo_label": pseudo,
            "tumor_gene_marker_hint": neo["is_neoplastic"], "organ_cancer_marker_hint": cancer["is_cancer"],
            "panglao_label": panglao, "panglao_confidence": source_p.get("confidence", 0.0),
            "cellmarker_label": cellmarker, "cellmarker_confidence": source_c.get("confidence", 0.0),
            "independent_source_agreement": panglao == cellmarker and panglao not in {"Unknown", "Others"},
            "combined_panglao_agreement": pseudo == panglao and pseudo not in {"Unknown", "Others"},
            "combined_cellmarker_agreement": pseudo == cellmarker and pseudo not in {"Unknown", "Others"},
            "mean_silhouette": float(silhouette[positions].mean()),
            "neighbor_consistency": float(neighbours[positions].mean()),
            "expression_cohesion": float(cohesion.mean()),
            "within_dispersion": float(distances.mean()),
            "dip_statistic": dip, "dip_p_value": dip_p, "dip_tested_components": dip_components,
            "multimodal_candidate": bool(dip_p < .05), "top_marker_genes": ", ".join(genes),
            "official_labels_policy": "not_loaded",
        }
        row.update(vote_metrics(result, genes, combined_votes))
        row["voting_top10_marker_coverage"] = sum(g in combined_votes for g in genes[:10]) / max(1, len(genes[:10]))
        rows.append(row)
        if number % 20 == 0 or number == len(unique_clusters):
            print(f"processed {number}/{len(unique_clusters)} clusters", flush=True)

    # 步骤 6：把所有指标转成 0~1 百分位（消除量纲）后等权组合成分数。
    # structural 5 项：silhouette↑ 邻居一致↑ cohesion↑ dispersion↓ dip 统计量↓
    # annotation 3 项：投票 confidence↑ margin↑ marker 覆盖率↑
    # source_support_count：两库(与合并票)独立一致的支持源个数(0~2)
    # reliability_band：结构分三档 reliable / uncertain / mixed_candidate
    table = pd.DataFrame(rows)
    structural = pd.DataFrame({
        "silhouette": percentile_score(table["mean_silhouette"], True),
        "neighbor": percentile_score(table["neighbor_consistency"], True),
        "cohesion": percentile_score(table["expression_cohesion"], True),
        "dispersion": percentile_score(table["within_dispersion"], False),
        "inverse_dip": percentile_score(table["dip_statistic"], False),
    })
    annotation = pd.DataFrame({
        "confidence": percentile_score(table["vote_confidence"], True),
        "margin": percentile_score(table["vote_margin"], True),
        "coverage": percentile_score(table["marker_coverage"], True),
    })
    table["structural_reliability_score"] = structural.mean(axis=1)
    table["annotation_confidence_score"] = annotation.mean(axis=1)
    table["source_support_count"] = (
        table["combined_panglao_agreement"].astype(int) + table["combined_cellmarker_agreement"].astype(int)
    )
    table["reliability_score"] = table[["structural_reliability_score", "annotation_confidence_score"]].mean(axis=1)
    table["reliability_band"] = pd.cut(
        table["structural_reliability_score"], [-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["mixed_candidate", "uncertain", "reliable"],
    ).astype(str)
    # 步骤 7：排序后写主证据表 + 运行配置 JSON。这张表是 03 选种子和
    # 11 gate 的唯一 cluster 级输入（"冻结"：一次生成，全程不再改动）。
    table = table.sort_values("reliability_score", ascending=False)
    table.to_csv(args.out_dir / "cluster_reliability_no_official_labels.csv", index=False)
    config = {
        "official_labels_policy": "not_loaded",
        **database_audit,
        "marker_voting": "original_CDGE_bottom_up_utils" if args.database_policy == "original" else "human_unique_per_gene_class_source_balanced_rank_weighted",
        "database_policy": args.database_policy,
        "evidence_interpretation": "database agreement is corroboration, not statistical independence",
        "structural_components": list(structural.columns),
        "annotation_components": list(annotation.columns),
        "cluster_count": int(table.shape[0]),
    }
    (args.out_dir / "run_config_no_official_labels.json").write_text(json.dumps(config, indent=2))
    print(table[["cluster", "n_cells", "pseudo_label", "reliability_score"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
