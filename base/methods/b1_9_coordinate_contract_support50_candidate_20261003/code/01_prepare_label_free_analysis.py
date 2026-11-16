#!/usr/bin/env python3
"""Prepare a Xenium slide for label-free pseudo-label selection.

This script intentionally never reads annotation files.  It transforms the
gene-count matrix, computes a PCA expression representation, builds the
Leiden neighbourhood graph, and preserves spatial coordinates for later
evidence checks.

【中文详解 —— 流水线第 1 步：无标签预处理】
====================================================================
这个脚本在整个伪标签流水线中的位置：
    01(本脚本) → 02(cluster 证据) → 03(选种子) → 05(CNV) → ... → 11(迭代扩展)

做什么：
    读入一张 Xenium 切片的 h5ad（基因计数矩阵），做质检过滤和标准化，
    计算 PCA 表达表示和 Leiden 表达聚类，全部过程【绝不读取任何人工
    注释/ground truth】("label-free" 的含义)。输出一个"冻结"的预处理
    AnnData，供后续所有步骤共用。

逐步流程（main → prepare_label_free_analysis）：
    1. 读 h5ad，取出表达矩阵（默认 X，或 --expression-layer 指定的层），
       校验它是"非负整数 raw counts"（validate_counts，来自
       reliability/data_contracts.py），并把 raw counts 存进 layers["counts"]
       ——后续 05 的 CNV 推断必须用 raw counts，所以这里特意保留。
    2. 基因名统一大写；要求细胞 ID、基因名都唯一。
    3. QC：过滤掉 total counts < --min-counts(10) 或检测基因数 <
       --min-genes(5) 的低质量细胞。
    4. 标准化三连：normalize_total(目标深度 10000) → log1p →
       z-score 缩放(scale, 上限 clip=10)。
    5. PCA：取 min(--n-pcs=30, 基因数-1, 细胞数-1) 个主成分，存 obsm["X_pca"]
       ——这是全流水线统一的"表达表示"，03/11 的分类器和距离计算都用它。
    6. 建 kNN 图(--n-neighbors=15)并跑 Leiden 聚类
       (--leiden-resolution=2.0)，cluster 标签存 obs["leiden_res_2.00"]。
    7. 把所有参数/过程记录写进 adata.uns["label_free_preprocessing"]
       （provenance 溯源），输出 h5ad(gzip 压缩) + provenance JSON 打印。

两种运行模式（main）：
    a) 常规模式：输入原始 h5ad，完整重算上面 1-7。
    b) --prepared-input 复用模式：已有预处理好的 h5ad 时，跳过重算，
       只把原始 h5ad(--input) 的 raw counts 按"细胞 ID + 大写基因名"严格
       对齐后挂到 layers["counts"]（attach_raw_counts）。用于后续补充
       counts 层而不改变已冻结的 PCA/聚类结果。

关键函数调用关系：
    main
     ├─ prepare_label_free_analysis          # 常规模式主流程
     │   ├─ reliability.data_contracts.validate_counts   # counts 合法性校验
     │   └─ scale_for_pca                    # 稀疏安全的 z-score
     │       └─ scale_sparse_equivalent      # 分块稀疏缩放实现
     └─ attach_raw_counts                    # 复用模式：对齐并挂 counts
         └─ validate_counts

下游依赖：02 需要 obs["leiden_res_2.00"] 和 obsm["X_pca"]；
          05/09/11 需要 layers["counts"] 和 obsm["X_pca"]。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis


def parse_args() -> argparse.Namespace:
    """【中文】解析命令行参数。

    --input              输入 h5ad（Xenium 计数矩阵）；复用模式下是"原始"数据源
    --out                输出 h5ad 路径
    --prepared-input     可选；给定后进入"复用模式"：只对齐挂载 raw counts，
                         不重算 PCA/Leiden
    --expression-layer   可选；标准化前改用该 AnnData layer（例如 "counts"）
    --n-pcs              PCA 主成分数，默认 30
    --n-neighbors        kNN 图邻居数，默认 15
    --leiden-resolution  Leiden 分辨率，默认 2.0（越大 cluster 越多越细）
    --min-counts         QC：细胞最少 UMI 总数，默认 10
    --min-genes          QC：细胞最少检测基因数，默认 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prepared-input",
        type=Path,
        help=(
            "Reuse an existing label-free AnnData and attach raw counts from --input "
            "after strict cell/gene alignment, without recomputing PCA or Leiden."
        ),
    )
    parser.add_argument(
        "--expression-layer",
        default=None,
        help="Use this AnnData layer as the expression matrix before normalization (for example: counts).",
    )
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--leiden-resolution", type=float, default=2.0)
    parser.add_argument("--min-counts", type=int, default=10)
    parser.add_argument("--min-genes", type=int, default=5)
    return parser.parse_args()


def scale_sparse_equivalent(
    matrix: sparse.spmatrix,
    *,
    max_value: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Represent Scanpy's centred scaling without materialising a dense matrix.

    The returned matrix stores ``scaled + mean / std``. PCA centres its input,
    and gene-wise rank tests are invariant to this column-wise offset, so both
    downstream operations are equivalent to using the dense centred matrix.

    【中文】稀疏等价的 z-score 缩放 —— 本文件的核心技巧。
    ------------------------------------------------------------------
    问题：Scanpy 的 sc.pp.scale 要先把稀疏矩阵 densify（2 万细胞 × 全基因
    的稠密矩阵内存放不下）。而稀疏矩阵里存 0 最省空间，z-score 的"减均值"
    会把所有 0 变成 -mean/std（非零），破坏稀疏性。

    实现思路：利用两个数学性质
      ① PCA 内部会先对输入做中心化（zero_center=True），所以"存
         (x-mean)/std"和"存 (x-mean)/std + mean/std"喂给 PCA 结果完全一样，
         只是列偏移；scale(max_value=10) 的 clip 只用上界 max，
         下界 -max 对稀疏存的 0 项不会触发（前提是 shift=mean/std ≤ max，
         否则报错要求换大内存机器，见下面的检查）。
      ② 基因级 Wilcoxon 秩检验对列偏移不变（同一基因整体加减常数不改变
         排名），所以 02 的 marker 检验也不受影响。

    稀疏存储格式是 CSR：data 数组存所有非零值、indices 存对应列号。
    只需把 data 分块（100 万个/块）除以所在基因的 std，再对超过
    upper = shift + max 的值 clip，最后 eliminate_zeros() 清掉缩放后变 0 的项。

    参数:
        matrix    稀疏 counts 矩阵（细胞 × 基因）
        max_value clip 上界（与 sc.pp.scale(max_value=10) 一致）
    返回:
        (缩放后的 CSR 矩阵, 每基因均值 mean, 每基因标准差 std)
        ——mean/std 存进 adata.var 供溯源，矩阵实际存的是
        "(x-mean)/std + mean/std"（等价于"缩放后 + 列偏移"）。
    """
    scaled = matrix.tocsr(copy=True)
    if not np.issubdtype(scaled.dtype, np.floating):
        scaled = scaled.astype(np.float64)

    mean, population_variance = mean_variance_axis(scaled, axis=0)
    variance = population_variance * scaled.shape[0] / (scaled.shape[0] - 1)
    std = np.sqrt(variance)
    std[std == 0] = 1
    shift = mean / std
    if np.any(shift > max_value):
        genes = np.flatnonzero(shift > max_value)
        raise RuntimeError(
            "Exact sparse scaling cannot represent the lower clipping bound for "
            f"{len(genes)} genes. Use a larger-memory host for dense scaling."
        )

    data = scaled.data
    columns = scaled.indices
    block_size = 1_000_000
    upper = shift + max_value
    for start in range(0, data.size, block_size):
        stop = min(start + block_size, data.size)
        block_columns = columns[start:stop]
        data[start:stop] /= std[block_columns]
        np.minimum(data[start:stop], upper[block_columns], out=data[start:stop])
    scaled.eliminate_zeros()
    return scaled, mean, std


def scale_for_pca(adata: sc.AnnData, *, max_value: float) -> str:
    """Scale expression while keeping sparse slides sparse when mathematically safe.

    【中文】给 PCA 准备缩放矩阵的调度器：
      - 矩阵本来就是稠密 → 直接用 scanpy 标准 sc.pp.scale（返回
        "dense_centered"，存的是真正中心化的 (x-mean)/std）；
      - 矩阵是稀疏 → 调 scale_sparse_equivalent 做稀疏等价缩放（返回
        "sparse_column_shift_equivalent"），mean/std 存进 adata.var 备查。
    返回值只是记录用了哪条路径，写进 provenance 供审计。
    """
    if not sparse.issparse(adata.X):
        sc.pp.scale(adata, max_value=max_value)
        return "dense_centered"

    scaled, mean, std = scale_sparse_equivalent(adata.X, max_value=max_value)
    adata.X = scaled
    adata.var["mean"] = mean
    adata.var["std"] = std
    adata.var["sparse_centering_shift"] = mean / std
    return "sparse_column_shift_equivalent"


def prepare_label_free_analysis(
    adata: sc.AnnData,
    *,
    expression_layer: str | None = None,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    leiden_resolution: float = 2.0,
    min_counts: int = 10,
    min_genes: int = 5,
) -> tuple[sc.AnnData, dict[str, object]]:
    """Prepare one slide while preserving unnormalised counts for CNV stages.

    【中文】本脚本的主处理函数（main 的常规模式调用它）。
    ------------------------------------------------------------------
    参数:
        adata                原始 AnnData（X 或指定 layer = raw counts）
        expression_layer     改从该 layer 取 raw counts（None 则用 X）
        n_pcs                PCA 维数（默认 30）
        n_neighbors          kNN 邻居数（默认 15）
        leiden_resolution    Leiden 分辨率（默认 2.0）
        min_counts/min_genes QC 阈值（默认 10 / 5）
    返回:
        (预处理后的 adata, provenance 字典)

    被调用: main；调用: validate_counts（reliability.data_contracts）、
    scale_for_pca、scanpy 的 normalize_total/log1p/pca/neighbors/leiden。
    """
    from reliability.data_contracts import validate_counts

    original_cells = adata.n_obs

    # ---- 步骤 1：锁定 raw counts。无论输入是 X 还是某个 layer，先把
    # "未经任何标准化的整数计数"备份出来：X 恢复为 raw counts，
    # layers["counts"] 永久保留原始计数（05/09 的 CNV/基因证据要用）。

    if expression_layer is not None:
        if expression_layer not in adata.layers:
            raise KeyError(f"Missing AnnData layer: {expression_layer!r}")
        raw_counts = adata.layers[expression_layer].copy()
    else:
        raw_counts = adata.X.copy()
    validate_counts(raw_counts)
    adata.X = raw_counts.copy()
    adata.layers["counts"] = raw_counts

    # ---- 步骤 2-3：基因名大写统一 + 唯一性校验 + QC 过滤低质量细胞。
    # total_counts_raw / n_genes_by_counts_raw 两列 QC 指标存在 obs 里备查。
    adata.var_names = adata.var_names.astype(str).str.upper()
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell IDs and uppercase gene symbols must be unique")
    adata.obs["total_counts_raw"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["n_genes_by_counts_raw"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    keep = (adata.obs["total_counts_raw"] >= min_counts) & (
        adata.obs["n_genes_by_counts_raw"] >= min_genes
    )
    adata = adata[keep].copy()

    # ---- 步骤 4：标准化三连（对过滤后的细胞）。
    # normalize_total: 每个细胞缩放到总 UMI=10000（消除测序深度差异）
    # log1p:           log(1+x) 压缩长尾分布
    # scale_for_pca:   z-score + clip（稀疏安全版，见 scale_sparse_equivalent）
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)
    scaling_representation = scale_for_pca(adata, max_value=10)
    # ---- 步骤 5：PCA。主成分数不能超过 基因数-1 和 细胞数-1（数学上限）。
    # svd_solver="arpack" 适合大稀疏矩阵；结果存 obsm["X_pca"]，是 03/11 的
    # 分类器特征、03/09 的 cluster 中心距离计算所共用的表达表示。
    n_comps = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack", zero_center=True)
    # ---- 步骤 6：表达 kNN 图 + Leiden 聚类。cluster key 名字里带分辨率，
    # 避免不同分辨率结果互相覆盖；02/03/09/11 都用 "leiden_res_2.00" 这一列。
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)
    cluster_key = f"leiden_res_{leiden_resolution:.2f}"
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added=cluster_key)

    # ---- 步骤 7：provenance 溯源字典 —— 记录所有参数与处理路径，
    # 写进 adata.uns 让输出 h5ad 自解释；official_labels_policy: "not_loaded"
    # 是全流水线的固定声明：从未加载人工注释。
    provenance: dict[str, object] = {
        "official_labels_policy": "not_loaded",
        "input_cells": original_cells,
        "retained_cells": adata.n_obs,
        "min_counts": min_counts,
        "min_genes": min_genes,
        "normalization": "normalize_total(target_sum=10000), log1p, scale(max_value=10)",
        "scaling_representation": scaling_representation,
        "scaling_equivalence": (
            "sparse X stores scaled + mean/std; PCA centering and gene-wise ranks "
            "are equivalent to dense centered scaling"
        ),
        "expression_source": expression_layer or "X",
        "raw_counts_layer": "counts",
        "representation": "X_pca",
        "n_pcs": n_comps,
        "n_neighbors": n_neighbors,
        "leiden_key": cluster_key,
        "leiden_resolution": leiden_resolution,
    }
    adata.uns["label_free_preprocessing"] = provenance
    return adata, provenance


def attach_raw_counts(
    prepared: sc.AnnData,
    raw: sc.AnnData,
    *,
    expression_layer: str | None = None,
    output_layer: str = "counts",
) -> tuple[sc.AnnData, dict[str, object]]:
    """Attach raw counts to an existing analysis after explicit ID alignment.

    【中文】复用模式的核心函数（main 在 --prepared-input 时调用）。
    ------------------------------------------------------------------
    场景：已有 01 的预处理产物（PCA/Leiden 已冻结），但当时没有保留
    raw counts 层。现在要把原始数据的 counts 挂上去，供 05 的 CNV 使用，
    同时【不重算】任何表示。

    参数:
        prepared          已预处理的 AnnData（将被原位修改）
        raw               原始 h5ad 读入的 AnnData
        expression_layer  raw 里取哪个 layer 当 counts（None 用 X）
        output_layer      counts 挂到 prepared 的哪个 layer（默认 "counts"）
    返回:
        (挂好 counts 的 prepared, provenance 字典)

    对齐规则（严格契约，任何不符直接报错，绝不静默丢弃）:
      ① 两边细胞 ID、基因名各自唯一；
      ② 基因名统一大写后仍唯一；
      ③ prepared 的每个细胞、每个基因在 raw 里都必须存在
         （missing_cells/missing_genes 非空 → ValueError）；
      ④ 按 prepared 的 (obs_names, var_names) 顺序切 raw 的子矩阵。
    """
    if not prepared.obs_names.is_unique or not raw.obs_names.is_unique:
        raise ValueError("Both AnnData objects must have unique cell IDs.")
    prepared.var_names = prepared.var_names.astype(str).str.upper()
    raw.var_names = raw.var_names.astype(str).str.upper()
    if not prepared.var_names.is_unique or not raw.var_names.is_unique:
        raise ValueError("Gene names must remain unique after upper-casing.")
    missing_cells = prepared.obs_names.difference(raw.obs_names)
    missing_genes = prepared.var_names.difference(raw.var_names)
    if len(missing_cells) or len(missing_genes):
        raise ValueError(
            f"Raw input misses {len(missing_cells)} prepared cells and "
            f"{len(missing_genes)} prepared genes."
        )
    aligned = raw[prepared.obs_names, prepared.var_names]
    if expression_layer is not None:
        if expression_layer not in aligned.layers:
            raise KeyError(f"Missing AnnData layer: {expression_layer!r}")
        counts = aligned.layers[expression_layer].copy()
    else:
        counts = aligned.X.copy()
    from reliability.data_contracts import validate_counts

    validate_counts(counts)
    prepared.layers[output_layer] = counts
    provenance: dict[str, object] = {
        "official_labels_policy": "not_loaded",
        "alignment": "cell_id_and_uppercase_gene_name",
        "source_expression": expression_layer or "X",
        "output_layer": output_layer,
        "cell_count": prepared.n_obs,
        "gene_count": prepared.n_vars,
    }
    prepared.uns["raw_counts_provenance"] = provenance
    return prepared, provenance


def main() -> None:
    """【中文】命令行入口。按是否传 --prepared-input 二选一：
      复用模式  → attach_raw_counts(已预处理, 原始)   （只挂 counts）
      常规模式  → prepare_label_free_analysis(原始)   （完整预处理）
    最后统一：创建输出目录 → 写 h5ad(gzip) → 打印 provenance JSON 和
    细胞/基因/cluster 数摘要。
    """
    args = parse_args()
    if args.prepared_input is not None:
        adata, counts_provenance = attach_raw_counts(
            sc.read_h5ad(args.prepared_input),
            sc.read_h5ad(args.input),
            expression_layer=args.expression_layer,
        )
        provenance = dict(adata.uns.get("label_free_preprocessing", {}))
        print(json.dumps(counts_provenance, indent=2))
    else:
        adata, provenance = prepare_label_free_analysis(
            sc.read_h5ad(args.input),
            expression_layer=args.expression_layer,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            leiden_resolution=args.leiden_resolution,
            min_counts=args.min_counts,
            min_genes=args.min_genes,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out, compression="gzip")
    print(json.dumps(provenance, indent=2))
    print(
        f"wrote {args.out}: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{adata.obs[str(provenance['leiden_key'])].nunique()} Leiden clusters"
    )


if __name__ == "__main__":
    main()
