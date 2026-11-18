"""Explicit cell axes and raw-expression contracts shared by pipeline stages.

【中文详解 —— reliability 库的"数据契约"模块】
====================================================================
这个文件没有任何业务逻辑，全部是【严格的数据合法性检查与规范化】工具。
流水线的哲学是"坏数据宁可当场报错，绝不静默带病下传"，所以 01/03/05/
07/09/10/11 都 import 本模块做前置校验。

包含的工具（函数级中文说明见各函数）：
    CANONICAL_CHROMOSOMES          合法人类染色体名常量
    normalize_chromosome           染色体名规范化（"1"/"CHR1"/"MT"→chr1/chrM）
    normalize_genomic_coordinates  基因坐标表整体规范化+校验
    validate_counts                校验矩阵确实是"非负整数 raw counts"
    usable_panel_coordinates       取"每条染色体≥3 基因"的可用 panel 坐标
    aligned_cell_table             按 cell_id 严格对齐逐细胞表（多/少行都报错）
    boolean_flags                  把 CSV 布尔列严格解析成 np.bool 数组
    positive_quantile_evidence     "正数且≥分位"的二值化证据工具
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


# 【中文】合法人类染色体名：chr1..chr22 + chrX/chrY/chrM。
# 05 的 validate_cnv_chromosomes 用它检查 CNV 轴没有混入奇怪 contig。
CANONICAL_CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY", "chrM")


def normalize_chromosome(values: pd.Series) -> pd.Series:
    """Accept bare or singly prefixed human chromosomes; reject malformed names.

    【中文】染色体名规范化（被 normalize_genomic_coordinates 调用）。
    接受 "1"/"chr1"/"CHR1"/"MT" 这类写法，统一输出 "chr1"/"chrM"；
    缺失值、非规范人类染色体（如 scaffold 名）一律报错——因为 CNV 开窗
    必须建立在干净的染色体轴上，静默丢弃会悄悄改变结果。"""
    if values.isna().any():
        raise ValueError("Chromosome names must not be missing")
    tokens = values.astype(str).str.strip().str.upper().str.removeprefix("CHR")
    tokens = tokens.replace({"MT": "M"})
    valid = {str(i) for i in range(1, 23)} | {"X", "Y", "M"}
    invalid = sorted(set(tokens) - valid)
    if invalid:
        raise ValueError(f"Noncanonical/malformed human chromosomes require explicit resolution: {invalid}")
    return "chr" + tokens


def normalize_genomic_coordinates(coordinates: pd.DataFrame) -> pd.DataFrame:
    """Canonical, unique GRCh38 gene intervals without changing assembly or positions.

    【中文】基因坐标表的统一契约（04/05 调用）。
    校验链（任何一条不过都直接抛错）：
      ① 必须有 gene_name/chromosome/start/end 四列且无缺失；
      ② 基因名大写去空白后必须唯一非空；
      ③ 保留原始染色体名到 chromosome_original 列（审计用），
         chromosome 列过 normalize_chromosome 规范化；
      ④ start/end 必须是有限整数；
      ⑤ start>0 且 end>=start（GRCh38 闭合正区间）。
    不做任何坐标换算——只换名字和查错，位置保持原样。
    """
    required = ["gene_name", "chromosome", "start", "end"]
    if not set(required).issubset(coordinates):
        raise ValueError(f"Coordinates require fields {required}")
    if coordinates[required].isna().any().any():
        raise ValueError("Genomic coordinates must not contain missing fields")
    result = coordinates.copy()
    result["gene_name"] = result["gene_name"].astype(str).str.strip().str.upper()
    if not result["gene_name"].is_unique or result["gene_name"].eq("").any():
        raise ValueError("Genomic coordinates must have unique nonempty gene symbols")
    if "chromosome_original" not in result:
        result["chromosome_original"] = result["chromosome"].astype(str)
    result["chromosome"] = normalize_chromosome(result["chromosome"])
    for column in ("start", "end"):
        numeric = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
            raise ValueError("Genomic interval endpoints must be finite integers")
        result[column] = numeric.astype(np.int64)
    if result["start"].le(0).any() or result["end"].lt(result["start"]).any():
        raise ValueError("GRCh38 coordinates must be positive closed intervals with end >= start")
    return result


def validate_counts(matrix: object) -> None:
    """【中文】校验矩阵确实是 raw counts（01/05/09 调用）。
    两个断言：
      ① 全部数值有限且 ≥0（不能有 NaN/负值）；
      ② 全部数值都是整数（容差 1e-6）——防止把已 normalize/log/scale
         过的矩阵误当 counts 用（这是流水线最容易犯的错）。
    对稀疏矩阵只检查 data 数组（存储的显式值）。"""
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Raw counts must be finite and non-negative")
    if not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise ValueError("Raw counts must be integer-valued, not normalized/log/scaled expression")


def usable_panel_coordinates(coordinates: pd.DataFrame, panel_genes: object) -> pd.DataFrame:
    """【中文】筛出"可用于 CNV 开窗"的 panel 坐标（05 调用）。
    两步过滤：① 基因名必须在 panel 的 var_names 里（交集）；
    ② 按染色体分组后，只保留【≥3 个基因】的染色体——infercnvpy
    按"染色体上连续 window_size 个基因"开窗，一条染色体基因太少
    （如只有 1 个）根本开不出有意义的窗，这是稀疏 panel 的硬性适配。
    """
    coordinates = normalize_genomic_coordinates(coordinates)
    panel = coordinates.loc[coordinates["gene_name"].astype(str).isin(set(panel_genes))]
    return panel.groupby("chromosome", observed=True).filter(lambda rows: len(rows) >= 3).copy()


def aligned_cell_table(frame: pd.DataFrame, cell_ids: object) -> pd.DataFrame:
    """【中文】逐细胞表严格对齐器（07/10/11 调用）。
    把 frame 的 cell_id 列变成索引，并【按传入的 cell_ids 顺序】重排。
    三条契约：分析细胞轴唯一、frame 索引唯一、两者集合完全相等
    （多一行或少一行都是错——"exactly one row per analysis cell"）。
    返回: 与 cell_ids 同序的 DataFrame（取子集对齐）。"""
    ids = pd.Index(np.asarray(cell_ids).astype(str), name="cell_id")
    keyed = frame.assign(cell_id=frame["cell_id"].astype(str)).set_index("cell_id")
    if not ids.is_unique or not keyed.index.is_unique or set(ids) != set(keyed.index):
        raise ValueError("Evidence must contain exactly one row per analysis cell")
    return keyed.reindex(ids)


def boolean_flags(values: pd.Series) -> np.ndarray:
    """【中文】严格布尔解析器（07/10/11 调用）。
    接受 "true"/"false"/"1"/"0"（大小写/空白不敏感）→ 返回 np.bool 数组。
    含缺失值或任何其他写法（如 "TRUE "/yes）直接报错——CSV 来的布尔列
    不允许有含糊地带。"""
    normalized = values.astype(str).str.strip().str.lower()
    if values.isna().any() or not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Evidence contains missing or invalid boolean flags")
    return normalized.isin({"true", "1"}).to_numpy()


def positive_quantile_evidence(scores: object, quantile: float) -> tuple[np.ndarray, float]:
    """【中文】"正数分位"二值化证据工具（09 的基因证据用）。
    输入一维连续分数和分位数 q（0~1），返回:
      (布尔数组: 分数>0 且 ≥q 分位, 阈值 cutoff)
    细节：若所有分数完全相同(ptp=0)则全 False（没有区分度就不产证据）；
    要求">0"是防止全为 0 的退化分布被分位阈值误判成证据。
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Gene scores must be a nonempty finite vector")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between zero and one")
    cutoff = float(np.quantile(values, quantile))
    if np.ptp(values) == 0:
        return np.zeros(len(values), dtype=bool), cutoff
    return (values > 0) & (values >= cutoff), cutoff
