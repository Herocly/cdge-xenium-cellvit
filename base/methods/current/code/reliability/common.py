"""Small label and ranking helpers shared by the reliability workflow.

【中文详解 —— 小工具箱：标签归并与排名】
====================================================================
被 02（percentile_score / pseudo_to_core）等模块共用。
注意本模块同时提供两条"标签归并"路径：
  official_to_core —— 数据集官方注释 → 四核心类（只许 evaluate 判卷用）
  pseudo_to_core   —— CDGE 投票标签 → 四核心类（流水线内部使用）
两者的类别体系一致，但流向严格分离：官方标签永远不进训练侧。
normalized_entropy 是事后描述 cluster 纯度的判卷指标（当前主流水线
未调用，保留供分析）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def official_to_core(label: object) -> str:
    # 把数据集自带的 17 种细粒度 official 注释（如 "0_Proliferative_Tumor_Cells"、
    # "1_Macrophages"、"2_Stromal_Cells"、"14_Normal_Myoepithelial_Cells" ...）
    # 归并成 4 个核心大类 + Unknown + Others，供最后判卷用。
    # 只在评价时调用，绝不参与训练 —— 它是"标准答案"，不是"输入特征"。
    value = str(label)
    if not value or value == "Unknown" or value == "nan":
        return "Unknown"
    if "Tumor_Cells" in value:
        return "Neoplastic"
    if "Normal_Epithelial" in value or "Myoepithelial" in value:
        return "Epithelial"
    if any(name in value for name in ("Macrophages", "Lymphocytes", "Plasma_Cells")):
        return "Inflammatory"
    if any(name in value for name in ("Stromal_Cells", "Perivascular_Cells", "Endothelial_Cells")):
        return "Connective_tissue"
    return "Others"


def pseudo_to_core(label: object) -> str:
    # marker voting 输出的标签归并到同一套核心类别，方便和 official 对表。
    # "Epithelial_Neoplastic" 是 CDGE 原代码里"上皮投票后再做肿瘤检查升级"的产物，
    # 在本次 20k 数据上没有任何 cluster 达到升级阈值，所以实际没有出现。
    value = str(label)
    if value == "Epithelial_Neoplastic":
        return "Neoplastic"
    if value in {"Epithelial", "Inflammatory", "Connective_tissue", "Others"}:
        return value
    return "Unknown"


def normalized_entropy(values: pd.Series) -> float:
    # 一个 cluster 内 official 标签分布的归一化熵：
    #   0.0 = 所有细胞同一标签（最纯）；1.0 = 标签完全均匀散布（最混）。
    # 只用 official 标签算，属于"事后描述 cluster 有多混"的判卷指标。
    counts = values.value_counts()
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    probabilities = counts.to_numpy(dtype=float) / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(counts)))


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    # 把一列数变成 0~1 的百分位排名。作用：不同量纲的指标（silhouette 是
    # -1~1，dispersion 是几十的欧氏距离……）可以等权相加，不被量纲绑架。
    # 这是纯无标签变换，只用 cluster 之间的相对排名。
    ranked = series.rank(method="average", pct=True)
    return ranked if higher_is_better else 1.0 - ranked
