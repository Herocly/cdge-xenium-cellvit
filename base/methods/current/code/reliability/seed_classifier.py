"""The small class-balanced classifier and local-neighbour diagnostics.

【中文详解 —— 迭代扩展的"分类器 + 邻居诊断"模块】
====================================================================
被 11 调用。四个函数：
  cell_neighbor_consistency       逐细胞 kNN 邻居同 cluster 比例（02 早期用）
  fit_seed_classifier             闭式岭回归分类器（早期小种子实验，保留）
  fit_balanced_logistic_classifier 11 每轮迭代用的均衡 logistic 主分类器
  prediction_neighbor_agreement   预测标签的 kNN 邻居一致率（gate ③）
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression


def cell_neighbor_consistency(
    adata: sc.AnnData, clusters: np.ndarray
) -> np.ndarray:
    """Return the fraction of each cell's kNN neighbours in its own cluster.

    【中文】逐细胞"表达图邻居同 cluster 比例"：在 01 建的
    connectivities 图上，对每个细胞取其邻居，算邻居与自己同 cluster
    的占比。缺图返回全 NaN。参数: adata(含邻居图), clusters(每细胞
    cluster 标签)。与 02 里 neighbour_consistency 逻辑相同，这里是
    reliability 库的共享版本。"""
    values = np.full(adata.n_obs, np.nan)
    if "connectivities" not in adata.obsp:
        return values
    graph = adata.obsp["connectivities"].tocsr()
    for index in range(adata.n_obs):
        neighbours = graph.indices[graph.indptr[index] : graph.indptr[index + 1]]
        if len(neighbours):
            values[index] = np.mean(clusters[neighbours] == clusters[index])
    return values


def fit_seed_classifier(
    features: np.ndarray, positions: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the small class-balanced linear classifier used in each iteration.

    【中文】早期闭式(ridge)分类器 —— 保留供对照，11 实际用的是下面的
    fit_balanced_logistic_classifier。
    实现：训练集标准化 → one-hot 目标 → 按类频率倒数给样本加权
    （class balancing）→ 加 L2 正则的正规方程直接解权重（np.linalg.solve，
    无迭代）→ softmax 出概率。
    参数: features(细胞×维特征), positions(训练细胞行号), labels(全表标签)
    返回: (每细胞预测类别, 最大类概率 confidence, 前两名概率差 margin,
           类别顺序 classes)
    """
    classes = np.unique(labels[positions])
    train_x = features[positions].astype(float)
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)
    train_x = (train_x - mean) / scale
    all_x = (features.astype(float) - mean) / scale
    train_x = np.column_stack([np.ones(len(train_x)), train_x])
    all_x = np.column_stack([np.ones(len(all_x)), all_x])
    targets = np.column_stack([(labels[positions] == label).astype(float) for label in classes])
    class_counts = np.maximum(targets.sum(axis=0), 1.0)
    sample_weights = (targets * (len(targets) / (len(classes) * class_counts))).sum(axis=1)
    weighted_x = train_x * np.sqrt(sample_weights)[:, None]
    weighted_y = targets * np.sqrt(sample_weights)[:, None]
    penalty = np.eye(weighted_x.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y)
    scores = all_x @ weights
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if probabilities.shape[1] > 1 else confidence.copy()
    return predicted, confidence, margin, classes


def fit_balanced_logistic_classifier(
    features: np.ndarray, positions: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a class-balanced logistic model; probabilities are not calibrated.

    The earlier closed-form ridge classifier is useful for the first small
    seed experiment, but its softmax scores are compressed around random
    chance.  Here we use true multinomial logistic regression with balanced
    class weights. Its scores require independent validation and do not certify
    real-label precision, especially on biased pseudo-seed samples.

    【pipeline 位置】第 11 步的核心模型,也是整个 pipeline 唯一的 classifier。
    输入:features = 每细胞前 30 维 PCA;positions = 当前所有已标注细胞
    (initial seeds + 已接受扩展);labels = 对应训练标签(E 程序子类分开)。
    流程:用训练集均值/方差标准化 → LogisticRegression(C=1.0,
    class_weight="balanced", max_iter=2000)→ 对全部细胞输出概率。
    返回:
      predicted  每个细胞的预测类别(classes 之一,含程序子类)
      confidence 最大类概率(≥0.80 gate 用)
      margin     第一名 − 第二名概率差(≥0.20 gate 用)
      classes    类别顺序
    【cell-level】此函数起,预测完全是逐细胞的;训练数据 = 种子/已扩展细胞,
    全部来自无标签流程,没有任何人工标签。
    """
    train_x = features[positions].astype(float)
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)
    train_x = (train_x - mean) / scale
    all_x = (features.astype(float) - mean) / scale
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    )
    model.fit(train_x, labels[positions])
    probabilities = model.predict_proba(all_x)
    classes = model.classes_
    predicted = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if probabilities.shape[1] > 1 else confidence.copy()
    return predicted, confidence, margin, classes


def prediction_neighbor_agreement(
    adata: sc.AnnData, predicted: np.ndarray
) -> np.ndarray:
    """How often a cell's predicted label agrees with its predicted neighbours.

    【中文】逐细胞"预测邻居一致率"（11 的 gate ③）：在表达 connectivities
    图上，每个细胞取邻居预测标签，算与自己预测相同的比例。空间/表达上
    相邻的细胞被预测成不同类的细胞会被这个 gate 拦下（低一致 → 保持
    Unknown）。缺图返回全 NaN。被 11 每轮迭代调用。"""
    agreement = np.full(adata.n_obs, np.nan)
    if "connectivities" not in adata.obsp:
        return agreement
    graph = adata.obsp["connectivities"].tocsr()
    for index in range(adata.n_obs):
        neighbours = graph.indices[graph.indptr[index] : graph.indptr[index + 1]]
        if len(neighbours):
            agreement[index] = np.mean(predicted[neighbours] == predicted[index])
    return agreement
