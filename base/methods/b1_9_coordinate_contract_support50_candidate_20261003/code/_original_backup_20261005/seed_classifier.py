"""The small class-balanced classifier and local-neighbour diagnostics."""

from __future__ import annotations

import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression


def cell_neighbor_consistency(
    adata: sc.AnnData, clusters: np.ndarray
) -> np.ndarray:
    """Return the fraction of each cell's kNN neighbours in its own cluster."""
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
    """Fit the small class-balanced linear classifier used in each iteration."""
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
    """How often a cell's predicted label agrees with its predicted neighbours."""
    agreement = np.full(adata.n_obs, np.nan)
    if "connectivities" not in adata.obsp:
        return agreement
    graph = adata.obsp["connectivities"].tocsr()
    for index in range(adata.n_obs):
        neighbours = graph.indices[graph.indptr[index] : graph.indptr[index + 1]]
        if len(neighbours):
            agreement[index] = np.mean(predicted[neighbours] == predicted[index])
    return agreement
