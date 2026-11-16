"""Shared coarse-cell ontology for database voting and evaluation adapters.

【中文详解 —— 共享"细胞类型本体"模块】
====================================================================
被两处使用：
  ① 02 的 marker 数据库映射链（经 ontology_tree.CellOntologyMapper）；
  ② evaluate 的 pancreas 真值适配器（把参考术语翻到核心类）。

它维护一张人工固定的术语表（CSV：term,core_class 两列，可选
database_override 列），把任意写法的细胞类型名归一化后查表映射到
6 个核心类之一：
    Epithelial / Connective_tissue / Neoplastic / Inflammatory / Others / Dead

设计要点：
  - 只做【显式、可审计】的映射——表中没有的术语不会猜；
  - 同一术语出现两种映射 → 当场报错（拒绝自相矛盾的表）；
  - normalize_cell_type 是"温和归一化"（只规整拼写，不推断类别）。
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


# 【中文】核心类常量：数据库投票与评估适配共用的 6 类封闭集合。
CORE_CLASSES = (
    "Epithelial",
    "Connective_tissue",
    "Neoplastic",
    "Inflammatory",
    "Others",
    "Dead",
)


def normalize_cell_type(value: object) -> str:
    """Normalize spelling and punctuation without inferring a class.

    【中文】细胞类型名的温和归一化（本模块所有查表的第一步）：
      小写 → 希腊字母转写(α→alpha 等) → 去变音符号(NFKD) →
      非字母数字压成空格 → 结尾 "cells" 改 "cell"。
    只规整拼写，【绝不推断类别】； ontology_tree 的别名查找也复用它。"""
    text = str(value).strip().lower()
    for source, replacement in (("α", " alpha "), ("β", " beta "), ("γ", " gamma "), ("δ", " delta ")):
        text = text.replace(source, replacement)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    tokens = re.sub(r"[^a-z0-9]+", " ", text).strip().split()
    if tokens and tokens[-1] == "cells":
        tokens[-1] = "cell"
    return " ".join(tokens)


def normalize_lookup(value: object) -> str:
    """Fold simple noun plurals as well as the existing cells -> cell rule.

    【中文】在 normalize_cell_type 基础上再折叠简单英语复数：
    结尾 "s"（但不是 ss/is）去掉，如 "fibroblasts"→"fibroblast"。
    供 ontology_tree 的名字/同义词查找用，提高命中率。"""
    tokens = normalize_cell_type(value).split()
    if tokens and tokens[-1].endswith("s") and not tokens[-1].endswith(("ss", "is")):
        tokens[-1] = tokens[-1][:-1]
    return " ".join(tokens)


def load_ontology(path: Path, *, database_only: bool = False) -> dict[str, str]:
    """Load explicit term-to-core-class assignments and reject conflicts.

    【中文】读本体 CSV（term,core_class 必需列）→ {归一化术语: 核心类}。
    校验：核心类必须在 CORE_CLASSES 里；同一术语映射到两个不同类 → 报错。
    database_only=True 时要求表里有 database_override 布尔列，只保留
    True 的行（"该术语也用于数据库投票"的显式声明，evaluate 的
    pancreas 适配器用默认 False）。
    """
    table = pd.read_csv(path)
    required = {"term", "core_class"}
    if not required.issubset(table.columns):
        raise ValueError(f"Ontology must contain columns {sorted(required)}")
    invalid = sorted(set(table["core_class"].astype(str)) - set(CORE_CLASSES))
    if invalid:
        raise ValueError(f"Ontology contains unsupported core classes: {invalid}")
    if database_only:
        if "database_override" not in table.columns:
            raise ValueError("Ontology must declare database_override for generation use")
        table = table.loc[table["database_override"].astype(bool)].copy()
    mapping: dict[str, str] = {}
    for row in table.itertuples(index=False):
        key = normalize_cell_type(row.term)
        label = str(row.core_class)
        if key in mapping and mapping[key] != label:
            raise ValueError(f"Conflicting ontology assignments for {row.term!r}")
        mapping[key] = label
    return mapping


def apply_ontology_overrides(
    table: pd.DataFrame,
    term_column: str,
    ontology: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply only explicit ontology rows to one database mapping table.

    【中文】把本体表的"人工覆盖"应用到一个数据库映射表上
    （ontology_tree.map_marker_databases 的数据库行 → 类别 one-hot 矩阵
    生成前的最后一道修正）。
    做法：term_column 列归一化后查 ontology；命中的行把所有类别列清 0、
    再把目标类别置 1（one-hot 覆写）；要求本体指定的类别在该表里真实
    存在，否则报错。
    返回: (覆写后的表, {类别: 被覆写行数} 审计计数)。"""
    output = table.copy()
    normalized = output[term_column].map(normalize_cell_type)
    assigned = normalized.map(ontology)
    available_classes = [label for label in CORE_CLASSES if label in output.columns]
    unsupported = sorted(set(assigned.dropna()) - set(available_classes))
    if unsupported:
        raise ValueError(
            f"Ontology requests classes unavailable in {term_column} mapping: {unsupported}"
        )
    mask = assigned.notna()
    output.loc[mask, available_classes] = 0
    for label in available_classes:
        output.loc[mask & assigned.eq(label), label] = 1
    counts = assigned.loc[mask].value_counts().astype(int).to_dict()
    return output, {str(key): int(value) for key, value in counts.items()}


def map_reference_terms(values: pd.Series, ontology: dict[str, str]) -> pd.Series:
    """Map external reference terms through the same frozen ontology.

    【中文】把外部参考注释术语逐个过 normalize_cell_type 后查本体字典，
    查不到返回 NaN（由调用方决定回退或报错；evaluate 的 pancreas
    适配器用 PANCREAS_TO_CORE 字典回退）。"""
    return values.astype(str).map(lambda value: ontology.get(normalize_cell_type(value)))
