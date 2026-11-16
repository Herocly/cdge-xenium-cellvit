"""Shared coarse-cell ontology for database voting and evaluation adapters."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


CORE_CLASSES = (
    "Epithelial",
    "Connective_tissue",
    "Neoplastic",
    "Inflammatory",
    "Others",
    "Dead",
)


def normalize_cell_type(value: object) -> str:
    """Normalize spelling and punctuation without inferring a class."""
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
    """Fold simple noun plurals as well as the existing cells -> cell rule."""
    tokens = normalize_cell_type(value).split()
    if tokens and tokens[-1].endswith("s") and not tokens[-1].endswith(("ss", "is")):
        tokens[-1] = tokens[-1][:-1]
    return " ".join(tokens)


def load_ontology(path: Path, *, database_only: bool = False) -> dict[str, str]:
    """Load explicit term-to-core-class assignments and reject conflicts."""
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
    """Apply only explicit ontology rows to one database mapping table."""
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
    """Map external reference terms through the same frozen ontology."""
    return values.astype(str).map(lambda value: ontology.get(normalize_cell_type(value)))
