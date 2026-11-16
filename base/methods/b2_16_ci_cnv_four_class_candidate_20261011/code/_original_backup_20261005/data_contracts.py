"""Explicit cell axes and raw-expression contracts shared by pipeline stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


CANONICAL_CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY", "chrM")


def normalize_chromosome(values: pd.Series) -> pd.Series:
    """Accept bare or singly prefixed human chromosomes; reject malformed names."""
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
    """Canonical, unique GRCh38 gene intervals without changing assembly or positions."""
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
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Raw counts must be finite and non-negative")
    if not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise ValueError("Raw counts must be integer-valued, not normalized/log/scaled expression")


def usable_panel_coordinates(coordinates: pd.DataFrame, panel_genes: object) -> pd.DataFrame:
    coordinates = normalize_genomic_coordinates(coordinates)
    panel = coordinates.loc[coordinates["gene_name"].astype(str).isin(set(panel_genes))]
    return panel.groupby("chromosome", observed=True).filter(lambda rows: len(rows) >= 3).copy()


def aligned_cell_table(frame: pd.DataFrame, cell_ids: object) -> pd.DataFrame:
    ids = pd.Index(np.asarray(cell_ids).astype(str), name="cell_id")
    keyed = frame.assign(cell_id=frame["cell_id"].astype(str)).set_index("cell_id")
    if not ids.is_unique or not keyed.index.is_unique or set(ids) != set(keyed.index):
        raise ValueError("Evidence must contain exactly one row per analysis cell")
    return keyed.reindex(ids)


def boolean_flags(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    if values.isna().any() or not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Evidence contains missing or invalid boolean flags")
    return normalized.isin({"true", "1"}).to_numpy()


def positive_quantile_evidence(scores: object, quantile: float) -> tuple[np.ndarray, float]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Gene scores must be a nonempty finite vector")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between zero and one")
    cutoff = float(np.quantile(values, quantile))
    if np.ptp(values) == 0:
        return np.zeros(len(values), dtype=bool), cutoff
    return (values > 0) & (values >= cutoff), cutoff
