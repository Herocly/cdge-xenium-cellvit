"""Expand three normal lineages from B1.9 seeds, with explicit rejection.

Generation never reads official annotations. Evaluation is a separate command.
This is a development experiment, not a frozen generalization method.
"""

from __future__ import annotations

import argparse

import hashlib

import json

import sys

from pathlib import Path

import numpy as np

import pandas as pd

import scanpy as sc

from sklearn.neighbors import NearestNeighbors

B19 = Path(__file__).resolve().parent / "code"

sys.path.insert(0, str(B19))

from reliability.cluster_evidence_gate import (  # noqa: E402
    prediction_allowed_by_cluster_evidence,
    strong_cluster_label_map,
)

from reliability.data_contracts import aligned_cell_table, boolean_flags  # noqa: E402

from reliability.expression_space import expression_features  # noqa: E402

from reliability.iterative_expansion import (  # noqa: E402
    accepted_candidate_mask,
    select_spatially_balanced_additions,
    spatial_quadrants,
    supporting_cluster_evidence,
)

from reliability.seed_classifier import (  # noqa: E402
    fit_balanced_logistic_classifier,
    prediction_neighbor_agreement,
)

NORMAL = ("Connective_tissue", "Inflammatory", "Epithelial")

MIN_CONFIDENCE = 0.80

MIN_MARGIN = 0.20

MIN_NEIGHBOR = 0.70

MIN_STRUCTURAL = 0.50

MIN_ANNOTATION = 0.50

RADIUS_NEIGHBORS = 5

RADIUS_QUANTILE = 0.99

MAX_PER_CLASS = 500

MAX_ROUNDS = 6

CONFIDENCE_WEIGHT_START = 0.50

CONFIDENCE_WEIGHT_END = 0.80

BUDGET_GROWTH_PER_ROUND = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def screen_low_information_seed_clusters(
    frame: pd.DataFrame, cell_ids: np.ndarray, clusters: np.ndarray,
    total_counts: np.ndarray, detected_genes: np.ndarray, quantile: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Withhold core seeds from clusters poor in both slide-relative QC axes."""
    if not 0 < quantile < 0.5:
        raise ValueError("QC quantile must be between zero and one half")
    cell_ids = np.asarray(cell_ids).astype(str)
    clusters = np.asarray(clusters).astype(str)
    total_counts = np.asarray(total_counts, dtype=float)
    detected_genes = np.asarray(detected_genes, dtype=float)
    if any(values.shape != (len(cell_ids),) for values in (clusters, total_counts, detected_genes)):
        raise ValueError("QC data must share the analysis cell axis")
    if not np.isfinite(total_counts).all() or not np.isfinite(detected_genes).all():
        raise ValueError("QC values must be finite")
    if (total_counts < 0).any() or (detected_genes < 0).any():
        raise ValueError("QC values cannot be negative")
    aligned = aligned_cell_table(frame, cell_ids)
    if not np.array_equal(aligned["source_leiden_cluster"].astype(str).to_numpy(), clusters):
        raise ValueError("Tiered seed clusters do not match the analysis clusters")
    selected = boolean_flags(aligned["is_initial_seed"])
    count_cutoff = float(np.quantile(total_counts, quantile))
    gene_cutoff = float(np.quantile(detected_genes, quantile))
    cluster_qc = pd.DataFrame({
        "cluster": clusters, "total_counts": total_counts, "detected_genes": detected_genes,
    }).groupby("cluster", sort=True).median()
    excluded = cluster_qc.index[
        (cluster_qc.total_counts < count_cutoff) & (cluster_qc.detected_genes < gene_cutoff)
    ].tolist()
    withheld = selected & np.isin(clusters, excluded)
    result = aligned.copy()
    result.loc[withheld, "is_initial_seed"] = False
    result.loc[withheld, "refined_pseudo_label"] = "Unknown"
    result.insert(0, "cell_id", cell_ids)
    audit = {
        "official_labels_policy": "not_loaded",
        "qc_quantile": quantile,
        "slide_count_cutoff": count_cutoff,
        "slide_detected_genes_cutoff": gene_cutoff,
        "excluded_clusters": excluded,
        "excluded_cluster_medians": cluster_qc.loc[excluded].reset_index().to_dict("records"),
        "withheld_seed_count": int(withheld.sum()),
        "retained_seed_counts": {
            label: int((boolean_flags(result["is_initial_seed"]) & result["refined_pseudo_label"].eq(label)).sum())
            for label in NORMAL
        },
    }
    if any(count == 0 for count in audit["retained_seed_counts"].values()):
        raise ValueError("QC screening removed an entire normal seed class")
    return result, audit


def screen_seeds(args: argparse.Namespace) -> None:
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    adata = sc.read_h5ad(args.adata, backed="r")
    try:
        cells = adata.obs_names.astype(str).to_numpy()
        if not pd.Index(cells).is_unique:
            raise ValueError("AnnData has duplicate cell IDs")
        result, audit = screen_low_information_seed_clusters(
            pd.read_csv(args.tiered_seeds, dtype={"cell_id": str}), cells,
            adata.obs["leiden_res_2.00"].astype(str).to_numpy(),
            adata.obs["total_counts_raw"].to_numpy(),
            adata.obs["n_genes_by_counts_raw"].to_numpy(),
        )
    finally:
        adata.file.close()
    out.mkdir(parents=True, exist_ok=True)
    output = out / "qc_screened_tiered_seeds_no_official_labels.csv"
    result.to_csv(output, index=False)
    audit["inputs"] = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in (("adata", args.adata), ("tiered_seeds", args.tiered_seeds),
                           ("screening_code", Path(__file__)))
    }
    audit["screened_seeds_sha256"] = sha256(output)
    (out / "qc_screening_manifest_no_official_labels.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    screening = parser.add_subparsers(dest="command", required=True).add_parser("screen-seeds")
    screening.add_argument("--adata", type=Path, required=True)
    screening.add_argument("--tiered-seeds", type=Path, required=True)
    screening.add_argument("--out-dir", type=Path, required=True)
    screen_seeds(parser.parse_args())

if __name__ == "__main__":
    main()
