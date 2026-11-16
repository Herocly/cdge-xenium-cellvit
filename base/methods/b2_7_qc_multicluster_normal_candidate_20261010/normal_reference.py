#!/usr/bin/env python3
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

PROJECT = Path(__file__).resolve().parents[2]
B19 = PROJECT / "methods" / "b1_9_coordinate_contract_support50_candidate_20261003" / "code"
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


def budget_for_round(iteration: int, budget_mode: str) -> int:
    """Change only selection capacity, never evidence eligibility."""
    if budget_mode not in {"fixed", "growing"}:
        raise ValueError("Budget mode must be fixed or growing")
    if not 1 <= iteration <= MAX_ROUNDS:
        raise ValueError("Iteration is outside the predeclared budget schedule")
    return MAX_PER_CLASS + (iteration - 1) * BUDGET_GROWTH_PER_ROUND if budget_mode == "growing" else MAX_PER_CLASS


def select_with_spatial_overflow(
    eligible: np.ndarray, predicted: np.ndarray, confidence: np.ndarray,
    margin: np.ndarray, regions: np.ndarray, rank: np.ndarray, cap: int,
) -> np.ndarray:
    """Reserve one quarter per region, then fill unused slots by global rank."""
    selected: list[int] = []
    for label in NORMAL:
        floor = select_spatially_balanced_additions(
            eligible, predicted, confidence, margin, regions,
            (label,), cap, rank_score=rank,
        )
        mask = eligible & (predicted == label)
        mask[floor] = False
        remaining = np.flatnonzero(mask)
        slots = cap - len(floor)
        if slots > 0 and len(remaining):
            order = np.lexsort((-margin[remaining], -confidence[remaining], -rank[remaining]))
            selected.extend(remaining[order[:slots]].tolist())
        selected.extend(floor.tolist())
    return np.asarray(sorted(selected), dtype=int)


def confidence_weight(iteration: int, schedule: str) -> float:
    """Predeclared confidence share; non-confidence shares retain their ratios."""
    if schedule not in {"fixed", "progressive"}:
        raise ValueError("Schedule must be fixed or progressive")
    if not 1 <= iteration <= MAX_ROUNDS:
        raise ValueError("Iteration is outside the predeclared schedule")
    if schedule == "fixed":
        return CONFIDENCE_WEIGHT_START
    fraction = (iteration - 1) / (MAX_ROUNDS - 1)
    return CONFIDENCE_WEIGHT_START + fraction * (
        CONFIDENCE_WEIGHT_END - CONFIDENCE_WEIGHT_START
    )


def combined_rank(
    confidence: np.ndarray, margin: np.ndarray, neighbor: np.ndarray,
    structure: np.ndarray, annotation: np.ndarray, iteration: int, schedule: str,
) -> np.ndarray:
    """B1.9 rank components with a fixed or progressively larger confidence share."""
    weight = confidence_weight(iteration, schedule)
    remaining = (1.0 - weight) / (1.0 - CONFIDENCE_WEIGHT_START)
    return np.nan_to_num(
        weight * confidence + remaining * (
            0.20 * margin + 0.15 * neighbor + 0.10 * structure + 0.05 * annotation
        ), nan=-np.inf,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_align_seeds(frame: pd.DataFrame, cell_ids: np.ndarray) -> np.ndarray:
    aligned = aligned_cell_table(frame, cell_ids)
    selected = boolean_flags(aligned["is_initial_seed"])
    labels = aligned["training_label"].astype(str).to_numpy()
    invalid = set(labels[selected]) - set(NORMAL) - {"Neoplastic"}
    if invalid:
        raise ValueError(f"Unexpected B1.9 seed classes: {sorted(invalid)}")
    if set(labels[selected & np.isin(labels, NORMAL)]) != set(NORMAL):
        raise ValueError("All three normal seed classes are required")
    # Old Neoplastic seeds are not used to fit the normal-only classifier.
    return np.where(selected & np.isin(labels, NORMAL), labels, "Unknown").astype(object)


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


def fixed_seed_radius(
    features: np.ndarray, original_labels: np.ndarray, predicted: np.ndarray,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Reject cells outside each original seed's local PCA neighborhood.

    The scale and 99th-percentile fifth-neighbor radius use only original
    normal seeds. Newly selected pseudo-labels never change this boundary.
    """
    seeds = original_labels != "Unknown"
    if candidate_mask is None:
        candidate_mask = np.ones(len(features), dtype=bool)
    if np.asarray(candidate_mask).shape != (len(features),):
        raise ValueError("Radius candidate mask must match the cell axis")
    mean = features[seeds].mean(axis=0)
    scale = np.maximum(features[seeds].std(axis=0), 1e-6)
    scaled = (features.astype(np.float64) - mean) / scale
    within = np.full(len(features), np.inf, dtype=np.float32)
    limits: dict[str, float] = {}
    for label in NORMAL:
        reference = scaled[original_labels == label]
        if len(reference) <= RADIUS_NEIGHBORS:
            raise ValueError(f"Too few original {label} seeds for radius calibration")
        model = NearestNeighbors(n_neighbors=RADIUS_NEIGHBORS + 1, algorithm="auto", n_jobs=1)
        model.fit(reference)
        leave_one_out = model.kneighbors(reference, return_distance=True)[0][:, RADIUS_NEIGHBORS]
        limit = float(np.quantile(leave_one_out, RADIUS_QUANTILE))
        limits[label] = limit
        positions = np.flatnonzero(candidate_mask & (predicted == label))
        if len(positions):
            distances = model.kneighbors(scaled[positions], n_neighbors=RADIUS_NEIGHBORS,
                                         return_distance=True)[0][:, -1]
            within[positions] = distances.astype(np.float32)
    accepted = np.zeros(len(features), dtype=bool)
    for label in NORMAL:
        selected = candidate_mask & (predicted == label)
        accepted[selected] = within[selected] <= limits[label]
    return accepted, limits


def anchored_cluster_mask(
    clusters: np.ndarray, predicted: np.ndarray, original_labels: np.ndarray
) -> np.ndarray:
    """Require an original same-class normal seed in the expression cluster."""
    if not (len(clusters) == len(predicted) == len(original_labels)):
        raise ValueError("Anchor inputs must share the cell axis")
    accepted = np.zeros(len(clusters), dtype=bool)
    for label in NORMAL:
        anchor_clusters = np.unique(clusters[original_labels == label])
        accepted |= (predicted == label) & np.isin(clusters, anchor_clusters)
    return accepted


def align_cluster_support(
    clusters: np.ndarray, predicted: np.ndarray, table: pd.DataFrame,
    original_labels: np.ndarray,
) -> np.ndarray:
    """Apply B1.9 normal evidence plus original-seed cluster anchoring."""
    if table["cluster"].astype(str).duplicated().any():
        raise ValueError("Cluster evidence has duplicate cluster IDs")
    if set(clusters) - set(table["cluster"].astype(str)):
        raise ValueError("Cluster evidence does not cover all cells")
    strong = strong_cluster_label_map(table)
    allowed = prediction_allowed_by_cluster_evidence(clusters, predicted, strong)
    supported = supporting_cluster_evidence(
        clusters, predicted, table, MIN_STRUCTURAL, MIN_ANNOTATION
    )
    return allowed & supported & anchored_cluster_mask(clusters, predicted, original_labels)


def expand_normal_reference(
    adata: sc.AnnData, original_labels: np.ndarray, cluster_table: pd.DataFrame,
    schedule: str, budget_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Iterate with fixed original-seed novelty radius and selective gates."""
    if len(original_labels) != adata.n_obs:
        raise ValueError("Seed axis does not match AnnData")
    features = expression_features(adata, 30, "X_pca")
    clusters = adata.obs["leiden_res_2.00"].astype(str).to_numpy()
    cluster_scores = cluster_table.set_index("cluster").reindex(clusters)
    if cluster_scores[["structural_reliability_score", "annotation_confidence_score"]].isna().any().any():
        raise ValueError("Missing cluster scores on the analysis cell axis")
    structure = cluster_scores["structural_reliability_score"].to_numpy(dtype=float)
    annotation = cluster_scores["annotation_confidence_score"].to_numpy(dtype=float)
    regions = spatial_quadrants(np.asarray(adata.obsm["spatial"], dtype=float))
    labels = original_labels.copy()
    round_added = np.where(labels == "Unknown", -1, 0)
    waterfalls: list[dict[str, object]] = []
    radius_limits: dict[str, float] = {}

    for iteration in range(1, MAX_ROUNDS + 1):
        train = np.flatnonzero(round_added >= 0)
        predicted, confidence, margin, classes = fit_balanced_logistic_classifier(
            features, train, labels
        )
        if set(classes) != set(NORMAL):
            raise ValueError("Normal-only training gained or lost a class")
        neighbor = prediction_neighbor_agreement(adata, predicted)
        cluster_ok = align_cluster_support(clusters, predicted, cluster_table, original_labels)
        unlabelled = round_added < 0
        before_radius = accepted_candidate_mask(
            unlabelled, confidence, margin, neighbor,
            np.ones(len(labels), dtype=bool), cluster_ok,
            MIN_CONFIDENCE, MIN_MARGIN, MIN_NEIGHBOR,
        )
        radius_ok, limits = fixed_seed_radius(features, original_labels, predicted, before_radius)
        if not radius_limits:
            radius_limits = limits
        eligible = before_radius & radius_ok
        rank = combined_rank(
            confidence, margin, neighbor, structure, annotation, iteration, schedule
        )
        cap = budget_for_round(iteration, budget_mode)
        additions = select_with_spatial_overflow(
            eligible, predicted, confidence, margin, regions, rank, cap,
        )
        for label in NORMAL:
            subset = unlabelled & (predicted == label)
            row: dict[str, object] = {
                "round": iteration, "class": label, "unlabelled_predicted": int(subset.sum()),
                "class_budget": cap,
            }
            for name, gate in (
                ("confidence", confidence >= MIN_CONFIDENCE),
                ("margin", margin >= MIN_MARGIN),
                ("neighbor", neighbor >= MIN_NEIGHBOR),
                ("cluster_evidence", cluster_ok),
                ("original_seed_radius", radius_ok),
            ):
                subset &= gate
                row["after_" + name] = int(subset.sum())
            row["selected_after_spatial_budget"] = int((predicted[additions] == label).sum())
            waterfalls.append(row)
        if not len(additions):
            break
        labels[additions] = predicted[additions]
        round_added[additions] = iteration

    result = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "leiden_cluster": clusters,
        "normal_label": labels,
        "seed_round": round_added,
        "is_original_normal_seed": round_added == 0,
    })
    return result, pd.DataFrame(waterfalls), radius_limits


def generate(args: argparse.Namespace) -> None:
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    adata = sc.read_h5ad(args.adata)
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData has duplicate cell IDs")
    seeds = pd.read_csv(args.seed_file, dtype={"cell_id": str})
    original = validate_and_align_seeds(seeds, adata.obs_names.astype(str).to_numpy())
    table = pd.read_csv(args.cluster_evidence, dtype={"cluster": str})
    result, waterfall, limits = expand_normal_reference(
        adata, original, table, args.schedule, args.budget_mode
    )
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "normal_reference_no_official_labels.csv", index=False)
    waterfall.to_csv(out / "normal_gate_waterfall_no_official_labels.csv", index=False)
    config = {
        "method": "B2.7-QC-multicluster-normal-development",
        "official_labels_policy": "not_loaded_during_generation",
        "normal_classes": NORMAL,
        "classifier": "B1.9 balanced logistic, 30 PCA components",
        "confidence": MIN_CONFIDENCE,
        "margin": MIN_MARGIN,
        "neighbor_agreement": MIN_NEIGHBOR,
        "min_structural": MIN_STRUCTURAL,
        "min_annotation": MIN_ANNOTATION,
        "cluster_anchor_policy": "each normal prediction requires original same-class seed in its expression cluster",
        "confidence_weight_schedule": args.schedule,
        "confidence_weight_start": CONFIDENCE_WEIGHT_START,
        "confidence_weight_end_if_progressive": CONFIDENCE_WEIGHT_END,
        "budget_mode": args.budget_mode,
        "budget_growth_per_round": BUDGET_GROWTH_PER_ROUND,
        "spatial_allocation": "quarter-per-region reservation then ranked overflow",
        "remaining_rank_components_at_start": {
            "margin": 0.20, "neighbor": 0.15, "structure": 0.10, "annotation": 0.05
        },
        "original_seed_radius_neighbors": RADIUS_NEIGHBORS,
        "original_seed_radius_quantile": RADIUS_QUANTILE,
        "original_seed_radius_by_class": limits,
        "max_new_per_class_per_round": MAX_PER_CLASS,
        "max_rounds": MAX_ROUNDS,
        "input_cells": int(adata.n_obs),
        "original_normal_seed_counts": {label: int((original == label).sum()) for label in NORMAL},
        "final_normal_counts": {label: int((result.normal_label == label).sum()) for label in NORMAL},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in (
            ("adata", args.adata), ("normal_seed_source", args.seed_file),
            ("cluster_evidence", args.cluster_evidence),
            ("normal_reference_code", Path(__file__)),
            ("b19_classifier", B19 / "reliability" / "seed_classifier.py"),
            ("b19_gates", B19 / "reliability" / "iterative_expansion.py"),
        )},
        "normal_reference_sha256": sha256(out / "normal_reference_no_official_labels.csv"),
    }
    (out / "generation_manifest_no_official_labels.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def evaluate(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "generation_manifest_no_official_labels.json").read_text())
    predictions = pd.read_csv(args.out_dir / "normal_reference_no_official_labels.csv",
                              dtype={"cell_id": str})
    if sha256(args.out_dir / "normal_reference_no_official_labels.csv") != manifest["normal_reference_sha256"]:
        raise ValueError("Generated predictions changed after fixation")
    truth = pd.read_csv(args.evaluation, dtype={"cell_id": str})[["cell_id", "truth"]]
    if predictions.cell_id.duplicated().any() or truth.cell_id.duplicated().any():
        raise ValueError("Cell IDs must be unique")
    if not set(truth.cell_id).issubset(set(predictions.cell_id)):
        raise ValueError("Evaluation cell IDs are not a subset of generated cells")
    merged = predictions.merge(truth, on="cell_id", how="left", validate="one_to_one")
    evaluable = merged.truth.notna() & merged.truth.isin((*NORMAL, "Neoplastic"))
    rows = []
    for stage, selected in (
        ("original_seeds", merged.seed_round.eq(0)),
        ("newly_expanded", merged.seed_round.gt(0)),
        ("all_normal_reference", merged.seed_round.ge(0)),
    ):
        for label in NORMAL:
            called = evaluable & selected & merged.normal_label.eq(label)
            correct = called & merged.truth.eq(label)
            total_truth = int((evaluable & merged.truth.eq(label)).sum())
            tp = int(correct.sum())
            n_called = int(called.sum())
            precision = tp / n_called if n_called else None
            coverage = tp / total_truth if total_truth else None
            f1 = 2 * tp / (n_called + total_truth) if n_called + total_truth else None
            rows.append({
                "stage": stage, "class": label, "truth_cells": total_truth,
                "called_evaluable": n_called, "correct": tp,
                "false_neoplastic_as_normal": int((called & merged.truth.eq("Neoplastic")).sum()),
                "precision": precision, "correct_class_coverage": coverage,
                "f1_unknown_as_missed": f1,
            })
    pd.DataFrame(rows).to_csv(args.out_dir / "normal_evaluation_only.csv", index=False)
    by_round = []
    for iteration in range(1, MAX_ROUNDS + 1):
        for label in NORMAL:
            selected = merged.seed_round.eq(iteration) & merged.normal_label.eq(label)
            called = selected & evaluable
            correct = called & merged.truth.eq(label)
            by_round.append({
                "round": iteration, "class": label,
                "selected_all": int(selected.sum()),
                "selected_evaluable": int(called.sum()),
                "correct": int(correct.sum()),
                "neoplastic_errors": int((called & merged.truth.eq("Neoplastic")).sum()),
                "precision": float(correct.sum() / called.sum()) if called.sum() else None,
                "selected_clusters": int(merged.loc[selected, "leiden_cluster"].nunique()),
            })
    pd.DataFrame(by_round).to_csv(args.out_dir / "normal_round_quality_evaluation_only.csv", index=False)
    diagnostics = {
        "evaluation_only_source": str(args.evaluation.resolve()),
        "evaluation_only_source_sha256": sha256(args.evaluation),
        "evaluated_cells": int(evaluable.sum()),
        "analysis_cells": int(len(merged)),
        "unmapped_reference_cells": int((~evaluable).sum()),
        "true_neoplastic_count": int((evaluable & merged.truth.eq("Neoplastic")).sum()),
        "true_neoplastic_called_any_normal": int((evaluable & merged.truth.eq("Neoplastic") & merged.seed_round.ge(0)).sum()),
    }
    (args.out_dir / "evaluation_provenance_only.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    screening = sub.add_parser("screen-seeds")
    screening.add_argument("--adata", type=Path, required=True)
    screening.add_argument("--tiered-seeds", type=Path, required=True)
    screening.add_argument("--out-dir", type=Path, required=True)
    generation = sub.add_parser("generate")
    generation.add_argument("--adata", type=Path, required=True)
    generation.add_argument("--seed-file", type=Path, required=True)
    generation.add_argument("--cluster-evidence", type=Path, required=True)
    generation.add_argument("--out-dir", type=Path, required=True)
    generation.add_argument("--schedule", choices=("fixed", "progressive"), required=True)
    generation.add_argument("--budget-mode", choices=("fixed", "growing"), required=True)
    assessment = sub.add_parser("evaluate")
    assessment.add_argument("--out-dir", type=Path, required=True)
    assessment.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "screen-seeds":
        screen_seeds(args)
    elif args.command == "generate":
        generate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
