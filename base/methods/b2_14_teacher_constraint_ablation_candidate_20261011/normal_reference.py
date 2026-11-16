#!/usr/bin/env python3
"""Expand normal lineages with staged cluster independence and Unknown rejection.

Generation never reads official annotations. Evaluation is a separate command.
This is a development experiment, not a frozen generalization method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

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
    select_spatially_balanced_additions,
    spatial_quadrants,
    supporting_cluster_evidence,
)
from reliability.seed_classifier import (  # noqa: E402
    fit_balanced_logistic_classifier,
)

NORMAL = ("Connective_tissue", "Inflammatory", "Epithelial")
MIN_CONFIDENCE = 0.80
MIN_MARGIN = 0.20
MIN_NEIGHBOR = 0.90
MIN_REFERENCE_NEIGHBORS = 5
MIN_REFERENCE_COVERAGE = 0.20
MIN_STRUCTURAL = 0.50
MIN_ANNOTATION = 0.50
RADIUS_NEIGHBORS = 5
RADIUS_QUANTILE = 0.99
MAX_PER_CLASS = 500
MAX_ROUNDS = 20
RELEASE_ROUNDS = 6
EMPTY_ROUND_PATIENCE = 3
SEED_CONSENSUS_NEIGHBORS = 10
CROSS_VOTE_MIN_CONFIDENCE = 0.90
CROSS_VOTE_MIN_SEED_CONSENSUS = 0.80
CROSS_MODEL_MIN_CONFIDENCE = 0.95
CROSS_MODEL_MIN_NEIGHBOR = 0.90
CROSS_MODEL_MIN_SEED_CONSENSUS = 0.90
CONFIDENCE_WEIGHT_START = 0.50
CONFIDENCE_WEIGHT_END = 0.80
BUDGET_GROWTH_PER_ROUND = 200


def budget_for_round(iteration: int, budget_mode: str) -> int:
    """Change only selection capacity, never evidence eligibility."""
    if budget_mode not in {"fixed", "growing", "late-growing"}:
        raise ValueError("Unknown budget mode")
    if not 1 <= iteration <= MAX_ROUNDS:
        raise ValueError("Iteration is outside the predeclared budget schedule")
    if budget_mode == "late-growing":
        return MAX_PER_CLASS + max(0, iteration - RELEASE_ROUNDS) * BUDGET_GROWTH_PER_ROUND
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
    fraction = (min(iteration, RELEASE_ROUNDS) - 1) / (RELEASE_ROUNDS - 1)
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


def cluster_support_components(
    clusters: np.ndarray, predicted: np.ndarray, table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strong-contradiction veto and matching marker-vote support."""
    if table["cluster"].astype(str).duplicated().any():
        raise ValueError("Cluster evidence has duplicate cluster IDs")
    if set(clusters) - set(table["cluster"].astype(str)):
        raise ValueError("Cluster evidence does not cover all cells")
    strong = strong_cluster_label_map(table)
    allowed = prediction_allowed_by_cluster_evidence(clusters, predicted, strong)
    supported = supporting_cluster_evidence(
        clusters, predicted, table, MIN_STRUCTURAL, MIN_ANNOTATION
    )
    return allowed, supported


def original_seed_consensus(
    features: np.ndarray, original_labels: np.ndarray, predicted: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Fraction of nearby *original* normal seeds agreeing with each candidate.

    Original seeds, their labels, and the feature scale are fixed across rounds.
    This is lineage support, not evidence that a candidate is benign.
    """
    n = len(features)
    if any(np.asarray(values).shape != (n,) for values in (original_labels, predicted, candidates)):
        raise ValueError("Seed consensus inputs must share the cell axis")
    result = np.zeros(n, dtype=np.float32)
    positions = np.flatnonzero(candidates)
    if not len(positions):
        return result
    seeds = np.flatnonzero(np.isin(original_labels, NORMAL))
    if len(seeds) <= SEED_CONSENSUS_NEIGHBORS:
        raise ValueError("Too few original seeds to check cross-cluster candidates")
    mean = features[seeds].mean(axis=0)
    scale = np.maximum(features[seeds].std(axis=0), 1e-6)
    normalized = (features.astype(np.float64) - mean) / scale
    neighbors = NearestNeighbors(n_neighbors=SEED_CONSENSUS_NEIGHBORS, n_jobs=1)
    neighbors.fit(normalized[seeds])
    indices = neighbors.kneighbors(normalized[positions], return_distance=False)
    result[positions] = np.mean(
        original_labels[seeds[indices]] == predicted[positions, None], axis=1
    )
    return result


def admission_routes(
    iteration: int, unlabelled: np.ndarray, predicted: np.ndarray,
    confidence: np.ndarray, margin: np.ndarray, neighbor: np.ndarray,
    radius_ok: np.ndarray, cluster_allowed: np.ndarray,
    cluster_supported: np.ndarray, anchored: np.ndarray,
    seed_consensus: np.ndarray,
) -> np.ndarray:
    """Admit normal predictions in three predeclared stages; reject the rest."""
    if not 1 <= iteration <= MAX_ROUNDS:
        raise ValueError("Iteration outside the predeclared release schedule")
    n = len(unlabelled)
    if any(np.asarray(values).shape != (n,) for values in (
        predicted, confidence, margin, neighbor, radius_ok, cluster_allowed,
        cluster_supported, anchored, seed_consensus,
    )):
        raise ValueError("Admission inputs must share the cell axis")
    routes = np.full(n, "Unknown", dtype=object)
    base = (
        unlabelled & np.isin(predicted, NORMAL) & radius_ok & cluster_allowed
        & (confidence >= MIN_CONFIDENCE) & (margin >= MIN_MARGIN)
        & (neighbor >= MIN_NEIGHBOR)
    )
    routes[base & anchored & cluster_supported] = "anchored"
    cross = base & ~anchored
    if iteration >= 3:
        voted = (
            cross & cluster_supported & (confidence >= CROSS_VOTE_MIN_CONFIDENCE)
            & (seed_consensus >= CROSS_VOTE_MIN_SEED_CONSENSUS)
        )
        routes[voted] = "cross_cluster_vote"
    if iteration >= 5:
        model = (
            cross & ~cluster_supported & (confidence >= CROSS_MODEL_MIN_CONFIDENCE)
            & (neighbor >= CROSS_MODEL_MIN_NEIGHBOR)
            & (seed_consensus >= CROSS_MODEL_MIN_SEED_CONSENSUS)
        )
        routes[model] = "cross_cluster_model"
    return routes


def validated_reference_graph(adata: sc.AnnData) -> sparse.csr_matrix:
    """Use actual expression-neighbor edges, never links only to seed cells."""
    if "connectivities" not in adata.obsp:
        raise ValueError("Normal expansion requires the expression-neighbor graph")
    graph = sparse.csr_matrix(adata.obsp["connectivities"], dtype=float).copy()
    if graph.shape != (adata.n_obs, adata.n_obs):
        raise ValueError("Reference graph must share the analysis cell axis")
    if not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise ValueError("Reference graph edge weights must be finite and nonnegative")
    graph.sum_duplicates()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[:] = 1.0
    return graph


def reference_neighbor_support(
    graph: sparse.csr_matrix, labels: np.ndarray, predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count only labels accepted before this round; Unknown neighbors abstain.

    Agreement is the proposed class's fraction among labeled neighbors.
    Coverage is labeled neighbors / all actual expression neighbors. Keeping
    both quantities prevents one distant or isolated seed from certifying a
    whole unlabeled population.
    """
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    n = len(labels)
    if predicted.shape != (n,) or graph.shape != (n, n):
        raise ValueError("Reference-support inputs must share the cell axis")
    if set(labels) - set(NORMAL) - {"Unknown"} or set(predicted) - set(NORMAL):
        raise ValueError("Reference support requires only normal labels and Unknown")
    counts = graph @ np.column_stack([labels == label for label in NORMAL]).astype(float)
    known = counts.sum(axis=1)
    degrees = np.diff(graph.indptr).astype(float)
    indices = np.array([NORMAL.index(label) for label in predicted])
    same = counts[np.arange(n), indices]
    agreement = same / np.maximum(known, 1)
    coverage = known / np.maximum(degrees, 1)
    return agreement, coverage, known


def expand_normal_reference(
    adata: sc.AnnData, original_labels: np.ndarray, cluster_table: pd.DataFrame,
    schedule: str, budget_mode: str, classifier_update: str = "accepted",
    reference_update: str = "accepted",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Iterate with fixed original-seed novelty radius and selective gates."""
    if len(original_labels) != adata.n_obs:
        raise ValueError("Seed axis does not match AnnData")
    if classifier_update not in {"accepted", "original"} or reference_update not in {"accepted", "original"}:
        raise ValueError("Feedback modes must be accepted or original")
    features = expression_features(adata, 30, "X_pca")
    graph = validated_reference_graph(adata)
    clusters = adata.obs["leiden_res_2.00"].astype(str).to_numpy()
    cluster_scores = cluster_table.set_index("cluster").reindex(clusters)
    if cluster_scores[["structural_reliability_score", "annotation_confidence_score"]].isna().any().any():
        raise ValueError("Missing cluster scores on the analysis cell axis")
    structure = cluster_scores["structural_reliability_score"].to_numpy(dtype=float)
    annotation = cluster_scores["annotation_confidence_score"].to_numpy(dtype=float)
    regions = spatial_quadrants(np.asarray(adata.obsm["spatial"], dtype=float))
    labels = original_labels.copy()
    round_added = np.where(labels == "Unknown", -1, 0)
    accepted_route = np.where(round_added == 0, "original_seed", "Unknown").astype(object)
    waterfalls: list[dict[str, object]] = []
    radius_limits: dict[str, float] = {}
    empty_rounds = 0
    fixed_prediction = None

    for iteration in range(1, MAX_ROUNDS + 1):
        train = np.flatnonzero(round_added == 0 if classifier_update == "original" else round_added >= 0)
        if classifier_update == "original" and fixed_prediction is not None:
            predicted, confidence, margin, classes = fixed_prediction
        else:
            predicted, confidence, margin, classes = fit_balanced_logistic_classifier(features, train, labels)
            if classifier_update == "original":
                fixed_prediction = (predicted, confidence, margin, classes)
        if set(classes) != set(NORMAL):
            raise ValueError("Normal-only training gained or lost a class")
        neighbor, reference_coverage, reference_count = reference_neighbor_support(
            graph, original_labels if reference_update == "original" else labels, predicted
        )
        prediction_neighbor, _, _ = reference_neighbor_support(graph, predicted, predicted)
        reference_ok = (
            (reference_count >= MIN_REFERENCE_NEIGHBORS)
            & (reference_coverage >= MIN_REFERENCE_COVERAGE)
            & (neighbor >= MIN_NEIGHBOR)
        )
        allowed, supported = cluster_support_components(clusters, predicted, cluster_table)
        anchored = anchored_cluster_mask(clusters, predicted, original_labels)
        unlabelled = round_added < 0
        potential = anchored & supported
        if iteration >= 3:
            potential |= ~anchored & supported & (confidence >= CROSS_VOTE_MIN_CONFIDENCE)
        if iteration >= 5:
            potential |= (
                ~anchored & ~supported & (confidence >= CROSS_MODEL_MIN_CONFIDENCE)
                & (neighbor >= CROSS_MODEL_MIN_NEIGHBOR)
            )
        before_radius = (
            unlabelled & (confidence >= MIN_CONFIDENCE) & (margin >= MIN_MARGIN)
            & reference_ok & allowed & potential
        )
        radius_ok, limits = fixed_seed_radius(features, original_labels, predicted, before_radius)
        if not radius_limits:
            radius_limits = limits
        cross_precheck = (
            before_radius & radius_ok & ~anchored & (iteration >= 3)
            & (confidence >= CROSS_VOTE_MIN_CONFIDENCE)
        )
        seed_consensus = original_seed_consensus(
            features, original_labels, predicted, cross_precheck
        )
        routes = admission_routes(
            iteration, unlabelled, predicted, confidence, margin, neighbor,
            radius_ok, allowed, supported, anchored, seed_consensus,
        )
        eligible = (routes != "Unknown") & reference_ok
        rank = combined_rank(
            confidence, margin, neighbor * reference_coverage, structure, annotation,
            iteration, schedule,
        )
        rank += np.select(
            [routes == "anchored", routes == "cross_cluster_vote"], [2.0, 1.0], default=0.0
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
            row["after_confidence"] = int((subset & (confidence >= MIN_CONFIDENCE)).sum())
            row["after_prediction_neighbor_audit_only"] = int((subset & (confidence >= MIN_CONFIDENCE) & (prediction_neighbor >= .7)).sum())
            reference_subset = subset & (confidence >= MIN_CONFIDENCE)
            for name, gate in (
                ("reference_neighbor_count", reference_count >= MIN_REFERENCE_NEIGHBORS),
                ("reference_coverage", reference_coverage >= MIN_REFERENCE_COVERAGE),
                ("reference_agreement", neighbor >= MIN_NEIGHBOR),
            ):
                reference_subset &= gate
                row["after_" + name] = int(reference_subset.sum())
            row["after_seed_radius"] = int((subset & before_radius & radius_ok).sum())
            for route in ("anchored", "cross_cluster_vote", "cross_cluster_model"):
                row["eligible_" + route] = int((subset & (routes == route)).sum())
                row["selected_" + route] = int(
                    ((predicted[additions] == label) & (routes[additions] == route)).sum()
                )
            row["selected_after_spatial_budget"] = int((predicted[additions] == label).sum())
            waterfalls.append(row)
        print(json.dumps({"round": iteration, "eligible": int(eligible.sum()),
                          "added": int(len(additions)),
                          "training_count": int(len(train)),
                          "normal_reference_total": int((round_added >= 0).sum() + len(additions))}),
              flush=True)
        empty_rounds = empty_rounds + 1 if not len(additions) and iteration >= RELEASE_ROUNDS else 0
        if empty_rounds >= EMPTY_ROUND_PATIENCE:
            break
        if not len(additions):
            continue
        labels[additions] = predicted[additions]
        round_added[additions] = iteration
        accepted_route[additions] = routes[additions]

    result = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "leiden_cluster": clusters,
        "normal_label": labels,
        "seed_round": round_added,
        "admission_route": accepted_route,
        "last_model_proposal": predicted,
        "last_reference_neighbor_count": reference_count.astype(int),
        "last_reference_coverage": reference_coverage,
        "last_reference_agreement": neighbor,
        "is_original_normal_seed": round_added == 0,
    })
    return result, pd.DataFrame(waterfalls), radius_limits


def pseudo_weight_ratio(iteration: int) -> float:
    """Cap pseudo-label mass per class relative to original seeds."""
    if not 1 <= iteration <= MAX_ROUNDS:
        raise ValueError("Iteration outside declared schedule")
    return 0.25 + 0.75 * (iteration - 1) / (MAX_ROUNDS - 1)


def anchored_training_weights(original: np.ndarray, labels: np.ndarray, iteration: int) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=float)
    seed_total = int(np.isin(original, NORMAL).sum())
    ratio = pseudo_weight_ratio(iteration)
    for label in NORMAL:
        seeds = original == label
        pseudo = (original == "Unknown") & (labels == label)
        if not seeds.any():
            raise ValueError("Each class needs original seeds")
        mass = seed_total / len(NORMAL)
        weights[seeds] = mass / seeds.sum()
        if pseudo.any():
            weights[pseudo] = ratio * mass / pseudo.sum()
    return weights


def fitted_probabilities(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    selected = weights > 0
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    model.fit(features[selected], labels[selected], sample_weight=weights[selected])
    if set(model.classes_) != set(NORMAL):
        raise ValueError("Classifier class contract changed")
    probabilities = model.predict_proba(features)
    return probabilities[:, [list(model.classes_).index(label) for label in NORMAL]]


def probability_decisions(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(probabilities, axis=1)
    return (np.asarray(NORMAL)[probabilities.argmax(axis=1)], ordered[:, -1],
            ordered[:, -1] - ordered[:, -2])


def revoke_provisional_labels(
    original: np.ndarray, labels: np.ndarray, predicted: np.ndarray,
    teacher: np.ndarray, confidence: np.ndarray, teacher_confidence: np.ndarray,
    radius: np.ndarray, teacher_required: bool = True,
) -> np.ndarray:
    """Never retract original seeds; inconsistent provisional labels abstain."""
    return ((original == "Unknown") & (labels != "Unknown") & (
        (labels != predicted) | (confidence < MIN_CONFIDENCE) | ~radius
        | (teacher_required & ((labels != teacher) | (teacher_confidence < MIN_CONFIDENCE)))
    ))


def strict_reference_mask(original: np.ndarray, labels: np.ndarray, predicted: np.ndarray,
                          teacher: np.ndarray, confidence: np.ndarray,
                          teacher_confidence: np.ndarray, stable: np.ndarray,
                          neighbor: np.ndarray, coverage: np.ndarray,
                          count: np.ndarray, consensus: np.ndarray,
                          radius: np.ndarray) -> np.ndarray:
    """Conservative expression reference candidates, NOT certified diploid cells."""
    return (np.isin(labels, NORMAL) & (labels == predicted) & (labels == teacher)
            & (confidence >= .95) & (teacher_confidence >= .95) & (stable >= 3)
            & (neighbor >= .95) & (coverage >= MIN_REFERENCE_COVERAGE)
            & (count >= MIN_REFERENCE_NEIGHBORS) & (consensus >= .9) & radius)


def cell_admission_mask(base: np.ndarray, confidence: np.ndarray,
                        stable: np.ndarray, consensus: np.ndarray) -> np.ndarray:
    """Late admission has no cluster ID, vote, score or ranking input."""
    return base & (confidence >= CROSS_MODEL_MIN_CONFIDENCE) & (stable >= 2) & (consensus >= .8)


def prediction_anchor_mask(predicted: np.ndarray, confidence: np.ndarray,
                           teacher: np.ndarray, teacher_confidence: np.ndarray,
                           teacher_mode: str) -> np.ndarray:
    if teacher_mode not in {"fixed", "off"}:
        raise ValueError("Unknown teacher mode")
    result = confidence >= MIN_CONFIDENCE
    if teacher_mode == "fixed":
        result &= (predicted == teacher) & (teacher_confidence >= MIN_CONFIDENCE)
    return result


def expand_anchored_reference(adata: sc.AnnData, original: np.ndarray,
                             table: pd.DataFrame, schedule: str, budget_mode: str,
                             teacher_mode: str = "fixed",
                             ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    features = expression_features(adata, 30, "X_pca").astype(float)
    seed_mask = np.isin(original, NORMAL)
    mean = features[seed_mask].mean(axis=0)
    scale = np.maximum(features[seed_mask].std(axis=0), 1e-6)
    x = (features - mean) / scale
    graph = validated_reference_graph(adata)
    clusters = adata.obs["leiden_res_2.00"].astype(str).to_numpy()
    regions = spatial_quadrants(np.asarray(adata.obsm["spatial"], dtype=float))
    scores = table.set_index("cluster").reindex(clusters)
    score_columns = ["structural_reliability_score", "annotation_confidence_score"]
    if scores[score_columns].isna().any().any():
        raise ValueError("Missing cluster scores")
    structure, annotation = [scores[column].to_numpy(float) for column in score_columns]
    teacher_p = fitted_probabilities(x, original, anchored_training_weights(original, original, 1))
    teacher, teacher_confidence, _ = probability_decisions(teacher_p)
    teacher_radius, limits = fixed_seed_radius(features, original, teacher, np.ones(len(original), bool))
    teacher_consensus = original_seed_consensus(features, original, teacher, np.ones(len(original), bool))
    # Cache support for each possible student class without changing seed geometry.
    support_by_class = {}
    if teacher_mode == "off":
        for label in NORMAL:
            proposal = np.full(len(original), label)
            support_by_class[label] = (
                fixed_seed_radius(features, original, proposal, np.ones(len(original), bool))[0],
                original_seed_consensus(features, original, proposal, np.ones(len(original), bool)),
            )
    def proposal_support(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if teacher_mode == "fixed":
            return teacher_radius, teacher_consensus
        radius, consensus = np.zeros(len(original), bool), np.zeros(len(original))
        for label, (class_radius, class_consensus) in support_by_class.items():
            selected = prediction == label
            radius[selected], consensus[selected] = class_radius[selected], class_consensus[selected]
        return radius, consensus
    labels = original.copy()
    rounds = np.where(seed_mask, 0, -1)
    routes = np.where(seed_mask, "original_seed", "Unknown").astype(object)
    stable = np.zeros(len(labels), dtype=int)
    previous_prediction = np.full(len(labels), "Unknown", dtype=object)
    retractions = np.zeros(len(labels), dtype=int)
    cooldown_until = np.zeros(len(labels), dtype=int)
    waterfall, events = [], []
    empty = 0
    for iteration in range(1, MAX_ROUNDS + 1):
        weights = anchored_training_weights(original, labels, iteration)
        p = fitted_probabilities(x, labels, weights)
        predicted, confidence, margin = probability_decisions(p)
        radius, consensus = proposal_support(predicted)
        revoked = revoke_provisional_labels(original, labels, predicted, teacher, confidence,
                                             teacher_confidence, radius, teacher_mode == "fixed")
        old_labels = labels.copy()
        for position in np.flatnonzero(revoked):
            events.append({"round": iteration, "cell_id": str(adata.obs_names[position]),
                           "event": "retracted", "label": str(labels[position])})
        labels[revoked] = "Unknown"
        rounds[revoked] = -1
        routes[revoked] = "Unknown"
        retractions[revoked] += 1
        cooldown_until[revoked] = iteration + 2
        if revoked.any():
            weights = anchored_training_weights(original, labels, iteration)
            p = fitted_probabilities(x, labels, weights)
            predicted, confidence, margin = probability_decisions(p)
            radius, consensus = proposal_support(predicted)
            # Refit can change more decisions; these labels abstain before propagation.
            more = revoke_provisional_labels(original, labels, predicted, teacher, confidence,
                                            teacher_confidence, radius, teacher_mode == "fixed")
            for position in np.flatnonzero(more):
                events.append({"round": iteration, "cell_id": str(adata.obs_names[position]),
                               "event": "retracted_after_refit", "label": str(labels[position])})
            labels[more], rounds[more], routes[more] = "Unknown", -1, "Unknown"
            retractions[more] += 1
            cooldown_until[more] = iteration + 2
            revoked |= more
        agreed = prediction_anchor_mask(predicted, confidence, teacher, teacher_confidence, teacher_mode)
        stable = np.where(agreed, np.where(predicted == previous_prediction, stable + 1, 1), 0)
        previous_prediction = predicted.copy()
        neighbor, coverage, count = reference_neighbor_support(graph, labels, predicted)
        reference_ok = ((count >= MIN_REFERENCE_NEIGHBORS) & (coverage >= MIN_REFERENCE_COVERAGE)
                        & (neighbor >= MIN_NEIGHBOR))
        unlabelled = (labels == "Unknown") & (iteration > cooldown_until)
        base = (unlabelled & agreed & radius & reference_ok
                & (margin >= MIN_MARGIN))
        if iteration <= RELEASE_ROUNDS:
            allowed, supported = cluster_support_components(clusters, predicted, table)
            anchored = anchored_cluster_mask(clusters, predicted, original)
            routes_now = admission_routes(iteration, unlabelled, predicted, confidence, margin,
                                         neighbor, radius, allowed, supported, anchored,
                                         consensus)
            eligible = base & (routes_now != "Unknown")
            rank = combined_rank(confidence, margin, neighbor * coverage, structure, annotation,
                                 iteration, schedule)
        else:
            eligible = cell_admission_mask(base, confidence, stable, consensus)
            routes_now = np.where(eligible, "cell_classifier_anchored", "Unknown")
            rank = .8 * confidence + .1 * margin + .1 * neighbor * coverage
        additions = select_with_spatial_overflow(eligible, predicted, confidence, margin, regions,
                                                 rank, budget_for_round(iteration, budget_mode))
        for label in NORMAL:
            waterfall.append({"round": iteration, "class": label,
                              "eligible": int((eligible & (predicted == label)).sum()),
                              "selected_after_spatial_budget": int((predicted[additions] == label).sum()),
                              "retracted": int((revoked & (old_labels == label)).sum()),
                              "pseudo_to_seed_weight_ratio": pseudo_weight_ratio(iteration),
                              "class_budget": budget_for_round(iteration, budget_mode)})
        labels[additions], rounds[additions], routes[additions] = predicted[additions], iteration, routes_now[additions]
        for position in additions:
            events.append({"round": iteration, "cell_id": str(adata.obs_names[position]),
                           "event": "added", "label": str(labels[position])})
        print(json.dumps({"round": iteration, "added": len(additions), "retracted": int(revoked.sum()),
                          "normal_total": int(np.isin(labels, NORMAL).sum()),
                          "pseudo_to_seed_weight_ratio": pseudo_weight_ratio(iteration)}), flush=True)
        empty = empty + 1 if iteration >= RELEASE_ROUNDS and not len(additions) and not revoked.any() else 0
        if empty >= EMPTY_ROUND_PATIENCE:
            break
    # Assess strict candidates only against prior accepted neighbors, never same-round additions.
    strict = strict_reference_mask(original, labels, predicted, teacher, confidence,
                                   teacher_confidence, stable, neighbor, coverage, count,
                                   teacher_consensus, teacher_radius)
    result = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "leiden_cluster": clusters,
                           "normal_label": labels, "seed_round": rounds, "admission_route": routes,
                           "last_model_proposal": predicted, "teacher_proposal": teacher,
                           "teacher_confidence": teacher_confidence, "student_confidence": confidence,
                           "stable_rounds": stable, "retraction_count": retractions,
                           "is_original_normal_seed": seed_mask,
                           "is_strict_cnv_reference_candidate": strict})
    result.attrs["events"] = events
    result.attrs["scaler_mean"] = mean.tolist()
    result.attrs["scaler_scale"] = scale.tolist()
    return result, pd.DataFrame(waterfall), limits


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
    result, waterfall, limits = expand_anchored_reference(
        adata, original, table, args.schedule, args.budget_mode, args.teacher_mode,
    )
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "normal_reference_no_official_labels.csv", index=False)
    waterfall.to_csv(out / "normal_gate_waterfall_no_official_labels.csv", index=False)
    pd.DataFrame(result.attrs["events"], columns=["round", "cell_id", "event", "label"]).to_csv(
        out / "label_events_no_official_labels.csv", index=False)
    strict_path = out / "strict_cnv_reference_candidates_no_official_labels.csv"
    result.loc[result.is_strict_cnv_reference_candidate].to_csv(strict_path, index=False)
    config = {
        "method": "B2.14-teacher-constraint-ablation-development",
        "teacher_mode": args.teacher_mode,
        "classifier_update": "weighted adaptive student; original seed model " + ("constraint" if args.teacher_mode == "fixed" else "audit and strict-reference only"),
        "reference_update": "accepted reversible labels from prior rounds",
        "fixed_seed_scaler": True,
        "pseudo_training_mass_per_class": "0.25 to 1.0 times original seed mass over 20 rounds",
        "label_revalidation": "student agreement, confidence>=0.8, seed radius; teacher constraint only in fixed arm; two-round cooldown",
        "strict_reference_policy": "candidate only, not certified normal/diploid; no CNV run",
        "strict_reference_gates": {"teacher_confidence": .95, "student_confidence": .95,
            "stable_rounds": 3, "neighbor_agreement": .95, "original_seed_consensus": .9,
            "min_neighbors": MIN_REFERENCE_NEIGHBORS, "min_neighbor_coverage": MIN_REFERENCE_COVERAGE},
        "original_seed_scaler_mean": result.attrs["scaler_mean"],
        "original_seed_scaler_scale": result.attrs["scaler_scale"],
        "strict_reference_sha256": sha256(strict_path),
        "official_labels_policy": "not_loaded_during_generation",
        "normal_classes": NORMAL,
        "classifier": "weighted logistic C=1, fixed original-seed scaling, 30 PCA components",
        "confidence": MIN_CONFIDENCE,
        "margin": MIN_MARGIN,
        "neighbor_agreement": MIN_NEIGHBOR,
        "neighbor_support_source": "prior accepted labels only, after retractions; self edges removed",
        "min_labeled_expression_neighbors": MIN_REFERENCE_NEIGHBORS,
        "min_labeled_expression_neighbor_coverage": MIN_REFERENCE_COVERAGE,
        "prediction_neighbor_agreement": "audit_only_not_an_acceptance_gate",
        "min_structural": MIN_STRUCTURAL,
        "min_annotation": MIN_ANNOTATION,
        "unknown_policy": "three normal scores plus explicit abstention; Unknown is never fitted as a class",
        "cluster_release_schedule": {
            "rounds_1_2": "matching vote and original same-class seed cluster",
            "rounds_3_4": "also matching-vote cross-cluster proposals with actual normal-reference neighbors and seed-neighbor support",
            "rounds_5_6": "existing weak-vote cross-cluster path",
            "rounds_7_20": "no cluster eligibility/rank; student>=0.95, stability>=2, seed consensus>=0.8, fixed radius, actual reference neighbors; teacher agreement only in fixed arm",
        },
        "seed_consensus_neighbors": SEED_CONSENSUS_NEIGHBORS,
        "cross_vote_confidence": CROSS_VOTE_MIN_CONFIDENCE,
        "cross_vote_seed_consensus": CROSS_VOTE_MIN_SEED_CONSENSUS,
        "cross_model_confidence": CROSS_MODEL_MIN_CONFIDENCE,
        "cross_model_neighbor_agreement": CROSS_MODEL_MIN_NEIGHBOR,
        "cross_model_seed_consensus": CROSS_MODEL_MIN_SEED_CONSENSUS,
        "confidence_weight_schedule": args.schedule,
        "confidence_weight_start": CONFIDENCE_WEIGHT_START,
        "confidence_weight_end_if_progressive": CONFIDENCE_WEIGHT_END,
        "confidence_schedule_saturates_at_round": RELEASE_ROUNDS,
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
        "completed_rounds": int(waterfall["round"].max()),
        "empty_round_patience_after_full_release": EMPTY_ROUND_PATIENCE,
        "stop_reason": "max_rounds" if int(waterfall["round"].max()) == MAX_ROUNDS else "three_empty_fully_released_rounds",
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
    for item in manifest["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Generation input or dependency changed after fixation")
    if sha256(args.out_dir / "strict_cnv_reference_candidates_no_official_labels.csv") != manifest["strict_reference_sha256"]:
        raise ValueError("Strict reference candidates changed after fixation")
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
    completed = manifest.get("completed_rounds", manifest["max_rounds"])
    # Final retained labels cannot reconstruct historical snapshots after retractions.
    snapshots = [("strict_cnv_reference_candidates", boolean_flags(merged["is_strict_cnv_reference_candidate"]))]
    for stage, selected in [
        ("original_seeds", merged.seed_round.eq(0)),
        ("newly_expanded", merged.seed_round.gt(0)),
        ("all_normal_reference", merged.seed_round.ge(0)),
    ] + snapshots:
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
                "effective_round": min(int(stage.rsplit("_", 1)[-1]), completed) if stage.startswith("through_round_") else None,
                "called_evaluable": n_called, "correct": tp,
                "false_neoplastic_as_normal": int((called & merged.truth.eq("Neoplastic")).sum()),
                "precision": precision, "correct_class_coverage": coverage,
                "f1_unknown_as_missed": f1,
            })
    pd.DataFrame(rows).to_csv(args.out_dir / "normal_evaluation_only.csv", index=False)
    by_round = []
    for iteration in range(1, completed + 1):
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


def run_regression(args: argparse.Namespace) -> None:
    """Sequential generation then evaluation, with exclusive output ownership."""
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=False)
    code_hash = sha256(Path(__file__))
    state = {"method_sha256": code_hash, "teacher_mode": args.teacher_mode, "status": "running", "slides": {}}
    def save_state() -> None:
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(root / "status.json")
    save_state()
    try:
        for slide in ("rep1", "pancreas"):
            baseline = PROJECT / "outputs" / slide / "b2_12_feedback_audit_20261011_frozen_classifier" / "normal_reference"
            manifest = json.loads((baseline / "generation_manifest_no_official_labels.json").read_text())
            inputs = manifest["inputs"]
            for name in ("adata", "normal_seed_source", "cluster_evidence"):
                if sha256(Path(inputs[name]["path"])) != inputs[name]["sha256"]:
                    raise ValueError(f"Baseline input changed: {slide}/{name}")
            if sha256(Path(__file__)) != code_hash:
                raise ValueError("Running candidate modified")
            out = root / slide / "normal_reference"
            state["slides"][slide] = "generating"
            save_state()
            command = [sys.executable, str(Path(__file__).resolve()), "generate",
                       "--adata", inputs["adata"]["path"], "--seed-file", inputs["normal_seed_source"]["path"],
                       "--cluster-evidence", inputs["cluster_evidence"]["path"], "--out-dir", str(out),
                       "--schedule", "progressive", "--budget-mode", "late-growing",
                       "--teacher-mode", args.teacher_mode]
            with (root / (slide + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                state["slides"][slide] = "evaluating"
                save_state()
                provenance = json.loads((baseline / "evaluation_provenance_only.json").read_text())
                evaluation = Path(provenance["evaluation_only_source"])
                if sha256(evaluation) != provenance["evaluation_only_source_sha256"]:
                    raise ValueError("Fixed evaluation source changed")
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "evaluate",
                                "--out-dir", str(out), "--evaluation", str(evaluation)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            state["slides"][slide] = "completed"
            save_state()
        state["status"] = "completed"
    except Exception as error:
        state["status"], state["error"] = "failed", repr(error)
        raise
    finally:
        save_state()


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
    generation.add_argument("--budget-mode", choices=("fixed", "growing", "late-growing"), required=True)
    generation.add_argument("--teacher-mode", choices=("fixed", "off"), required=True)
    assessment = sub.add_parser("evaluate")
    assessment.add_argument("--out-dir", type=Path, required=True)
    assessment.add_argument("--evaluation", type=Path, required=True)
    regression = sub.add_parser("regression")
    regression.add_argument("--out-dir", type=Path, required=True)
    regression.add_argument("--teacher-mode", choices=("fixed", "off"), required=True)
    args = parser.parse_args()
    if args.command == "screen-seeds":
        screen_seeds(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        run_regression(args)


if __name__ == "__main__":
    main()
