#!/usr/bin/env python3
"""Compare paper-described initial-cell selectors using the same marker panel.

These are Python reproductions of Garnett's representative-cell rule and
Binary-SPA's clear-cell rule, not executions of either official R package.
Reference annotations are never opened by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


CLASSES = ("Connective_tissue", "Inflammatory", "Epithelial")
TISSUE = {"rep1": "Breast", "pancreas": "Pancreas"}


def load_cluster_module(repo: Path):
    source = repo / "methods/b2_16_ci_cnv_four_class_candidate_20261011/code/02_build_cluster_evidence.py"
    spec = importlib.util.spec_from_file_location("cluster_evidence_for_baselines", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def marker_panel(repo: Path, slide: str, genes: pd.Index, prevalence: np.ndarray, top_n: int):
    method = repo / "methods/b2_16_ci_cnv_four_class_candidate_20261011"
    module = load_cluster_module(repo)
    voting = module.load_original_voting_functions(method / "vendor/cdge")
    db = module.load_databases_compat(
        method / "resources/marker_databases", voting, TISSUE[slide],
        method / "resources/cell_ontology/cl.obo",
        method / "resources/cell_ontology/four_class_policy.json",
    )
    votes = voting["build_gene_votes"](db[0], db[2], db[1], db[3])
    categories = voting["VOTE_CATEGORIES"]
    ranked = {label: [] for label in CLASSES}
    for gene, frequency in zip(genes, prevalence):
        vector = votes.get(str(gene).upper())
        if vector is None or frequency < 0.0005:
            continue
        total = float(np.sum(vector))
        if total <= 0:
            continue
        winner = categories[int(np.argmax(vector))]
        if winner not in ranked:
            continue
        specificity = float(np.max(vector)) / total
        if specificity < 0.60:
            continue
        support = float(np.max(vector))
        score = specificity * np.log1p(support) * np.sqrt(float(frequency))
        ranked[winner].append((score, str(gene), specificity, support, float(frequency)))
    panel = {}
    details = {}
    for label in CLASSES:
        rows = sorted(ranked[label], key=lambda row: (-row[0], row[1]))[:top_n]
        if len(rows) < 5:
            raise ValueError(f"Only {len(rows)} usable markers for {label}")
        panel[label] = [row[1] for row in rows]
        details[label] = [
            {"gene": row[1], "rank_score": row[0], "specificity": row[2],
             "database_support": row[3], "detected_fraction": row[4]}
            for row in rows
        ]
    return panel, details


def garnett_style(matrix: np.ndarray, library: np.ndarray, membership: np.ndarray):
    positive = library > 0
    size_factor = np.ones_like(library, dtype=np.float64)
    size_factor[positive] = library[positive] / np.exp(np.mean(np.log(library[positive])))
    normalized = matrix / size_factor[:, None]
    gene_total = normalized.sum(axis=0)
    tf = normalized / np.maximum((library / size_factor)[:, None], 1.0)
    tfidf = tf * np.log1p(matrix.shape[0] / np.maximum(gene_total, 1.0))[None, :]
    cutoff = 0.25 * np.percentile(tfidf, 95, axis=0)
    tfidf[tfidf < cutoff[None, :]] = 0.0
    scores = tfidf @ membership
    thresholds = np.percentile(scores, 75, axis=0)
    qualifies = (scores >= thresholds[None, :]) & (scores > 0)
    unique = qualifies.sum(axis=1) == 1
    chosen = np.where(unique, qualifies.argmax(axis=1), -1)
    return chosen, scores, {label: float(thresholds[i]) for i, label in enumerate(CLASSES)}


def binary_spa_style(matrix: np.ndarray, membership: np.ndarray):
    counts = (matrix > 0).astype(np.float32) @ membership
    minimum = counts.min(axis=0)
    maximum = counts.max(axis=0)
    scores = (counts - minimum) / np.maximum(maximum - minimum, 1.0)
    order = np.argsort(scores, axis=1)
    top = order[:, -1]
    top_score = np.take_along_axis(scores, top[:, None], axis=1).ravel()
    runner_up = np.take_along_axis(scores, order[:, -2, None], axis=1).ravel()
    margin = top_score - runner_up
    chosen = np.where((margin >= 0.15) & (top_score > 0), top, -1)
    return chosen, scores, margin


def capped_indices(chosen, rank, fixed_n, forbidden, budgets):
    result = np.full(chosen.shape, -1, dtype=np.int8)
    eligible = (~forbidden) & (~fixed_n)
    available = {}
    for index, label in enumerate(CLASSES):
        candidates = np.flatnonzero((chosen == index) & eligible)
        order = np.argsort(-rank[candidates], kind="stable")
        available[label] = int(len(candidates))
        result[candidates[order[:budgets[label]]]] = index
    return result, available


def save_seeds(path: Path, cell_ids: np.ndarray, assigned: np.ndarray, fixed_n: np.ndarray):
    labels = np.full(len(cell_ids), "Unknown", dtype=object)
    labels[fixed_n] = "Neoplastic"
    for index, label in enumerate(CLASSES):
        labels[assigned == index] = label
    pd.DataFrame({"cell_id": cell_ids, "training_label": labels,
                  "is_initial_seed": labels != "Unknown"}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--slide", choices=TISSUE, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--markers-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = args.outputs_root / args.slide
    adata_path = base / "01_preparation/label_free_analysis.h5ad"
    original_path = base / "16_four_class_seeds/combined_initial_seeds_no_official_labels.csv"
    conflict_path = base / "06_bootstrap_evidence/neoplastic_evidence_table_no_official_labels.csv"
    data = ad.read_h5ad(adata_path, backed="r")
    try:
        cell_ids = np.asarray(data.obs_names.astype(str))
        genes = pd.Index(data.var_names.astype(str))
        counts = data.layers["counts"][:]
        if not sparse.issparse(counts):
            counts = sparse.csr_matrix(counts)
        counts = counts.tocsr().astype(np.float32)
    finally:
        data.file.close()
    if not genes.is_unique or len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("Gene and cell axes must be unique")
    prevalence = np.asarray(counts.getnnz(axis=0)).ravel() / counts.shape[0]
    panel, panel_details = marker_panel(args.repo, args.slide, genes, prevalence, args.markers_per_class)
    (args.out_dir / "marker_panel.json").write_text(json.dumps({
        "source": "current organ-filtered PanglaoDB/CellMarker ontology vote resources",
        "selection": "top measured class-specific genes; no reference labels",
        "classes": panel_details}, indent=2) + "\n")

    all_markers = sorted(set().union(*[set(genes) for genes in panel.values()]))
    positions = genes.get_indexer(all_markers)
    marker_matrix = counts[:, positions].toarray().astype(np.float64)
    membership = np.asarray([[float(gene in panel[label]) for label in CLASSES]
                             for gene in all_markers], dtype=np.float64)
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)

    original = pd.read_csv(original_path, dtype={"cell_id": str}).set_index("cell_id").reindex(cell_ids)
    if original.is_initial_seed.isna().any():
        raise ValueError("Original seed file and AnnData have different cell axes")
    original_labels = original.training_label.to_numpy()
    fixed_n = (original_labels == "Neoplastic") & original.is_initial_seed.to_numpy()
    budgets = {label: int(((original_labels == label) & original.is_initial_seed.to_numpy()).sum())
               for label in CLASSES}

    conflict = pd.read_csv(conflict_path, usecols=["cell_id", "variant_C_gene_plus_cnv"],
                           dtype={"cell_id": str}).set_index("cell_id").reindex(cell_ids)
    if conflict.variant_C_gene_plus_cnv.isna().any():
        raise ValueError("Conflict evidence and AnnData have different cell axes")
    conflict_flag = conflict.variant_C_gene_plus_cnv
    if conflict_flag.dtype == object:
        forbidden = conflict_flag.astype(str).str.lower().eq("true").to_numpy()
    else:
        forbidden = conflict_flag.astype(bool).to_numpy()
    # The original source withholds this tumor-supported conflict only for E.
    non_e_forbidden = np.zeros(len(cell_ids), dtype=bool)

    garnett_choice, garnett_scores, garnett_thresholds = garnett_style(marker_matrix, library, membership)
    binary_choice, binary_scores, binary_margin = binary_spa_style(marker_matrix, membership)
    methods = {
        "garnett_style": (garnett_choice, garnett_scores, np.max(garnett_scores, axis=1)),
        "binary_spa_style": (binary_choice, binary_scores, binary_margin),
    }
    audit = {"slide": args.slide, "method_status": "paper-rule Python reproductions; not official R packages",
             "input_cells": len(cell_ids), "fixed_neoplastic_seeds": int(fixed_n.sum()),
             "per_class_budgets": budgets, "garnett_75th_percentile": garnett_thresholds,
             "binary_spa_delta_cutoff": 0.15, "methods": {}}
    for name, (choice, scores, rank) in methods.items():
        method_forbidden = np.where(choice == CLASSES.index("Epithelial"), forbidden, non_e_forbidden)
        assigned, available = capped_indices(choice, rank, fixed_n, method_forbidden, budgets)
        save_seeds(args.out_dir / f"{name}_matched_seeds.csv", cell_ids, assigned, fixed_n)
        audit["methods"][name] = {"eligible_after_shared_conflict_screen": available,
                                 "selected": {label: int((assigned == i).sum())
                                              for i, label in enumerate(CLASSES)}}
    (args.out_dir / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
