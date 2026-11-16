#!/usr/bin/env python3
"""Withhold E seeds lacking Binary-SPA-style cell-level marker support.

This runs after the original C/I/E seed merge and before normal expansion.
It reads no evaluation/reference labels. C, I and bootstrap N seeds are fixed.
The same marker panel and Binary-SPA-style equation as the earlier diagnostic
are used; this is not the full official Binary-SPA pipeline.
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

from run_selection import CLASSES, binary_spa_style


def marker_panel(method_dir: Path, tissue: str, genes: pd.Index,
                 prevalence: np.ndarray, top_n: int) -> tuple[dict, dict]:
    source = method_dir / "code/02_build_cluster_evidence.py"
    spec = importlib.util.spec_from_file_location("cell_gate_cluster_evidence", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    voting = module.load_original_voting_functions(method_dir / "vendor/cdge")
    db = module.load_databases_compat(
        method_dir / "resources/marker_databases", voting, tissue,
        method_dir / "resources/cell_ontology/cl.obo",
        method_dir / "resources/cell_ontology/four_class_policy.json",
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
    panel, details = {}, {}
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


def selected_seed_flags(frame: pd.DataFrame) -> np.ndarray:
    values = frame["is_initial_seed"]
    if values.dtype == bool:
        return values.to_numpy()
    parsed = values.astype(str).str.lower()
    if not parsed.isin(("true", "false")).all():
        raise ValueError("Invalid is_initial_seed values")
    return parsed.eq("true").to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--tissue", required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--markers-per-class", type=int, default=20)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")

    data = ad.read_h5ad(args.adata, backed="r")
    try:
        cell_ids = np.asarray(data.obs_names.astype(str))
        genes = pd.Index(data.var_names.astype(str))
        counts = data.layers["counts"][:]
        if not sparse.issparse(counts):
            counts = sparse.csr_matrix(counts)
        counts = counts.tocsr().astype(np.float32)
    finally:
        data.file.close()
    if not genes.is_unique or not pd.Index(cell_ids).is_unique:
        raise ValueError("Duplicate gene or cell IDs")
    seeds = pd.read_csv(args.seed_file, dtype={"cell_id": str})
    if (len(seeds) != len(cell_ids) or seeds.cell_id.duplicated().any()
            or not np.array_equal(seeds.cell_id.to_numpy(), cell_ids)):
        raise ValueError("Seed table and AnnData cell axes differ")
    selected = selected_seed_flags(seeds)
    old_label = seeds.training_label.astype(str).to_numpy()
    original_e = selected & (old_label == "Epithelial")
    if not original_e.any():
        raise ValueError("No E seeds to filter")

    prevalence = np.asarray(counts.getnnz(axis=0)).ravel() / counts.shape[0]
    panel, details = marker_panel(args.method_dir, args.tissue, genes,
                                  prevalence, args.markers_per_class)
    all_markers = sorted(set().union(*[set(values) for values in panel.values()]))
    locations = genes.get_indexer(all_markers)
    if (locations < 0).any():
        raise ValueError("Marker panel and expression matrix disagree")
    matrix = counts[:, locations].toarray().astype(np.float64)
    membership = np.asarray(
        [[float(gene in panel[label]) for label in CLASSES] for gene in all_markers],
        dtype=np.float64,
    )
    choice, scores, margin = binary_spa_style(matrix, membership)
    accepted = original_e & (choice == CLASSES.index("Epithelial"))
    rejected = original_e & ~accepted
    if not accepted.any():
        raise ValueError("Binary-SPA-style rule removed every E seed")

    result = seeds.copy()
    result.loc[rejected, "training_label"] = "Unknown"
    result.loc[rejected, "is_initial_seed"] = False
    if "seed_source" in result:
        result.loc[rejected, "seed_source"] = "withheld_binary_spa_e_marker"
    unchanged = ~original_e
    if not result.loc[unchanged].equals(seeds.loc[unchanged]):
        raise AssertionError("A non-E cell was modified")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "combined_initial_seeds_no_official_labels.csv", index=False)
    pd.DataFrame({
        "cell_id": cell_ids[original_e],
        "accepted": accepted[original_e],
        "binary_choice": np.where(choice[original_e] < 0, "Unknown",
                                  np.asarray(CLASSES)[np.maximum(choice[original_e], 0)]),
        "binary_e_score": scores[original_e, CLASSES.index("Epithelial")],
        "binary_margin": margin[original_e],
    }).to_csv(args.out_dir / "e_gate_decisions_no_official_labels.csv", index=False)
    (args.out_dir / "marker_panel.json").write_text(
        json.dumps({"tissue": args.tissue, "classes": details}, indent=2) + "\n"
    )
    audit = {
        "reference_labels_policy": "not_loaded",
        "input_seeds": str(args.seed_file.resolve()),
        "analysis_adata": str(args.adata.resolve()),
        "tissue": args.tissue,
        "rule": "original E seed retained only when Binary-SPA-style choice is E",
        "binary_margin_cutoff": 0.15,
        "original_e": int(original_e.sum()),
        "retained_e": int(accepted.sum()),
        "withheld_e": int(rejected.sum()),
        "other_seed_counts_unchanged": {
            label: int(np.sum(selected & (old_label == label)))
            for label in ("Connective_tissue", "Inflammatory", "Neoplastic")
        },
    }
    (args.out_dir / "gate_audit_no_official_labels.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
