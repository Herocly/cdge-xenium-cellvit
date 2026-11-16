#!/usr/bin/env python3
"""Evaluation-only metrics for a label-free pseudo-label run.

This script never writes to or alters pipeline outputs.  It joins a completed
pseudo-label table to a reference annotation only after selection is finished.
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
from sklearn.metrics import f1_score, precision_score


CORE = ("Connective_tissue", "Inflammatory", "Epithelial", "Neoplastic")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.ontology import load_ontology, map_reference_terms  # noqa: E402
from reliability.data_contracts import boolean_flags  # noqa: E402
REP1_TO_CORE = {
    "Tumor": "Neoplastic", "DCIS": "Neoplastic", "Cancer": "Neoplastic",
    "Basal": "Epithelial", "LP": "Epithelial", "ML": "Epithelial",
    "Luminal": "Epithelial", "Myoepithelial": "Epithelial", "Epithelial": "Epithelial",
    "Fibroblast": "Connective_tissue", "Endothelial": "Connective_tissue",
    "Adipocyte": "Connective_tissue", "Stroma": "Connective_tissue",
    "T": "Inflammatory", "B": "Inflammatory", "Plasma": "Inflammatory",
    "Macrophage": "Inflammatory", "Dendritic": "Inflammatory", "Mast": "Inflammatory", "NK": "Inflammatory",
}
PANCREAS_TO_CORE = {
    "Fibroblasts": "Connective_tissue",
    "Endothelial": "Connective_tissue",
    "Lymphatic Endothelial Cells": "Connective_tissue",
    "Smooth Muscle Cells": "Connective_tissue",
    "Macrophages": "Inflammatory",
    "T Cells": "Inflammatory",
    "B Cells": "Inflammatory",
    "Mast Cells": "Inflammatory",
    "CXCL9/10 Cells": "Inflammatory",
    "Acinar": "Epithelial",
    "Ductal": "Epithelial",
    "Endocrine 1": "Epithelial",
    "Endocrine 2": "Epithelial",
    "Tumor Cells": "Neoplastic",
    "CFTR- Tumor Cells": "Neoplastic",
    "Metaplastic Cells": "Neoplastic",
}
OVARY_TO_CORE = {
    "Smooth Muscle Cells": "Connective_tissue",
    "Tumor Associated Fibroblasts": "Connective_tissue",
    "Stromal Associated Fibroblasts": "Connective_tissue",
    "Stromal Associated Endothelial Cells": "Connective_tissue",
    "Pericytes": "Connective_tissue",
    "Tumor Associated Endothelial Cells": "Connective_tissue",
    "Macrophages": "Inflammatory",
    "T and NK Cells": "Inflammatory",
    "Fallopian Tube Epithelium": "Epithelial",
    "Ciliated Epithelial Cells": "Epithelial",
    "Urothelial-like Cells": "Epithelial",
    "Tumor Cells": "Neoplastic",
    "Proliferative Tumor Cells": "Neoplastic",
    "SOX2-OT+ Tumor Cells": "Neoplastic",
    "VEGFA+ Tumor Cells": "Neoplastic",
    "Malignant Cells Lining Cyst": "Neoplastic",
    "Inflammatory Tumor Cells": "Neoplastic",
    "Unassigned": "Others",
}


def to_core_pipeline(labels: pd.Series) -> pd.Series:
    values = labels.astype(str).copy()
    values.loc[values.str.startswith("Epithelial_program_")] = "Epithelial"
    return values.where(values.isin(CORE), "Unknown")


def xrrun_truth(adata_path: Path) -> pd.Series:
    adata = sc.read_h5ad(adata_path, backed="r")
    raw = adata.obs["cell_type"].astype(str)
    truth = pd.Series("Others", index=adata.obs_names.astype(str), dtype=object)
    truth.loc[raw.str.contains("Tumor_Cells", na=False)] = "Neoplastic"
    truth.loc[raw.str.contains("Normal_Epithelial|Myoepithelial", regex=True, na=False)] = "Epithelial"
    truth.loc[raw.str.contains("Macrophages|Lymphocytes|Plasma_Cells", regex=True, na=False)] = "Inflammatory"
    truth.loc[raw.str.contains("Stromal_Cells|Perivascular_Cells|Endothelial_Cells", regex=True, na=False)] = "Connective_tissue"
    return truth


def rep1_truth(annotation: Path, column: str) -> pd.Series:
    table = pd.read_csv(annotation, dtype={"cell_id": str}).set_index("cell_id")
    return table[column].astype(str).map(REP1_TO_CORE).fillna("Others")


def pancreas_truth(annotation: Path, column: str, ontology_path: Path) -> pd.Series:
    table = pd.read_csv(annotation, dtype={"cell_id": str}).set_index("cell_id")
    raw = table[column].astype(str)
    ontology = load_ontology(ontology_path)
    mapped = map_reference_terms(raw, ontology)
    fallback = raw.map(PANCREAS_TO_CORE)
    mapped = mapped.fillna(fallback)
    unknown = sorted(set(raw.loc[mapped.isna()]))
    if unknown:
        raise ValueError(f"Unmapped Pancreas reference groups: {unknown}")
    return mapped


def ovary_truth(annotation: Path, column: str) -> pd.Series:
    table = pd.read_csv(annotation, dtype={"cell_id": str}).set_index("cell_id")
    raw = table[column].astype(str)
    mapped = raw.map(OVARY_TO_CORE)
    unknown = sorted(set(raw.loc[mapped.isna()]))
    if unknown:
        raise ValueError(f"Unmapped Ovary reference groups: {unknown}")
    return mapped


def evaluate_selection(frame: pd.DataFrame, selection: pd.Series, truth: pd.Series, name: str) -> tuple[dict[str, object], pd.DataFrame]:
    selected = frame.loc[selection & frame["truth"].isin(CORE)].copy()
    total_reference = int(frame["truth"].isin(CORE).sum())
    whole = frame.loc[frame["truth"].isin(CORE)].copy()
    whole["prediction"] = whole["prediction"].where(selection.reindex(whole.index).fillna(False), "Unknown")
    summary: dict[str, object] = {
        "stage": name,
        "reference_cells": total_reference,
        "selected_cells": int(len(selected)),
        "coverage": float(len(selected) / total_reference) if total_reference else np.nan,
        "overall_precision": float((selected["prediction"] == selected["truth"]).mean()) if len(selected) else np.nan,
        "balanced_f1_selected_cells": float(f1_score(selected["truth"], selected["prediction"], labels=CORE, average="macro", zero_division=0)) if len(selected) else np.nan,
        "macro_f1_all_reference_cells_with_rejection": float(f1_score(whole["truth"], whole["prediction"], labels=CORE, average="macro", zero_division=0)) if len(whole) else np.nan,
        "selected_noncore_reference_cells": int((selection & ~frame["truth"].isin(CORE)).sum()),
    }
    rows: list[dict[str, object]] = []
    for label in CORE:
        predicted = selected["prediction"].eq(label)
        actual = selected["truth"].eq(label)
        truth_count = int(frame["truth"].eq(label).sum())
        correct = int((predicted & actual).sum())
        rows.append({
            "stage": name,
            "class": label,
            "reference_class_cells": truth_count,
            "selected_as_class": int(predicted.sum()),
            "correct_selected_as_class": correct,
            "pseudo_label_precision": float(precision_score(actual, predicted, zero_division=0)),
            "pseudo_label_f1_selected_cells": float(f1_score(actual, predicted, zero_division=0)),
            "correct_label_coverage_of_reference_class": float(correct / truth_count) if truth_count else np.nan,
            "recall_all_reference_cells": float(correct / truth_count) if truth_count else np.nan,
            "f1_all_reference_cells_with_rejection": float(f1_score(whole["truth"].eq(label), whole["prediction"].eq(label), zero_division=0)) if truth_count else np.nan,
        })
    return summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("xrrun", "rep1", "pancreas", "ovary"), required=True)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--prediction-adata", type=Path, help="QC-retained analysis axis; --adata remains the full evaluation population. Missing QC cells count as rejected.")
    parser.add_argument(
        "--pseudolabels",
        type=Path,
        help="Completed pseudo-label table. Omit for an initial-seed-only diagnostic.",
    )
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--reference-column", default="predicted_cell_type_Manual")
    parser.add_argument("--ontology", type=Path, default=ROOT / "resources" / "core_ontology_v1.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "xrrun":
        truth = xrrun_truth(args.adata)
    elif args.dataset == "rep1":
        truth = rep1_truth(args.annotation, args.reference_column)
    elif args.dataset == "pancreas":
        truth = pancreas_truth(args.annotation, args.reference_column, args.ontology)
    else:
        truth = ovary_truth(args.annotation, args.reference_column)
    def cell_axis(path):
        dataset = sc.read_h5ad(path, backed="r")
        ids = dataset.obs_names.astype(str)
        dataset.file.close()
        if not ids.is_unique:
            raise ValueError("AnnData cell IDs must be unique")
        return ids
    analysis_cells = cell_axis(args.adata)
    prediction_cells = cell_axis(args.prediction_adata) if args.prediction_adata else analysis_cells
    if not set(prediction_cells).issubset(set(analysis_cells)):
        raise ValueError("QC-retained analysis contains cells outside the evaluation population")
    if not truth.index.is_unique:
        raise ValueError("Reference cell IDs must be unique")
    truth = truth.reindex(analysis_cells).dropna()
    if not truth.index.is_unique:
        raise ValueError("Reference cell IDs must be unique")
    seeds = pd.read_csv(args.seeds, dtype={"cell_id": str}).set_index("cell_id")
    if not seeds.index.is_unique or set(seeds.index) != set(prediction_cells):
        raise ValueError("Seed table must match the unique analysis cell axis")
    seed_label_column = next(
        (column for column in ("training_label", "refined_pseudo_label") if column in seeds),
        None,
    )
    if seed_label_column is None:
        raise ValueError("Seed table must contain training_label or refined_pseudo_label.")
    frame = pd.DataFrame(index=truth.index)
    frame["truth"] = truth
    frame["seed_prediction"] = to_core_pipeline(seeds.reindex(frame.index)[seed_label_column])
    seed_flags = pd.Series(boolean_flags(seeds["is_initial_seed"]), index=seeds.index)
    frame["is_initial_seed"] = seed_flags.reindex(frame.index).fillna(False).astype(bool)
    frame["prediction"] = "Unknown"
    frame["is_pseudolabel"] = False
    if args.pseudolabels:
        final = pd.read_csv(args.pseudolabels, dtype={"cell_id": str}).set_index("cell_id")
        if not final.index.is_unique or set(final.index) != set(prediction_cells):
            raise ValueError("Prediction table must match the unique analysis cell axis")
        if "refined_pseudo_label" not in final:
            raise ValueError("Pseudo-label table must contain refined_pseudo_label.")
        frame["prediction"] = to_core_pipeline(final.reindex(frame.index)["refined_pseudo_label"])
        frame["is_pseudolabel"] = frame["prediction"].isin(CORE)
        if "seed_round" in final:
            frame["seed_round"] = final.reindex(frame.index)["seed_round"]

    summaries, by_class = [], []
    stages = [("initial_seed", frame["is_initial_seed"])]
    if args.pseudolabels:
        stages.append(("final_pseudolabel", frame["is_pseudolabel"]))
    for name, selection in stages:
        work = frame.copy()
        work["prediction"] = work["seed_prediction"] if name == "initial_seed" else work["prediction"]
        summary, class_rows = evaluate_selection(work, selection, truth, name)
        summaries.append(summary)
        by_class.append(class_rows)
    pd.DataFrame(summaries).to_csv(args.out_dir / "selection_summary_evaluation_only.csv", index=False)
    pd.concat(by_class, ignore_index=True).to_csv(args.out_dir / "selection_by_class_evaluation_only.csv", index=False)
    frame.reset_index(names="cell_id").to_csv(args.out_dir / "cell_level_evaluation_only.csv", index=False)
    if "seed_round" in frame:
        rounds = []
        for round_id in sorted(frame.loc[frame.is_pseudolabel, "seed_round"].unique()):
            summary, _ = evaluate_selection(frame, frame.seed_round.eq(round_id), truth, f"round_{int(round_id)}_only")
            rounds.append(summary)
        pd.DataFrame(rounds).to_csv(args.out_dir / "round_quality_evaluation_only.csv", index=False)
    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    audit = {
        "analysis_cells": len(analysis_cells), "matched_reference_cells": len(truth),
        "qc_retained_prediction_cells": len(prediction_cells),
        "qc_rejected_cells_counted_as_unknown": len(analysis_cells) - len(prediction_cells),
        "missing_reference_cells": len(analysis_cells) - len(truth),
        "reference_class_counts": truth.value_counts().astype(int).to_dict(),
        "all_analysis_prediction_coverage": float(to_core_pipeline(final.refined_pseudo_label).isin(CORE).sum() / len(analysis_cells)) if args.pseudolabels else None,
        "reference_mapping_policy": "historical_reference_adapters_unchanged_not_fitted_to_predictions",
        "others_policy": "reported_separately_not_silently_counted_as_correct_or_false",
        "prediction_sha256": sha(args.pseudolabels) if args.pseudolabels else None,
        "reference_sha256": sha(args.annotation) if args.annotation else None,
        "reference_mapping_sha256": sha(Path(__file__)),
        "reference_ontology_sha256": sha(args.ontology) if args.dataset == "pancreas" else None,
    }
    if args.annotation:
        raw_reference = pd.read_csv(args.annotation, dtype={"cell_id": str}).set_index("cell_id")
        mapping_audit = pd.DataFrame({"reference_term": raw_reference[args.reference_column].reindex(truth.index), "mapped_class": truth})
        mapping_audit.groupby(["reference_term", "mapped_class"], dropna=False).size().rename("reference_cells").reset_index().to_csv(
            args.out_dir / "reference_mapping_audit_evaluation_only.csv", index=False
        )
    (args.out_dir / "evaluation_population_and_provenance.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
