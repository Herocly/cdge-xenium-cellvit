#!/usr/bin/env python3
"""Run the active research stages from one method directory."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "methods/current"

STAGES = {
    "prepare": (CURRENT / "code/01_prepare_label_free_analysis.py", []),
    "cluster-evidence": (CURRENT / "code/02_build_cluster_evidence.py", [
        "--cdge-root", str(CURRENT / "vendor/cdge"),
        "--db-dir", str(CURRENT / "resources/marker_databases"),
        "--ontology", str(CURRENT / "resources/cell_ontology/cl.obo"),
        "--ontology-policy", str(CURRENT / "resources/cell_ontology/four_class_policy.json"),
        "--database-policy", "original", "--groupby", "leiden_res_2.00", "--top-n", "20"]),
    "normal-seeds": (CURRENT / "code/03_select_non_tumor_seeds.py", [
        "--seed-policy", "multi", "--core-fraction", "0.30"]),
    "bootstrap-cnv": (CURRENT / "code/05_run_insitucnv.py", [
        "--coordinates", str(CURRENT / "resources/ensembl_hg38_coordinates.tsv")]),
    "bootstrap-audit": (CURRENT / "code/06_audit_cnv_candidates.py", []),
    "bootstrap-evidence": (CURRENT / "code/09_build_neoplastic_evidence_ablation.py", [
        "--neoplastic-genes", str(CURRENT / "resources/marker_databases/neoplastic_genes.csv"),
        "--gene-evidence-quantile", "0.90", "--cnv-reference-quantile", "0.99"]),
    "bootstrap-seeds": (CURRENT / "code/07_export_neoplastic_seeds.py", []),
    "seed-qc": (CURRENT / "normal_seed_qc.py", ["screen-seeds"]),
    "historical-seed-merge": (CURRENT / "code/10_combine_initial_seeds.py", ["--use-tiered-epithelial"]),
    "normal-expand": (CURRENT / "normal_expansion.py", [
        "generate", "--schedule", "progressive", "--budget-mode", "late-growing", "--teacher-mode", "off"]),
    "cnv": (CURRENT / "code/05_run_strict_ci_cnv.py", [
        "--reference-source", "strict", "--reference-classes", "Connective_tissue", "Inflammatory",
        "--coordinates", str(CURRENT / "resources/ensembl_hg38_coordinates.tsv")]),
    "cnv-audit": (CURRENT / "code/06_audit_cnv_candidates.py", []),
    "neoplastic-evidence": (CURRENT / "code/09_build_neoplastic_evidence_ablation.py", [
        "--neoplastic-genes", str(CURRENT / "resources/marker_databases/neoplastic_genes.csv"),
        "--gene-evidence-quantile", "0.90", "--cnv-reference-quantile", "0.99"]),
    "neoplastic-seeds": (CURRENT / "code/07_export_neoplastic_seeds.py", []),
    "expand": (CURRENT / "code/11_export_iterative_pseudolabels.py", [
        "--max-iterations", "20", "--cluster-release-rounds", "6",
        "--support-min-structural", "0.5", "--support-min-annotation", "0.5"]),
}


def load_four_class_entry():
    path = CURRENT / "code/seed_merge_metrics.py"
    sys.path.insert(0, str(CURRENT / "code"))
    spec = importlib.util.spec_from_file_location("bundled_b216_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def combine(arguments):
    parser = argparse.ArgumentParser(description="Merge four-class seeds with explicit paths.")
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--normal-seeds", type=Path, required=True)
    parser.add_argument("--neoplastic-seeds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(arguments)
    method = load_four_class_entry()
    if args.out_dir.exists():
        raise FileExistsError("Use a new output directory")
    adata = method.sc.read_h5ad(args.adata, backed="r")
    try:
        cells = adata.obs_names.astype(str).copy()
        if any(any(token in str(column).lower() for token in (
            "annotation", "ground_truth", "official_label", "cell_type", "truth"
        )) for column in adata.obs.columns):
            raise ValueError("Annotation-like column in generation AnnData")
    finally:
        adata.file.close()
    result = method.merge_original_and_cnv(
        method.pd.read_csv(args.normal_seeds, dtype={"cell_id": str}),
        method.pd.read_csv(args.neoplastic_seeds, dtype={"cell_id": str}), cells)
    counts = result.loc[result.is_initial_seed].training_label.value_counts().to_dict()
    if set(counts) != set(method.CLASSES):
        raise ValueError("Each of the four classes requires seeds: " + str(counts))
    args.out_dir.mkdir(parents=True)
    destination = args.out_dir / "combined_initial_seeds_no_official_labels.csv"
    result.to_csv(destination, index=False)
    inputs = {key: {"path": str(path.resolve()), "sha256": method.sha(path)} for key, path in (
        ("adata", args.adata), ("normal_seeds", args.normal_seeds),
        ("neoplastic_seeds", args.neoplastic_seeds), ("source_code", CURRENT / "code/seed_merge_metrics.py"))}
    audit = {"counts": counts, "conflicting_cells": int(result.seed_conflict.sum()),
             "inputs": inputs, "output_sha256": method.sha(destination),
             "official_labels_policy": "not_loaded",
             "policy": "Original C/I/E seeds plus new D; conflicting seeds withheld"}
    (args.out_dir / "seed_merge_audit_no_official_labels.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


def evaluate(arguments):
    parser = argparse.ArgumentParser(description="Evaluate fixed predictions on an already mapped reference.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--method-name", default="B2.16")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(arguments)
    method = load_four_class_entry()
    if args.out_dir.exists():
        raise FileExistsError("Use a new evaluation output directory")
    fixed_hash = method.sha(args.predictions)
    prediction = method.pd.read_csv(args.predictions, dtype={"cell_id": str})
    reference_hash = method.sha(args.reference)
    reference = method.pd.read_csv(args.reference, usecols=["cell_id", "truth"], dtype={"cell_id": str})
    metrics, confusion = method.per_class_metrics(prediction, reference, args.method_name)
    if not len(reference) or not len(prediction):
        raise ValueError("Prediction and evaluation populations cannot be empty")
    aligned = reference.merge(prediction[["cell_id", "predicted_class"]], on="cell_id", validate="one_to_one")
    selected = aligned.predicted_class.isin(method.CLASSES)
    all_selected = prediction.predicted_class.isin(method.CLASSES)
    correct = int((selected & aligned.truth.eq(aligned.predicted_class)).sum())
    summary = {
        "method": args.method_name, "input_cells": len(prediction), "evaluated_cells": len(reference),
        "overall_precision": correct / int(selected.sum()) if selected.any() else None,
        "macro_f1_all_reference": float(metrics.f1_all_reference_cells.mean()),
        "coverage_evaluation_population": float(selected.mean()),
        "coverage_all_input_cells": float(all_selected.mean()),
        "unknown_evaluation_cells": int((~selected).sum()),
        "prediction_path": str(args.predictions.resolve()), "prediction_sha256": fixed_hash,
        "reference_path": str(args.reference.resolve()), "reference_sha256": reference_hash,
        "policy": "Fixed predictions; four-class reference supplied externally; Unknown counts as missed"}
    if fixed_hash != method.sha(args.predictions) or reference_hash != method.sha(args.reference):
        raise RuntimeError("Evaluation inputs changed")
    args.out_dir.mkdir(parents=True)
    metrics.to_csv(args.out_dir / "per_class_evaluation_only.csv", index=False)
    confusion.to_csv(args.out_dir / "confusion_evaluation_only.csv")
    (args.out_dir / "evaluation_summary_only.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def export_full_axis(arguments):
    parser = argparse.ArgumentParser(description="Restore QC-rejected input cells as Unknown.")
    parser.add_argument("--input-adata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(arguments)
    method = load_four_class_entry()
    if args.out_dir.exists():
        raise FileExistsError("Use a new output directory")
    data = method.sc.read_h5ad(args.input_adata, backed="r")
    try:
        cells = data.obs_names.astype(str).copy()
    finally:
        data.file.close()
    prediction_hash = method.sha(args.predictions)
    prediction = method.pd.read_csv(args.predictions, dtype={"cell_id": str})
    if not cells.is_unique or not prediction.cell_id.is_unique:
        raise ValueError("Input and prediction cell IDs must be unique")
    if not set(prediction.cell_id) <= set(cells):
        raise ValueError("Prediction contains cells outside the original input")
    if not set(prediction.predicted_class) <= set((*method.CLASSES, "Unknown")):
        raise ValueError("Invalid prediction classes")
    output = prediction.set_index("cell_id").reindex(cells)
    dropped = ~output.index.isin(prediction.cell_id)
    output.loc[dropped, "predicted_class"] = "Unknown"
    output["rejected_by_preprocessing"] = dropped
    output.index.name = "cell_id"
    if prediction_hash != method.sha(args.predictions):
        raise RuntimeError("Predictions changed")
    args.out_dir.mkdir(parents=True)
    destination = args.out_dir / "all_input_predictions_no_official_labels.csv"
    output.reset_index().to_csv(destination, index=False)
    (args.out_dir / "generation_manifest_no_official_labels.json").write_text(json.dumps({
        "input_cells": len(cells), "processed_cells": len(prediction), "qc_rejected_unknown": int(dropped.sum()),
        "input_adata": str(args.input_adata.resolve()), "input_adata_sha256": method.sha(args.input_adata),
        "source_prediction_sha256": prediction_hash, "output_sha256": method.sha(destination),
        "official_labels_policy": "not_loaded", "policy": "Keep model predictions; add only QC-rejected cells as Unknown"}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("stage", choices=["list", *STAGES, "combine", "export-full-axis", "evaluate"])
    parser.add_argument("--python", type=Path, help="Scientific Python interpreter for the selected stage")
    parser.add_argument("--dry-run", action="store_true")
    args, rest = parser.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if args.stage == "list":
        for stage, (script, _) in STAGES.items():
            print(stage + " -> " + str(script.relative_to(ROOT)))
        print("combine -> merge_original_and_cnv")
        print("export-full-axis -> retain QC-rejected original cells as Unknown")
        print("evaluate -> per_class_metrics + explicit coverage denominators")
        return
    interpreter = str(args.python.expanduser().absolute()) if args.python else sys.executable
    if args.stage in STAGES:
        script, defaults = STAGES[args.stage]
        command = [interpreter, str(script), *defaults, *rest]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    elif args.python:
        command = [interpreter, str(Path(__file__).resolve()), args.stage, "--", *rest]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    elif args.dry_run:
        print(shlex.join([interpreter, str(Path(__file__).resolve()), args.stage, "--", *rest]))
    else:
        {"combine": combine, "export-full-axis": export_full_axis, "evaluate": evaluate}[args.stage](rest)


if __name__ == "__main__":
    main()
