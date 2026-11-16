#!/usr/bin/env python3
"""Compare scientific cell-level outputs from two runs by cell ID.

Use on the same input slide and the same method variant. This is a regression
check, not a biological accuracy test.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TABLES = {
    "initial_normal_seeds": ("10_original_seed_source/combined_initial_seeds_no_official_labels.csv",
                             ("training_label", "is_initial_seed")),
    "normal_reference": ("11_normal_expansion/normal_reference_no_official_labels.csv",
                         ("normal_label", "seed_round")),
    "neoplastic_seeds": ("15_new_neoplastic_seeds/neoplastic_seed_candidates.csv",
                         ("is_neoplastic_seed",)),
    "four_class_seeds": ("16_four_class_seeds/combined_initial_seeds_no_official_labels.csv",
                         ("training_label", "is_initial_seed", "seed_conflict")),
    "accepted_training_labels": ("17_four_class_iteration/final_pseudolabels_no_official_labels.csv",
                                 ("seed_round", "is_initial_seed", "training_program_label",
                                  "refined_pseudo_label")),
    "final_predictions": ("18_full_input_predictions/all_input_predictions_no_official_labels.csv",
                          ("predicted_class",)),
}
METRICS = ("input_cells", "evaluated_cells", "overall_precision",
           "macro_f1_all_reference", "coverage_evaluation_population",
           "coverage_all_input_cells", "unknown_evaluation_cells")


def read_index(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if "cell_id" not in columns:
            raise ValueError(f"Missing cell_id in {path}")
        rows = {}
        for row in reader:
            cell_id = row["cell_id"]
            if cell_id in rows:
                raise ValueError(f"Duplicate cell_id {cell_id} in {path}")
            rows[cell_id] = row
    return rows, columns


def compare(old: Path, new: Path, columns: tuple[str, ...]) -> dict:
    old_rows, old_columns = read_index(old)
    new_rows, new_columns = read_index(new)
    if not columns:
        columns = tuple(sorted(set(old_columns) & set(new_columns) - {"cell_id"}))
    absent = set(columns) - set(old_columns) | set(columns) - set(new_columns)
    if absent:
        raise ValueError(f"Missing comparison columns: {sorted(absent)}")
    old_ids, new_ids = set(old_rows), set(new_rows)
    differences = []
    for cell_id in sorted(old_ids & new_ids):
        for column in columns:
            if old_rows[cell_id][column] != new_rows[cell_id][column]:
                differences.append({"cell_id": cell_id, "column": column,
                                    "old": old_rows[cell_id][column],
                                    "new": new_rows[cell_id][column]})
    return {"old_cells": len(old_ids), "new_cells": len(new_ids),
            "missing_in_new": len(old_ids - new_ids),
            "new_only": len(new_ids - old_ids),
            "columns": columns, "different_values": len(differences),
            "examples": differences[:10],
            "exact_match": old_ids == new_ids and not differences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for name, (relative, columns) in TABLES.items():
        old, new = args.old / relative, args.new / relative
        if not old.is_file() or not new.is_file():
            report[name] = {"status": "missing_file", "old": old.is_file(), "new": new.is_file()}
        else:
            report[name] = compare(old, new, columns)
    old_eval = args.old / "19_evaluation_only/evaluation_summary_only.json"
    new_eval = args.new / "19_evaluation_only/evaluation_summary_only.json"
    if old_eval.is_file() and new_eval.is_file():
        old_metrics = json.loads(old_eval.read_text())
        new_metrics = json.loads(new_eval.read_text())
        differences = {key: {"old": old_metrics.get(key), "new": new_metrics.get(key)}
                       for key in METRICS if old_metrics.get(key) != new_metrics.get(key)}
        report["evaluation"] = {"exact_match": not differences,
                                "different_metrics": differences}
    elif old_eval.is_file() != new_eval.is_file():
        report["evaluation"] = {"status": "missing_file", "old": old_eval.is_file(),
                                "new": new_eval.is_file()}
    print(json.dumps(report, indent=2))
    if not all(item.get("exact_match") for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
