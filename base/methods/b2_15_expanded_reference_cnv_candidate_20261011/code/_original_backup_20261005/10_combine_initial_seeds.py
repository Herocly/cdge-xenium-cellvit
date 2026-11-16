#!/usr/bin/env python3
"""Combine the existing label-free initial seed sources for iterative training.

The script only aligns cell IDs and resolves contradictory initial labels
conservatively. It never reads reference annotations. Existing Connective /
Inflammatory seeds take precedence over a Neoplastic candidate. Epithelial
lineage alone is not proof of benign state; C conflicts are withheld and
complete D evidence can supply a Neoplastic seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from reliability.data_contracts import aligned_cell_table, boolean_flags


EPITHELIAL_PREFIX = "Epithelial_program_"


def withhold_ambiguous_epithelial_seeds(result: pd.DataFrame, evidence: pd.DataFrame) -> int:
    aligned = aligned_cell_table(evidence, result.index)
    tumor_support = boolean_flags(aligned["variant_C_gene_plus_cnv"])
    epithelial = result["training_label"].eq("Epithelial") | result["training_label"].str.startswith(EPITHELIAL_PREFIX)
    conflicting = epithelial & tumor_support & result["is_initial_seed"]
    result.loc[conflicting, "is_initial_seed"] = False
    result.loc[conflicting, "training_label"] = "Unknown"
    result.loc[conflicting, "seed_source"] = "withheld_lineage_status_conflict"
    return int(conflicting.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--tiered-seeds", type=Path, required=True)
    parser.add_argument("--epithelial-program-seeds", type=Path)
    parser.add_argument(
        "--use-tiered-epithelial",
        action="store_true",
        help="Use the existing generic Epithelial core cells from --tiered-seeds instead of breast program seeds.",
    )
    parser.add_argument("--neoplastic-seeds", type=Path, required=True)
    parser.add_argument("--neoplastic-evidence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.use_tiered_epithelial and args.epithelial_program_seeds:
        raise ValueError("Choose either --use-tiered-epithelial or --epithelial-program-seeds, not both.")
    if not args.use_tiered_epithelial and not args.epithelial_program_seeds:
        raise ValueError("Provide --epithelial-program-seeds unless --use-tiered-epithelial is selected.")

    adata = sc.read_h5ad(args.adata, backed="r")
    cells = adata.obs_names.astype(str)
    adata.file.close()
    if not cells.is_unique:
        raise ValueError("Analysis cell IDs must be unique")
    result = pd.DataFrame(
        {
            "cell_id": cells,
            "is_initial_seed": False,
            "training_label": "Unknown",
            "seed_source": "",
        }
    ).set_index("cell_id")

    tiered = aligned_cell_table(pd.read_csv(args.tiered_seeds, dtype={"cell_id": str}), cells)
    tiered = tiered[boolean_flags(tiered["is_initial_seed"])]
    tiered_labels = ["Connective_tissue", "Inflammatory"]
    if args.use_tiered_epithelial:
        tiered_labels.append("Epithelial")
    tiered = tiered[tiered["refined_pseudo_label"].isin(tiered_labels)]
    result.loc[tiered.index, "is_initial_seed"] = True
    result.loc[tiered.index, "training_label"] = tiered["refined_pseudo_label"]
    result.loc[tiered.index, "seed_source"] = "reliable_cluster_core"

    epithelial = pd.DataFrame()
    nonconflicting_epithelial = pd.Index([], dtype=object)
    if args.epithelial_program_seeds:
        epithelial = aligned_cell_table(pd.read_csv(args.epithelial_program_seeds, dtype={"cell_id": str}), cells)
        epithelial = epithelial[boolean_flags(epithelial["is_initial_seed"])]
        epithelial = epithelial[
            epithelial["training_label"].astype(str).str.startswith(EPITHELIAL_PREFIX)
        ]
        nonconflicting_epithelial = epithelial.index[~epithelial.index.isin(result.index[result["is_initial_seed"]])]
        result.loc[nonconflicting_epithelial, "is_initial_seed"] = True
        result.loc[nonconflicting_epithelial, "training_label"] = epithelial.loc[
            nonconflicting_epithelial, "training_label"
        ]
        result.loc[nonconflicting_epithelial, "seed_source"] = "epithelial_expression_program"

    withheld_epithelial = withhold_ambiguous_epithelial_seeds(
        result, pd.read_csv(args.neoplastic_evidence, dtype={"cell_id": str})
    )
    neoplastic = aligned_cell_table(pd.read_csv(args.neoplastic_seeds, dtype={"cell_id": str}), cells)
    neoplastic = neoplastic[boolean_flags(neoplastic["is_neoplastic_seed"])]
    # D may override an epithelial lineage seed, not a non-epithelial reference.
    allowed = result["training_label"].isin(["Unknown", "Epithelial"])
    nonconflicting_neoplastic = neoplastic.index[neoplastic.index.isin(result.index[allowed])]
    result.loc[nonconflicting_neoplastic, "is_initial_seed"] = True
    result.loc[nonconflicting_neoplastic, "training_label"] = "Neoplastic"
    result.loc[nonconflicting_neoplastic, "seed_source"] = "multievidence_neoplastic"

    if not result.index.equals(cells):
        raise RuntimeError("Seed table lost the analysis AnnData cell order.")

    output = result.reset_index()
    output.to_csv(args.out_dir / "combined_initial_seeds_no_official_labels.csv", index=False)
    summary = (
        output[output["is_initial_seed"]]
        .groupby(["training_label", "seed_source"])
        .size()
        .rename("cell_count")
        .reset_index()
    )
    summary.to_csv(args.out_dir / "combined_initial_seed_counts_no_official_labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "official_labels_policy": "not_loaded",
                "tiered_core_candidates": len(tiered),
                "epithelial_program_candidates": len(epithelial),
                "epithelial_selection_mode": "tiered_cluster_core" if args.use_tiered_epithelial else "epithelial_expression_program",
                "neoplastic_multievidence_candidates": len(neoplastic),
                "epithelial_C_conflicts_withheld_before_D": withheld_epithelial,
                "epithelial_conflicts_excluded": len(epithelial) - len(nonconflicting_epithelial),
                "neoplastic_conflicts_excluded": len(neoplastic) - len(nonconflicting_neoplastic),
            }
        ]
    ).to_csv(args.out_dir / "seed_merge_audit_no_official_labels.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
