"""Export the fixed D-variant candidates as Neoplastic seeds for training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.cluster_evidence_gate import epithelial_marker_conflict_mask  # noqa: E402
from reliability.data_contracts import boolean_flags  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-table",
        type=Path,
        default=ROOT / "results/07_neoplastic_evidence/neoplastic_evidence_table_no_official_labels.csv",
    )
    parser.add_argument("--cluster-evidence", type=Path)
    parser.add_argument(
        "--evidence-variant",
        help="Optional row filter for historical evidence tables that store multiple variants.",
    )
    parser.add_argument(
        "--exclude-epithelial-marker-conflicts",
        action="store_true",
        help=(
            "Do not export a variant-D Neoplastic seed when both PanglaoDB and "
            "CellMarker support Epithelial for that cell's expression cluster."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/09_multievidence_neoplastic_seeds")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(args.evidence_table)
    if args.evidence_variant:
        if "variant" not in evidence.columns:
            raise ValueError("--evidence-variant was provided, but the evidence table has no 'variant' column.")
        evidence = evidence.loc[evidence["variant"].astype(str).eq(args.evidence_variant)].copy()
        if evidence.empty:
            raise ValueError(f"No evidence rows found for variant {args.evidence_variant!r}.")
    if args.exclude_epithelial_marker_conflicts and not args.cluster_evidence:
        raise ValueError("--cluster-evidence is required with --exclude-epithelial-marker-conflicts.")

    output = evidence[["cell_id", "variant_D_all_evidence"]].copy()
    if not output["cell_id"].is_unique or output["cell_id"].isna().any():
        raise ValueError("Neoplastic evidence must have unique nonmissing cell IDs")
    output["is_neoplastic_seed"] = boolean_flags(output.pop("variant_D_all_evidence"))
    output["marker_conflict_excluded"] = False
    if args.exclude_epithelial_marker_conflicts:
        cluster_column = next(
            (column for column in ("leiden_cluster", "expression_cluster") if column in evidence.columns),
            None,
        )
        if cluster_column is None:
            raise ValueError("Neoplastic evidence needs either 'leiden_cluster' or 'expression_cluster' for the conflict gate.")
        cluster_table = pd.read_csv(args.cluster_evidence, dtype={"cluster": str})
        required = {"cluster", "panglao_label", "cellmarker_label"}
        missing = required.difference(cluster_table.columns)
        if missing:
            raise ValueError(f"Cluster evidence misses conflict-gate columns: {sorted(missing)}")
        cluster_table = cluster_table[list(required)].copy()
        cluster_table["cluster"] = cluster_table["cluster"].astype(str)
        output["cluster_for_conflict_gate"] = evidence[cluster_column].astype(str)
        output = output.merge(cluster_table, left_on="cluster_for_conflict_gate", right_on="cluster", how="left")
        conflict = epithelial_marker_conflict_mask(output["cluster_for_conflict_gate"].to_numpy(), cluster_table)
        output["marker_conflict_excluded"] = output["is_neoplastic_seed"] & conflict
        output.loc[output["marker_conflict_excluded"], "is_neoplastic_seed"] = False

    output["pseudo_label"] = output["is_neoplastic_seed"].map(
        {True: "Neoplastic", False: "Unknown"}
    )
    output.to_csv(args.out_dir / "neoplastic_seed_candidates.csv", index=False)
    output.groupby("marker_conflict_excluded", dropna=False).size().rename("cell_count").reset_index().to_csv(
        args.out_dir / "neoplastic_conflict_gate_audit_no_official_labels.csv", index=False
    )
    print(f"Exported {int(output['is_neoplastic_seed'].sum())} multi-evidence Neoplastic seeds.")


if __name__ == "__main__":
    main()
