#!/usr/bin/env python3
"""Build the frozen cluster-evidence table without loading ground truth.

The marker-voting calls intentionally come from the original CDGE code.  This
script adds only the project reliability diagnostics and never opens an
annotation file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from reliability.common import percentile_score, pseudo_to_core  # noqa: E402
from reliability.expression_space import centroid_silhouette, cluster_multimodality_dip, expression_features  # noqa: E402
from reliability.marker_voting_metrics import vote_metrics  # noqa: E402
from reliability.ontology_tree import (  # noqa: E402
    CellOntologyMapper, DEFAULT_POLICY, DEFAULT_SNAPSHOT,
    map_marker_databases, write_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--cdge-root", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, default=ROOT / "resources" / "marker_databases")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--ontology-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--tissue",
        default=None,
        help=(
            "Optional CDGE tissue filter. By default no tissue filter is applied, "
            "matching the original CDGE database-loader default."
        ),
    )
    parser.add_argument("--groupby", default="leiden_res_2.00")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--database-policy", choices=("original", "human_unique_source_balanced"), default="original")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def neighbour_consistency(adata: sc.AnnData, clusters: np.ndarray) -> np.ndarray:
    graph = adata.obsp["connectivities"].tocsr()
    values = np.zeros(adata.n_obs, dtype=np.float32)
    for index in range(adata.n_obs):
        neighbours = graph.indices[graph.indptr[index] : graph.indptr[index + 1]]
        values[index] = np.mean(clusters[neighbours] == clusters[index]) if len(neighbours) else 0.0
    return values


def load_original_voting_functions(cdge_root: Path) -> dict[str, object]:
    """Load CDGE's marker-voting definitions without its optional image stack.

    The source module imports a geometry helper used only by unrelated image
    routines.  The replacements below remove that import line only; the
    database filters and vote equations are executed verbatim from the
    original ``bottom_up_utils.py`` source.
    """
    source_file = cdge_root / "preprocessing_hest" / "preproce_utils" / "bottom_up_utils.py"
    source = source_file.read_text(encoding="utf-8")
    source = source.replace("from .common import LABEL_MAP, map_category_to_label", "")
    namespace: dict[str, object] = {"__file__": str(source_file), "__name__": "cdge_voting_compat"}
    exec(compile(source, str(source_file), "exec"), namespace)
    return namespace


def load_databases_compat(
    db_dir: Path,
    voting: dict[str, object],
    tissue: str | None,
    ontology_path: Path,
    policy_path: Path = DEFAULT_POLICY,
) -> tuple[object, object, object, object, set[str], set[str], dict[str, object]]:
    """Load marker databases with an optional, explicit tissue restriction."""
    normalized_tissue = None if tissue is None or tissue.lower() in {"all", "none"} else tissue
    panglao = pd.read_csv(db_dir / "marker_gene_list.tsv", sep="\t")
    cellmarker_all = pd.read_csv(
        db_dir / "database_gene_functions" / "Cell_marker_All.csv",
        low_memory=False,
    )
    cellmarker = cellmarker_all.loc[cellmarker_all["cell_type"].eq("Normal cell")].copy()
    cancer = cellmarker_all.loc[cellmarker_all["cell_type"].eq("Cancer cell")].copy()

    if normalized_tissue is not None:
        supported = set(voting["PANGLAO_ORGAN_MAP"]) | set(voting["CELLMARKER_TISSUE_MAP"])
        if normalized_tissue not in supported:
            raise ValueError(
                f"Unknown CDGE tissue filter {normalized_tissue!r}; choose one of {sorted(supported)} "
                "or omit --tissue for the unfiltered default."
            )
        panglao = voting["_filter_panglao"](panglao, normalized_tissue)
        cellmarker = voting["_filter_cellmarker"](cellmarker, normalized_tissue)
        cancer = voting["_filter_cellmarker"](cancer, normalized_tissue)

    mapper = CellOntologyMapper.from_files(ontology_path, policy_path, organ=normalized_tissue)
    panglao_map, cellmarker_map, crosswalk, ontology_audit = map_marker_databases(
        panglao, cellmarker, mapper
    )
    neoplastic = pd.read_csv(db_dir / "neoplastic_genes.csv")
    neoplastic_set = {str(gene).strip().upper() for gene in neoplastic["gene"].tolist()}
    cancer_set: set[str] = set()
    human_cancer = cancer.loc[cancer["species"].fillna("").str.lower().eq("human")]
    for symbol in human_cancer["Symbol"].dropna():
        cancer_set.update(voting["_split_cellmarker_symbols"](symbol))
    audit = {
        "tissue_filter": normalized_tissue,
        "database_scope": "all_tissues" if normalized_tissue is None else "tissue_filtered",
        "panglao_rows": int(len(panglao)),
        "cellmarker_normal_rows": int(len(cellmarker)),
        "cellmarker_cancer_rows": int(len(cancer)),
        "panglao_species_rows": panglao["species"].fillna("missing").value_counts().to_dict(),
        "cellmarker_species_rows": cellmarker["species"].fillna("missing").value_counts().to_dict(),
        "panglao_duplicate_gene_type_records": int(panglao.duplicated(["official gene symbol", "cell type"]).sum()),
        "cellmarker_duplicate_symbol_type_records": int(cellmarker.duplicated(["Symbol", "cell_name"]).sum()),
        "ontology_file": str(ontology_path),
        "ontology_version": ontology_audit["policy_version"],
        "ontology_audit": ontology_audit,
    }
    audit["ontology_crosswalk"] = crosswalk
    return panglao, cellmarker, panglao_map, cellmarker_map, neoplastic_set, cancer_set, audit


def source_balanced_gene_votes(panglao, cellmarker, panglao_map, cellmarker_map, voting):
    """One unit per covered source/gene; repeated records cannot add evidence."""
    panglao = panglao.loc[panglao["species"].fillna("").str.split().map(lambda values: "Hs" in values)]
    cellmarker = cellmarker.loc[cellmarker["species"].fillna("").str.lower().eq("human")]
    sources = [voting["build_gene_votes"](panglao, panglao_map),
               voting["build_gene_votes"](panglao.iloc[:0], panglao_map, cellmarker, cellmarker_map)]
    normalized = []
    for source in sources:
        result = {}
        for gene, vector in source.items():
            present = (vector > 0).astype(float)
            if present.sum():
                result[gene] = present / present.sum()
        normalized.append(result)
    combined = {}
    for gene in set(normalized[0]) | set(normalized[1]):
        combined[gene] = sum((source[gene] for source in normalized if gene in source),
                             np.zeros(len(voting["VOTE_CATEGORIES"])))
    return combined, *normalized


def classify_normalized_votes(genes, gene_votes, categories, top_n=10):
    total = np.zeros(len(categories), dtype=float)
    for rank, gene in enumerate(genes[:top_n]):
        total += gene_votes.get(gene, np.zeros(len(categories))) * (top_n - rank)
    if not total.sum():
        return {"predicted_category": "Unknown", "confidence": 0.0, "total_votes": {}}
    return {"predicted_category": categories[int(total.argmax())],
            "confidence": float(total.max() / total.sum()),
            "total_votes": dict(zip(categories, total.tolist()))}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    voting = load_original_voting_functions(args.cdge_root)
    build_gene_votes = voting["build_gene_votes"]
    check_cancer_markers = voting["check_cancer_markers"]
    check_neoplastic = voting["check_neoplastic"]
    classify_cluster = voting["classify_cluster"]
    adata = sc.read_h5ad(args.adata)
    if args.groupby not in adata.obs:
        raise KeyError(f"Missing {args.groupby}; run label-free preprocessing first.")
    clusters = adata.obs[args.groupby].astype(str).to_numpy()
    print(f"Ranking marker genes for {adata.n_obs} cells and {pd.unique(clusters).size} clusters", flush=True)
    sc.tl.rank_genes_groups(adata, groupby=args.groupby, method="wilcoxon", use_raw=False)
    ranked = sc.get.rank_genes_groups_df(adata, group=None)
    ranked = ranked.sort_values(["group", "scores"], ascending=[True, False])
    top_markers = ranked.groupby("group", observed=True).head(args.top_n).reset_index(drop=True)
    top_markers.to_csv(args.out_dir / "cluster_top_marker_genes.csv", index=False)

    db = load_databases_compat(args.db_dir, voting, args.tissue, args.ontology, args.ontology_policy)
    panglao_db, cellmarker_db, panglao_map, cellmarker_map, neoplastic_set, cancer_set, database_audit = db
    crosswalk = database_audit.pop("ontology_crosswalk")
    write_audit(args.out_dir, crosswalk, database_audit["ontology_audit"])
    combined_votes = build_gene_votes(panglao_db, panglao_map, cellmarker_db, cellmarker_map)
    panglao_votes = build_gene_votes(panglao_db, panglao_map)
    cellmarker_votes = build_gene_votes(panglao_db.iloc[:0], panglao_map, cellmarker_db, cellmarker_map)
    if args.database_policy == "human_unique_source_balanced":
        combined_votes, panglao_votes, cellmarker_votes = source_balanced_gene_votes(
            panglao_db, cellmarker_db, panglao_map, cellmarker_map, voting)
        classify_cluster = lambda genes, votes, top_n: classify_normalized_votes(genes, votes, voting["VOTE_CATEGORIES"], top_n)

    features = expression_features(adata, 30, "X_pca")
    silhouette = centroid_silhouette(features, clusters)
    neighbours = neighbour_consistency(adata, clusters)
    rows: list[dict[str, object]] = []
    unique_clusters = sorted(pd.unique(clusters), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    for number, cluster in enumerate(unique_clusters, start=1):
        mask = clusters == cluster
        positions = np.flatnonzero(mask)
        genes = top_markers.loc[top_markers["group"].astype(str).eq(cluster), "names"].astype(str).tolist()
        result = classify_cluster(genes, combined_votes, top_n=10)
        # Tumor hints must not overwrite the normal-lineage vote or its support.
        neo = check_neoplastic(genes, neoplastic_set, top_n=20, threshold=.25)
        cancer = check_cancer_markers(genes, cancer_set, top_n=20, threshold=.25)
        source_p = classify_cluster(genes, panglao_votes, top_n=10)
        source_c = classify_cluster(genes, cellmarker_votes, top_n=10)
        cluster_features = features[positions]
        centre = cluster_features.mean(axis=0)
        distances = np.linalg.norm(cluster_features - centre, axis=1)
        norms = np.linalg.norm(cluster_features, axis=1) * max(np.linalg.norm(centre), 1e-8)
        cohesion = np.divide(cluster_features @ centre, norms, out=np.zeros_like(norms), where=norms > 0)
        dip, dip_p, dip_components = cluster_multimodality_dip(cluster_features)
        pseudo = pseudo_to_core(result["predicted_category"])
        panglao = pseudo_to_core(source_p["predicted_category"])
        cellmarker = pseudo_to_core(source_c["predicted_category"])
        row: dict[str, object] = {
            "cluster": cluster, "n_cells": len(positions), "pseudo_label": pseudo,
            "tumor_gene_marker_hint": neo["is_neoplastic"], "organ_cancer_marker_hint": cancer["is_cancer"],
            "panglao_label": panglao, "panglao_confidence": source_p.get("confidence", 0.0),
            "cellmarker_label": cellmarker, "cellmarker_confidence": source_c.get("confidence", 0.0),
            "independent_source_agreement": panglao == cellmarker and panglao not in {"Unknown", "Others"},
            "combined_panglao_agreement": pseudo == panglao and pseudo not in {"Unknown", "Others"},
            "combined_cellmarker_agreement": pseudo == cellmarker and pseudo not in {"Unknown", "Others"},
            "mean_silhouette": float(silhouette[positions].mean()),
            "neighbor_consistency": float(neighbours[positions].mean()),
            "expression_cohesion": float(cohesion.mean()),
            "within_dispersion": float(distances.mean()),
            "dip_statistic": dip, "dip_p_value": dip_p, "dip_tested_components": dip_components,
            "multimodal_candidate": bool(dip_p < .05), "top_marker_genes": ", ".join(genes),
            "official_labels_policy": "not_loaded",
        }
        row.update(vote_metrics(result, genes, combined_votes))
        row["voting_top10_marker_coverage"] = sum(g in combined_votes for g in genes[:10]) / max(1, len(genes[:10]))
        rows.append(row)
        if number % 20 == 0 or number == len(unique_clusters):
            print(f"processed {number}/{len(unique_clusters)} clusters", flush=True)

    table = pd.DataFrame(rows)
    structural = pd.DataFrame({
        "silhouette": percentile_score(table["mean_silhouette"], True),
        "neighbor": percentile_score(table["neighbor_consistency"], True),
        "cohesion": percentile_score(table["expression_cohesion"], True),
        "dispersion": percentile_score(table["within_dispersion"], False),
        "inverse_dip": percentile_score(table["dip_statistic"], False),
    })
    annotation = pd.DataFrame({
        "confidence": percentile_score(table["vote_confidence"], True),
        "margin": percentile_score(table["vote_margin"], True),
        "coverage": percentile_score(table["marker_coverage"], True),
    })
    table["structural_reliability_score"] = structural.mean(axis=1)
    table["annotation_confidence_score"] = annotation.mean(axis=1)
    table["source_support_count"] = (
        table["combined_panglao_agreement"].astype(int) + table["combined_cellmarker_agreement"].astype(int)
    )
    table["reliability_score"] = table[["structural_reliability_score", "annotation_confidence_score"]].mean(axis=1)
    table["reliability_band"] = pd.cut(
        table["structural_reliability_score"], [-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["mixed_candidate", "uncertain", "reliable"],
    ).astype(str)
    table = table.sort_values("reliability_score", ascending=False)
    table.to_csv(args.out_dir / "cluster_reliability_no_official_labels.csv", index=False)
    config = {
        "official_labels_policy": "not_loaded",
        **database_audit,
        "marker_voting": "original_CDGE_bottom_up_utils" if args.database_policy == "original" else "human_unique_per_gene_class_source_balanced_rank_weighted",
        "database_policy": args.database_policy,
        "evidence_interpretation": "database agreement is corroboration, not statistical independence",
        "structural_components": list(structural.columns),
        "annotation_components": list(annotation.columns),
        "cluster_count": int(table.shape[0]),
    }
    (args.out_dir / "run_config_no_official_labels.json").write_text(json.dumps(config, indent=2))
    print(table[["cluster", "n_cells", "pseudo_label", "reliability_score"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
