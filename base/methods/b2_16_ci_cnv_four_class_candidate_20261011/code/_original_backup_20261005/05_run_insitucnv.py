"""Run the authors' InSituCNV workflow on the XRrun 20k local sample.

Official labels are intentionally absent. Reference categories are made only
from the previously selected high-confidence Inflammatory and Connective seeds.
The authors' workflow itself leaves tumor/normal assignment for manual review
after CNV heatmaps and CNV-based clustering are generated.

【pipeline 位置】第 5 步(CNV 证据入口)。输入 03 输出的 C/I 种子文件 + 基因坐标,
输出 adata_cnv.h5ad:每个细胞的 CNV 剖面(obsm["X_cnv"])和 CNV Leiden cluster
(obs["cnv_leiden_res0.1/0.2/0.3"])。这是全流程唯一引入"非表达证据"的地方;
CNV 结果只作为 Neoplastic 的辅助证据,绝不直接等于 Neoplastic。

【cluster/cell 边界】CNV 推断与聚类是细胞级的;但"哪些 CNV cluster 算肿瘤候选"
的判断在 06(审计)里以 cluster 为单位,最终逐细胞种子在 07 生成。

【关键稀疏 panel 适配】infercnvpy 默认 60 基因窗口在 280 基因 panel 上不可行,
这里固定为"染色体上 ≥3 个 panel 基因才可用,窗口 3、步长 1"。参考细胞只取
03 选出的 Inflammatory / Connective_tissue 种子(每类 ≥30 个),其余细胞都是
待观察的 "Observation"。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from reliability.data_contracts import validate_counts, usable_panel_coordinates, normalize_genomic_coordinates, CANONICAL_CHROMOSOMES


ROOT = Path(__file__).resolve().parents[1]
# The caller must pass --adata; no workstation-specific default.
ADATA = None


def build_gene_reference(coordinates_path: Path) -> pd.DataFrame:
    """Build the frozen sparse-panel coordinate table."""
    coordinates = pd.read_csv(coordinates_path, sep="\t")
    return normalize_genomic_coordinates(coordinates.rename(columns={"gene": "gene_name"}))


def validate_cnv_chromosomes(adata: sc.AnnData) -> None:
    invalid = set(adata.var["chromosome"].astype(str)) - set(CANONICAL_CHROMOSOMES)
    invalid |= set(adata.uns.get("cnv", {}).get("chr_pos", {})) - set(CANONICAL_CHROMOSOMES)
    if invalid:
        raise RuntimeError(f"CNV chromosome axes are malformed: {sorted(invalid)}")


def coordinate_audit(reference: pd.DataFrame, panel: pd.DataFrame) -> dict[str, object]:
    return {
        "coordinate_contract": "GRCh38_closed_positive_intervals_canonical_chr_v1",
        "raw_chromosome_counts": reference["chromosome_original"].value_counts().astype(int).to_dict(),
        "canonical_chromosome_counts": reference["chromosome"].value_counts().astype(int).to_dict(),
        "normalized_name_rows": int(reference["chromosome_original"].ne(reference["chromosome"]).sum()),
        "eligible_panel_genes": len(panel),
        "eligible_panel_chromosome_gene_counts": panel.groupby("chromosome", observed=True).size().astype(int).to_dict(),
        "official_labels_policy": "not_loaded",
    }


def smooth_mapped_expression(
    connectivities: sparse.spmatrix,
    normalized_mapped: sparse.spmatrix,
    all_gene_totals: np.ndarray,
    *,
    target_sum: float,
) -> np.ndarray:
    """Reproduce all-gene smoothing while materialising mapped genes only."""
    graph = sparse.csr_matrix(connectivities)
    mapped = sparse.csr_matrix(normalized_mapped)
    smoothed = (graph @ mapped).astype(np.float32).toarray()

    smoothed_all_gene_totals = np.asarray(graph @ np.asarray(all_gene_totals).ravel()).ravel()
    if np.any(smoothed_all_gene_totals <= 0):
        raise RuntimeError("CNV smoothing produced non-positive all-gene totals.")
    smoothed *= (target_sum / smoothed_all_gene_totals).astype(np.float32)[:, None]
    np.log1p(smoothed, out=smoothed)
    return smoothed


def prepare_cnv_input_memory_efficient(
    adata: sc.AnnData,
    gene_reference: pd.DataFrame,
    *,
    target_sum: float = 10_000,
    smoothing_neighbors: int = 20,
    neighbors_n_neighbors: int = 15,
    neighbors_n_pcs: int = 50,
) -> sc.AnnData:
    """Prepare the authors' CNV input without densifying the full gene panel."""
    from scvelo.preprocessing.neighbors import get_connectivities

    if "counts" not in adata.layers:
        raise KeyError("AnnData must contain raw counts in adata.layers['counts'].")
    raw_counts = sparse.csr_matrix(adata.layers["counts"])
    validate_counts(raw_counts)
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("CNV input cell and gene axes must be unique")
    adata.X = raw_counts

    panel_reference = usable_panel_coordinates(gene_reference, adata.var_names)
    reference_genes = set(panel_reference["gene_name"].astype(str))
    mapped_mask = np.asarray(adata.var_names.astype(str).isin(reference_genes), dtype=bool)
    if mapped_mask.sum() < 3:
        raise RuntimeError(f"Only {int(mapped_mask.sum())} genes have genomic coordinates.")

    scratch = ad.AnnData(
        X=raw_counts.copy(),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    sc.pp.normalize_total(scratch, target_sum=target_sum)
    normalized_mapped = sparse.csr_matrix(scratch.X[:, mapped_mask]).copy()
    all_gene_totals = np.asarray(scratch.X.sum(axis=1)).ravel()

    sc.pp.log1p(scratch)
    n_comps = max(2, min(neighbors_n_pcs, min(scratch.n_obs, scratch.n_vars) - 1))
    sc.pp.pca(scratch, n_comps=n_comps)
    sc.pp.neighbors(scratch, n_neighbors=neighbors_n_neighbors)
    graph = get_connectivities(
        scratch,
        mode="connectivities",
        n_neighbors=smoothing_neighbors,
        recurse_neighbors=False,
    )
    log_mapped = smooth_mapped_expression(
        graph,
        normalized_mapped,
        all_gene_totals,
        target_sum=target_sum,
    )

    out = ad.AnnData(
        X=log_mapped,
        obs=adata.obs.copy(),
        var=adata.var.loc[mapped_mask].copy(),
    )
    out.layers["counts"] = raw_counts[:, mapped_mask].copy()
    out.layers["log_norm"] = log_mapped.copy()
    if "spatial" in adata.obsm:
        out.obsm["spatial"] = np.asarray(adata.obsm["spatial"]).copy()
    out.obsm["X_pca"] = scratch.obsm["X_pca"]
    out.obsp["distances"] = scratch.obsp["distances"]
    out.obsp["connectivities"] = scratch.obsp["connectivities"]
    out.uns["neighbors"] = scratch.uns["neighbors"]
    out.uns["pca"] = scratch.uns.get("pca", {})
    out.uns["insitucnv"] = {
        "cnv_input": {
            "raw_layer": "counts",
            "normalized_layer": "computed_all_genes_not_retained",
            "smoothed_layer": "computed_for_coordinate_genes_not_retained",
            "log_layer": "log_norm",
            "target_sum": target_sum,
            "smoothing_neighbors": smoothing_neighbors,
            "smoothing_mode": "connectivities",
            "neighbors_built": True,
            "neighbors_n_neighbors": neighbors_n_neighbors,
            "neighbors_n_pcs": n_comps,
            "memory_strategy": "all-gene graph; coordinate-gene smoothing; analytic all-gene totals",
            "chromosome_panel_gene_counts": panel_reference.groupby("chromosome").size().astype(int).to_dict(),
        }
    }
    del scratch, normalized_mapped, all_gene_totals, log_mapped
    gc.collect()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument(
        "--non-tumor-seeds",
        type=Path,
        default=ROOT / "results/01_cluster_reliability/cell_level_refined_pseudo_labels.csv",
    )
    parser.add_argument(
        "--coordinates", type=Path, default=ROOT / "resources/ensembl_hg38_coordinates.tsv"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results/02_insitucnv/rerun"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import insitucnv as icv

    print("[1/4] Preparing raw counts and label-free reference categories", flush=True)
    adata = sc.read_h5ad(args.adata)
    if "counts" not in adata.layers:
        raise KeyError("The XRrun h5ad must contain raw counts in adata.layers['counts'].")
    seeds = pd.read_csv(
        args.non_tumor_seeds,
        usecols=["cell_id", "is_initial_seed", "refined_pseudo_label"],
    )
    seeds = seeds.loc[
        seeds["is_initial_seed"]
        & seeds["refined_pseudo_label"].isin(["Inflammatory", "Connective_tissue"])
    ]
    seed_label = dict(zip(seeds["cell_id"].astype(str), seeds["refined_pseudo_label"].astype(str)))
    adata.obs["cnv_reference_category"] = [seed_label.get(str(cell), "Observation") for cell in adata.obs_names]
    reference_categories = ["Inflammatory", "Connective_tissue"]
    present = adata.obs["cnv_reference_category"].value_counts().to_dict()
    if min(present.get(category, 0) for category in reference_categories) < 30:
        raise RuntimeError(f"Insufficient label-free reference seeds: {present}")

    # The h5ad has a preexisting neighbour graph from the original pipeline.
    # The authors' workflow builds its own graph from normalized raw counts, so
    # remove the old graph rather than silently reuse a different preprocessing.
    adata.uns.pop("neighbors", None)
    adata.obsp.pop("connectivities", None)
    adata.obsp.pop("distances", None)
    adata.obsm.pop("X_pca", None)

    gene_reference = build_gene_reference(args.coordinates)
    # infercnvpy expects chromosome names such as "chr1".  With a 280-gene
    # targeted panel, its default 60-gene window is impossible on most
    # chromosomes; retain only chromosomes with >=3 panel genes and use the
    # smallest meaningful 3-gene local window as an explicit sparse-panel
    # adaptation (not the paper's default high-plex setting).
    reference_path = args.out_dir / "gene_reference_grch38.csv"
    gene_reference.to_csv(reference_path, index=False)
    eligible_panel = usable_panel_coordinates(gene_reference, adata.var_names)
    (args.out_dir / "coordinate_contract_audit_no_official_labels.json").write_text(
        json.dumps(coordinate_audit(gene_reference, eligible_panel), indent=2), encoding="utf-8"
    )

    print("[2/4] Building the all-gene graph and memory-efficient mapped-gene smoothing", flush=True)
    cnv_adata = prepare_cnv_input_memory_efficient(
        adata,
        gene_reference,
        target_sum=10_000,
        smoothing_neighbors=20,
        neighbors_n_neighbors=15,
        neighbors_n_pcs=50,
    )
    del adata
    gc.collect()
    cnv_adata = icv.pp.add_genomic_positions(
        cnv_adata,
        reference=gene_reference,
        drop_unmapped_genes=False,
        copy=False,
    )
    validate_cnv_chromosomes(cnv_adata)

    print("[3/4] Running inferCNV and fixed-resolution CNV clustering", flush=True)
    icv.tl.run_infercnv(
        cnv_adata,
        reference_key="cnv_reference_category",
        reference_categories=reference_categories,
        input_layer="log_norm",
        window_size=3,
        step=1,
        lfc_clip=4.0,
        chunksize=1000,
        calculate_gene_values=True,
        copy=False,
    )
    icv.tl.compute_cnv_neighbors(cnv_adata, copy=False)
    cluster_keys = icv.tl.cluster_cnv_resolutions(
        cnv_adata,
        resolutions=[0.1, 0.2, 0.3],
        dendrogram=True,
        copy=False,
    )
    validate_cnv_chromosomes(cnv_adata)
    output_h5ad = args.out_dir / "adata_cnv.h5ad"
    cnv_adata.write_h5ad(output_h5ad, compression="gzip")
    run_summary = {
        "n_cells": int(cnv_adata.n_obs),
        "n_genes": int(cnv_adata.n_vars),
        "reference_key": "cnv_reference_category",
        "reference_categories": reference_categories,
        "raw_layer": "counts",
        "target_sum": 10_000,
        "smoothing_neighbors": 20,
        "smoothing_mode": "connectivities",
        "build_neighbors": True,
        "neighbors_n_neighbors": 15,
        "neighbors_n_pcs": 50,
        "window_size": 3,
        "step": 1,
        "lfc_clip": 4.0,
        "cluster_resolutions": [0.1, 0.2, 0.3],
        "cluster_keys": cluster_keys,
        "primary_cluster_key": "cnv_leiden_res0.2",
        "output_h5ad": str(output_h5ad),
        "memory_strategy": "all-gene graph; coordinate-gene smoothing; analytic all-gene totals",
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    print("[4/4] Writing provenance summary", flush=True)
    summary = {
        "method": "authors_InSituCNV_workflow",
        "official_labels_policy": "not_loaded",
        "reference_source": "high-confidence non-tumor seeds from label-free structural and marker evidence",
        "reference_categories": reference_categories,
        "reference_cell_counts": present,
        "raw_count_layer": "counts",
        "smoothing_neighbors": 20,
        "sparse_panel_cnv_window_size": 3,
        "sparse_panel_cnv_step": 1,
        "cnv_cluster_keys": cluster_keys,
        "memory_strategy": "all-gene graph; coordinate-gene smoothing; analytic all-gene totals",
        "important_note": "InSituCNV intentionally leaves tumor/normal assignment for manual review after CNV heatmaps and clustering.",
    }
    (args.out_dir / "cdge_provenance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
