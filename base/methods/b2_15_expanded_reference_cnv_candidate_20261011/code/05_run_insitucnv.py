"""Run the authors' InSituCNV workflow on the XRrun 20k local sample.

Official labels are intentionally absent. Reference categories are made only
from the previously selected high-confidence Inflammatory and Connective seeds.
The authors' workflow itself leaves tumor/normal assignment for manual review
after CNV heatmaps and CNV-based clustering are generated.

【pipeline 位置】第 5 步(CNV 证据入口)。输入 03 输出的 C/I 种子文件 + 基因坐标,
输出 adata_cnv.h5ad:每个细胞的 CNV 剖面(obsm["X_cnv"])和 CNV Leiden cluster
(obs["cnv_leiden_res0.1/0.2/0.3"])。这里引入染色体位置约束的 RNA-derived CNV;
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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from reliability.data_contracts import validate_counts, usable_panel_coordinates, normalize_genomic_coordinates, CANONICAL_CHROMOSOMES, aligned_cell_table, boolean_flags


ROOT = Path(__file__).resolve().parents[1]
# The caller must pass --adata; no workstation-specific default.
ADATA = None

NORMAL = ("Connective_tissue", "Inflammatory", "Epithelial")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reference_categories_from_table(frame: pd.DataFrame, cells: object,
                                    classes: list[str], source: str) -> np.ndarray:
    """Explicit source contract; never disguise expansion labels as initial seeds."""
    if not classes or len(classes) != len(set(classes)) or not set(classes) <= set(NORMAL):
        raise ValueError("Reference classes must be distinct normal classes")
    aligned = aligned_cell_table(frame, cells)
    if source == "initial":
        chosen = boolean_flags(aligned["is_initial_seed"])
        column = "training_label" if "training_label" in aligned else "refined_pseudo_label"
        labels = aligned[column].astype(str).to_numpy()
        if "training_label" in aligned and "refined_pseudo_label" in aligned:
            other = aligned["refined_pseudo_label"].astype(str).to_numpy()
            if np.any(chosen & (labels != other)):
                raise ValueError("Initial seed label columns disagree")
    elif source in {"expanded", "strict"}:
        labels = aligned["normal_label"].astype(str).to_numpy()
        chosen = np.isin(labels, NORMAL)
        if source == "strict":
            chosen &= boolean_flags(aligned["is_strict_cnv_reference_candidate"])
    else:
        raise ValueError("Unknown reference source")
    return np.where(chosen & np.isin(labels, classes), labels, "Observation")


def evaluate_reference_experiment(out: Path, evaluation_path: Path, manifest_path: Path) -> None:
    """Evaluate fixed candidate files only; never modify reference membership."""
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Experiment input changed")
    for item in manifest["outputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Generated candidate changed")
    truth = pd.read_csv(evaluation_path, dtype={"cell_id": str})[["cell_id", "truth"]]
    evidence = pd.read_csv(out / "neoplastic_evidence/neoplastic_evidence_table_no_official_labels.csv", dtype={"cell_id": str})
    if truth.cell_id.duplicated().any() or evidence.cell_id.duplicated().any():
        raise ValueError("Evaluation and predictions require unique cell axes")
    if not set(truth.cell_id) <= set(evidence.cell_id):
        raise ValueError("Evaluation has cells outside predictions")
    merged = evidence.merge(truth, on="cell_id", how="left", validate="one_to_one")
    usable = merged.truth.isin((*NORMAL, "Neoplastic"))
    total = int((usable & merged.truth.eq("Neoplastic")).sum())
    rows = []
    for column in ("variant_A_gene_only", "variant_B_cnv_only", "variant_C_gene_plus_cnv", "variant_D_all_evidence"):
        selected = boolean_flags(merged[column])
        called = selected & usable.to_numpy()
        correct = called & merged.truth.eq("Neoplastic").to_numpy()
        count, tp = int(called.sum()), int(correct.sum())
        rows.append({"variant": column, "candidate_cells": int(selected.sum()), "evaluable_candidates": count,
                     "correct_neoplastic": tp, "precision": tp / count if count else None,
                     "correct_neoplastic_coverage": tp / total if total else None,
                     "f1_unknown_as_missed": 2 * tp / (count + total) if count + total else None,
                     "epithelial_contamination": int((called & merged.truth.eq("Epithelial").to_numpy()).sum()),
                     "reference_overlap": int((selected & boolean_flags(merged.is_cnv_reference)).sum())})
    pd.DataFrame(rows).to_csv(out / "neoplastic_evaluation_only.csv", index=False)
    seeds = pd.read_csv(out / "neoplastic_seeds/neoplastic_seed_candidates.csv", dtype={"cell_id": str})
    fixed = seeds.merge(truth, on="cell_id", how="left", validate="one_to_one")
    called = boolean_flags(fixed.is_neoplastic_seed) & fixed.truth.isin((*NORMAL, "Neoplastic")).to_numpy()
    tp = int((called & fixed.truth.eq("Neoplastic").to_numpy()).sum())
    pd.DataFrame([{"evaluable_neoplastic_seeds": int(called.sum()), "correct_neoplastic_seeds": tp,
                   "precision": tp / called.sum() if called.any() else None,
                   "correct_neoplastic_coverage": tp / total if total else None}]).to_csv(
                       out / "neoplastic_seed_evaluation_only.csv", index=False)
    reference = pd.read_csv(out / "insitucnv/reference_membership_no_official_labels.csv", dtype={"cell_id": str})
    reference = reference.merge(truth, on="cell_id", how="left", validate="one_to_one")
    pd.crosstab(reference.cnv_reference_category, reference.truth).to_csv(out / "reference_composition_evaluation_only.csv")
    (out / "evaluation_provenance_only.json").write_text(json.dumps({
        "source": str(evaluation_path), "source_sha256": sha256(evaluation_path),
        "prediction_fixed_manifest": str(manifest_path), "policy": "evaluation_only_no_reference_filtering",
    }, indent=2) + "\n")


def run_reference_experiment(output: Path) -> None:
    """Run three reference policies on each slide, serially and without label tuning."""
    project = ROOT.parents[1]
    output.mkdir(parents=True, exist_ok=False)
    code_files = list((ROOT / "code").rglob("*.py"))
    code_hashes = {str(path): sha256(path) for path in code_files}
    state = {"status": "running", "runs": {}, "code_hashes": code_hashes}
    def save_state() -> None:
        temp = output / "status.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(output / "status.json")
    def run(command: list[str], log: object) -> None:
        if any(sha256(Path(path)) != value for path, value in code_hashes.items()):
            raise ValueError("Running package changed")
        print(json.dumps({"stage_command": command}), flush=True)
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    save_state()
    try:
        for slide in ("rep1", "pancreas"):
            expanded_out = project / "outputs/b2_14_teacher_off_expansion_20261011" / slide / "normal_reference"
            original_manifest = json.loads((project / "outputs" / slide / "b2_12_feedback_audit_20261011_frozen_classifier/normal_reference/generation_manifest_no_official_labels.json").read_text())
            expanded_manifest = json.loads((expanded_out / "generation_manifest_no_official_labels.json").read_text())
            inputs = original_manifest["inputs"]
            for name in ("adata", "normal_seed_source", "cluster_evidence"):
                if sha256(Path(inputs[name]["path"])) != inputs[name]["sha256"]:
                    raise ValueError("Baseline input changed")
            expanded = expanded_out / "normal_reference_no_official_labels.csv"
            if sha256(expanded) != expanded_manifest["normal_reference_sha256"]:
                raise ValueError("Expansion changed after fixation")
            policies = [
                ("original_ci", Path(inputs["normal_seed_source"]["path"]), "initial", list(NORMAL[:2])),
                ("expanded_strict_ci", expanded, "strict", list(NORMAL[:2])),
                ("expanded_all_cie", expanded, "expanded", list(NORMAL)),
            ]
            for arm, source, policy, classes in policies:
                key = slide + "/" + arm
                out = output / slide / arm
                out.mkdir(parents=True)
                state["runs"][key] = "cnv_generation"
                save_state()
                with (out / "run.log").open("w") as log:
                    cnv_dir = out / "insitucnv"
                    run([sys.executable, str(Path(__file__)), "--adata", inputs["adata"]["path"],
                         "--non-tumor-seeds", str(source), "--reference-source", policy,
                         "--reference-classes", *classes, "--coordinates", str(ROOT / "resources/ensembl_hg38_coordinates.tsv"),
                         "--out-dir", str(cnv_dir)], log)
                    state["runs"][key] = "evidence_and_seeds"
                    save_state()
                    run([sys.executable, str(ROOT / "code/06_audit_cnv_candidates.py"), "--cnv-adata", str(cnv_dir / "adata_cnv.h5ad"),
                         "--out-dir", str(out / "cnv_audit")], log)
                    evidence_dir = out / "neoplastic_evidence"
                    run([sys.executable, str(ROOT / "code/09_build_neoplastic_evidence_ablation.py"),
                         "--cnv-adata", str(cnv_dir / "adata_cnv.h5ad"), "--expression-adata", inputs["adata"]["path"],
                         "--cluster-table", inputs["cluster_evidence"]["path"], "--cnv-audit", str(out / "cnv_audit/cnv_candidate_audit_label_free.csv"),
                         "--neoplastic-genes", str(ROOT / "resources/marker_databases/neoplastic_genes.csv"),
                         "--out-dir", str(evidence_dir)], log)
                    run([sys.executable, str(ROOT / "code/07_export_neoplastic_seeds.py"), "--evidence-table",
                         str(evidence_dir / "neoplastic_evidence_table_no_official_labels.csv"),
                         "--out-dir", str(out / "neoplastic_seeds")], log)
                fixed_inputs = [Path(inputs["adata"]["path"]), Path(inputs["cluster_evidence"]["path"]), source,
                                ROOT / "resources/ensembl_hg38_coordinates.tsv", ROOT / "resources/marker_databases/neoplastic_genes.csv", *code_files]
                fixed_outputs = [cnv_dir / "adata_cnv.h5ad", cnv_dir / "reference_membership_no_official_labels.csv",
                                 evidence_dir / "neoplastic_evidence_table_no_official_labels.csv",
                                 out / "neoplastic_seeds/neoplastic_seed_candidates.csv"]
                manifest_path = out / "generation_manifest_no_official_labels.json"
                manifest_path.write_text(json.dumps({"reference_policy": policy, "reference_classes": classes,
                    "official_labels_policy": "not_loaded_during_generation", "epithelial_conflict_gate": "not_enabled_same_as_B1.9",
                    "inputs": {str(i): {"path": str(p), "sha256": sha256(p)} for i, p in enumerate(fixed_inputs)},
                    "outputs": {str(i): {"path": str(p), "sha256": sha256(p)} for i, p in enumerate(fixed_outputs)},
                }, indent=2) + "\n")
                state["runs"][key] = "evaluation_only"
                save_state()
                provenance = json.loads((expanded_out / "evaluation_provenance_only.json").read_text())
                evaluation = Path(provenance["evaluation_only_source"])
                if sha256(evaluation) != provenance["evaluation_only_source_sha256"]:
                    raise ValueError("Evaluation source changed")
                evaluate_reference_experiment(out, evaluation, manifest_path)
                state["runs"][key] = "completed"
                save_state()
        comparisons = []
        for slide in ("rep1", "pancreas"):
            for arm in ("original_ci", "expanded_strict_ci", "expanded_all_cie"):
                frame = pd.read_csv(output / slide / arm / "neoplastic_evaluation_only.csv")
                frame.insert(0, "arm", arm)
                frame.insert(0, "slide", slide)
                comparisons.append(frame)
        pd.concat(comparisons, ignore_index=True).to_csv(output / "neoplastic_reference_comparison_evaluation_only.csv", index=False)
        state["status"] = "completed"
    except Exception as error:
        state["status"], state["error"] = "failed", repr(error)
        raise
    finally:
        save_state()


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
    parser.add_argument("--reference-source", choices=("initial", "expanded", "strict"), default="initial")
    parser.add_argument("--reference-classes", nargs="+", choices=NORMAL, default=["Inflammatory", "Connective_tissue"])
    parser.add_argument("--run-reference-experiment", action="store_true")
    args = parser.parse_args()
    if args.run_reference_experiment:
        run_reference_experiment(args.out_dir)
        return
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite existing CNV output")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import insitucnv as icv

    print("[1/4] Preparing raw counts and label-free reference categories", flush=True)
    adata = sc.read_h5ad(args.adata)
    if "counts" not in adata.layers:
        raise KeyError("The XRrun h5ad must contain raw counts in adata.layers['counts'].")
    seeds = pd.read_csv(args.non_tumor_seeds, dtype={"cell_id": str})
    adata.obs["cnv_reference_category"] = reference_categories_from_table(
        seeds, adata.obs_names, args.reference_classes, args.reference_source)
    reference_categories = args.reference_classes
    present = adata.obs["cnv_reference_category"].value_counts().to_dict()
    if min(present.get(category, 0) for category in reference_categories) < 30:
        raise RuntimeError(f"Insufficient label-free reference seeds: {present}")
    pd.DataFrame({"cell_id": adata.obs_names.astype(str),
                  "cnv_reference_category": adata.obs["cnv_reference_category"].astype(str).to_numpy()}).to_csv(
        args.out_dir / "reference_membership_no_official_labels.csv", index=False)

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
        "reference_source_policy": args.reference_source,
        "reference_table": str(args.non_tumor_seeds.resolve()),
        "reference_table_sha256": sha256(args.non_tumor_seeds),
        "reference_membership_sha256": sha256(args.out_dir / "reference_membership_no_official_labels.csv"),
        "reference_purity": "candidate reference, not independently certified normal or diploid",
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
