"""Bottom-up cluster classification: vote-based cell typing from marker DBs."""

import argparse
import warnings
from pathlib import Path

import random
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import scanpy as sc

from .common import LABEL_MAP, map_category_to_label

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Default DB paths
# ---------------------------------------------------------------------------

DEFAULT_PANGLAO_MARKER_PATH = ROOT_DIR / "data" / "marker_gene_list.tsv"
DEFAULT_PANGLAO_MAP_PATH = ROOT_DIR / "data" / "cell_type_mapping.csv"
DEFAULT_CELLMARKER_DB_PATH = ROOT_DIR / "data" / "database_gene_functions" / "Cell_marker_All.csv"
DEFAULT_CELLMARKER_MAP_PATH = ROOT_DIR / "data" / "cell_marker_mapping.csv"
DEFAULT_NEOPLASTIC_GENE_PATH = ROOT_DIR / "data" / "neoplastic_genes.csv"

# ---------------------------------------------------------------------------
# Vote categories
# ---------------------------------------------------------------------------

VOTE_CATEGORIES = ["Epithelial", "Inflammatory", "Connective_tissue", "Others"]
VOTE_IDX = {cat: idx for idx, cat in enumerate(VOTE_CATEGORIES)}

BREAST_EPITHELIAL_LINEAGE_GENES = {
    "EPCAM",
    "KRT5",
    "KRT6B",
    "KRT8",
    "KRT14",
    "KRT15",
    "KRT17",
    "KRT18",
    "KRT19",
    "KRT23",
    "COL17A1",
    "DSP",
    "CDH3",
    "GATA3",
    "FOXA1",
    "ESR1",
    "PGR",
    "AR",
    "ANKRD30A",
    "SLC39A6",
    "SLC40A1",
    "MLPH",
    "SCUBE2",
    "BCL2",
    "GPR160",
    "LTF",
    "PIGR",
    "KIT",
    "WFDC2",
    "CXCL14",
    "SEMA3C",
    "LAMB3",
    "LAMC2",
    "DST",
    "MYLK",
    "ACTA2",
    "TAGLN",
    "MYH11",
}

BREAST_TUMOR_SCORE_GENES = {
    "EPCAM",
    "CD24",
    "TACSTD2",
    "KRT6B",
    "KRT19",
    "KRT23",
    "FOXC1",
    "RARRES1",
    "CLDN4",
    "CEACAM6",
    "CXCL10",
    "PDZK1IP1",
    "NDRG1",
    "VEGFA",
    "VTCN1",
    "SFRP1",
    "ASPH",
    "SDC1",
    "PTN",
    "CHI3L1",
    "LDHA",
    "LDHB",
    "SLC2A1",
    "ENO1",
    "PGK1",
    "PKM",
    "PHGDH",
    "GAPDH",
    "MTHFD2",
    "HMGA1",
    "HDGF",
    "FADS2",
    "TOP2A",
    "BIRC5",
    "TYMS",
    "MYBL2",
    "TUBA1B",
    "SET",
}

BREAST_NORMAL_EPITHELIAL_SCORE_GENES = {
    "GATA3",
    "FOXA1",
    "ESR1",
    "PGR",
    "AR",
    "ANKRD30A",
    "SLC39A6",
    "SLC40A1",
    "MLPH",
    "SCUBE2",
    "BCL2",
    "GPR160",
    "LTF",
    "PIGR",
    "KIT",
    "KRT5",
    "KRT14",
    "COL17A1",
    "CXCL14",
    "SEMA3C",
    "MYLK",
    "ACTA2",
    "TAGLN",
    "MYH11",
    "IGFBP5",
    "DST",
}

BREAST_CONNECTIVE_ENDOTHELIAL_SCORE_GENES = {
    "PECAM1",
    "VWF",
    "PLVAP",
    "CLEC14A",
    "FLT1",
    "AQP1",
    "CDH5",
    "KDR",
    "ENG",
    "ESAM",
    "HSPG2",
    "COL4A1",
    "LAMB1",
    "LAMA4",
}

BREAST_CONNECTIVE_PERIVASCULAR_SCORE_GENES = {
    "RGS5",
    "PDGFRB",
    "ACTA2",
    "TAGLN",
    "MYH11",
    "MYLK",
    "NDUFA4L2",
    "HEYL",
    "OLFML2A",
    "PDGFD",
    "ITGB1",
    "TIMP1",
}

BREAST_CONNECTIVE_STROMAL_SCORE_GENES = {
    "MMP2",
    "CXCL12",
    "FBLN1",
    "C1R",
    "SFRP4",
    "CCDC80",
    "LOXL1",
    "LAMA2",
    "MMP14",
    "COL1A1",
    "COL1A2",
    "DCN",
    "LUM",
    "IGFBP3",
    "IL1R1",
}

# ---------------------------------------------------------------------------
# Organ mapping (HEST organ → DB organ/tissue names)
# ---------------------------------------------------------------------------

PANGLAO_ORGAN_MAP: dict[str, list[str]] = {
    "Lung": ["Lungs"],
    "Breast": ["Mammary gland"],
    "Colon": ["GI tract"],
    "Bowel": ["GI tract"],
    "Skin": ["Skin"],
    "Pancreas": ["Pancreas"],
    "Kidney": ["Kidney"],
    "Liver": ["Liver"],
    "Ovary": ["Reproductive"],
    "Prostate": ["Reproductive"],
}
PANGLAO_SHARED_ORGANS: list[str] = [
    "Immune system",
    "Connective tissue",
    "Vasculature",
    "Blood",
    "Bone",
]

CELLMARKER_TISSUE_MAP: dict[str, list[str]] = {
    "Lung": ["Lung"],
    "Breast": ["Breast"],
    "Colon": ["Colon", "Colon epithelium", "Colonic crypt"],
    "Bowel": ["Colon", "Colon epithelium", "Colonic crypt"],
    "Skin": ["Skin"],
    "Pancreas": ["Pancreas"],
    "Kidney": ["Kidney", "Kidney cortex"],
    "Liver": ["Liver"],
    "Ovary": ["Ovary"],
    "Prostate": ["Prostate"],
}


def _normalize_gene_symbol(gene) -> str:
    return str(gene).strip().upper()


def _split_cellmarker_symbols(value) -> list[str]:
    """Split CellMarker Symbol fields that contain multiple comma-separated genes."""
    if pd.isna(value):
        return []
    text = str(value).replace(";", ",").replace("|", ",")
    genes = []
    for token in text.split(","):
        gene = _normalize_gene_symbol(token)
        if not gene or gene in {"NA", "NAN", "NONE", "NULL"}:
            continue
        genes.append(gene)
    return genes


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------


def _filter_panglao(panglao_db: pd.DataFrame, organ: str) -> pd.DataFrame:
    """Filter PanglaoDB to organ-specific + shared organ markers."""
    organ_specific = PANGLAO_ORGAN_MAP.get(organ, [])
    allowed = organ_specific + PANGLAO_SHARED_ORGANS
    return panglao_db[panglao_db["organ"].isin(allowed)].copy()


def _filter_cellmarker(cellmarker_db: pd.DataFrame, organ: str) -> pd.DataFrame:
    """Filter CellMarker to tissue-specific markers."""
    tissues = CELLMARKER_TISSUE_MAP.get(organ, [])
    return cellmarker_db[cellmarker_db["tissue_type"].isin(tissues)].copy()


def load_databases(
    panglao_marker_file: Path = DEFAULT_PANGLAO_MARKER_PATH,
    panglao_map_file: Path = DEFAULT_PANGLAO_MAP_PATH,
    cellmarker_db_file: Path = DEFAULT_CELLMARKER_DB_PATH,
    cellmarker_map_file: Path = DEFAULT_CELLMARKER_MAP_PATH,
    neoplastic_gene_file: Path = DEFAULT_NEOPLASTIC_GENE_PATH,
    use_cellmarker: bool = True,
    use_neoplastic: bool = True,
    organ: str | None = None,
):
    panglao_db = pd.read_csv(panglao_marker_file, sep="\t")
    panglao_mapping = pd.read_csv(panglao_map_file)

    if organ and organ in PANGLAO_ORGAN_MAP:
        panglao_db = _filter_panglao(panglao_db, organ)

    cellmarker_db, cellmarker_mapping = None, None
    if use_cellmarker:
        cellmarker_db = pd.read_csv(cellmarker_db_file, sep="\t")
        cellmarker_db = cellmarker_db[cellmarker_db["cell_type"] == "Normal cell"]
        if organ and organ in CELLMARKER_TISSUE_MAP:
            cellmarker_db = _filter_cellmarker(cellmarker_db, organ)
        cellmarker_mapping = pd.read_csv(cellmarker_map_file)

    neoplastic_set = None
    if use_neoplastic:
        neoplastic_genes = pd.read_csv(neoplastic_gene_file)
        neoplastic_set = {_normalize_gene_symbol(gene) for gene in neoplastic_genes["gene"].tolist()}

    # Cancer cell marker set from CellMarker (organ-filtered if possible)
    cancer_marker_set = None
    if use_cellmarker and use_neoplastic:
        cancer_db = pd.read_csv(cellmarker_db_file, sep="\t")
        cancer_db = cancer_db[cancer_db["cell_type"] == "Cancer cell"]
        if organ and organ in CELLMARKER_TISSUE_MAP:
            cancer_db = _filter_cellmarker(cancer_db, organ)
        cancer_marker_set = set()
        for symbol in cancer_db["Symbol"].dropna():
            cancer_marker_set.update(_split_cellmarker_symbols(symbol))

    return panglao_db, cellmarker_db, panglao_mapping, cellmarker_mapping, neoplastic_set, cancer_marker_set


# ---------------------------------------------------------------------------
# Gene-vote building
# ---------------------------------------------------------------------------


def _build_category_map(mapping_df: pd.DataFrame, key_col: str) -> dict[str, np.ndarray]:
    """Map cell_type/cell_name -> vote vector."""
    cat_map: dict[str, np.ndarray] = {}
    for _, row in mapping_df.iterrows():
        key = row[key_col]
        if pd.isna(key):
            continue
        vec = np.zeros(len(VOTE_CATEGORIES), dtype=np.int32)
        for cat, idx in VOTE_IDX.items():
            if row.get(cat, 0) == 1:
                vec[idx] = 1
        cat_map[str(key)] = vec
    return cat_map


def build_gene_votes(
    panglao_db: pd.DataFrame,
    panglao_mapping: pd.DataFrame,
    cellmarker_db: pd.DataFrame | None = None,
    cellmarker_mapping: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Precompute gene -> vote vector."""
    gene_votes: dict[str, np.ndarray] = {}

    panglao_map = _build_category_map(panglao_mapping, "cell_type")
    for _, row in panglao_db.iterrows():
        gene, cell_type = row["official gene symbol"], row["cell type"]
        if pd.isna(gene) or pd.isna(cell_type):
            continue
        vec = panglao_map.get(str(cell_type))
        if vec is None:
            continue
        gkey = _normalize_gene_symbol(gene)
        gene_votes[gkey] = gene_votes.get(gkey, np.zeros_like(vec)) + vec

    if cellmarker_db is not None and cellmarker_mapping is not None:
        cm_map = _build_category_map(cellmarker_mapping, "cell_name")
        for _, row in cellmarker_db.iterrows():
            gene, cell_name = row["Symbol"], row["cell_name"]
            if pd.isna(gene) or pd.isna(cell_name):
                continue
            vec = cm_map.get(str(cell_name))
            if vec is None:
                continue
            for gkey in _split_cellmarker_symbols(gene):
                gene_votes[gkey] = gene_votes.get(gkey, np.zeros_like(vec)) + vec

    return gene_votes


# ---------------------------------------------------------------------------
# Cluster classification
# ---------------------------------------------------------------------------


def check_neoplastic(genes, neoplastic_set, top_n=20, threshold=0.25):
    top_genes = genes[:top_n]
    overlap = [g for g in top_genes if g in neoplastic_set]
    ratio = len(overlap) / len(top_genes) if top_genes else 0
    return {
        "is_neoplastic": ratio >= threshold,
        "neoplastic_ratio": ratio,
        "neoplastic_count": len(overlap),
        "total_checked": len(top_genes),
        "neoplastic_genes": overlap,
    }


def check_cancer_markers(genes, cancer_marker_set, top_n=20, threshold=0.20):
    """Vote-based cancer check using CellMarker cancer cell markers."""
    top_genes = genes[:top_n]
    overlap = [g for g in top_genes if g in cancer_marker_set]
    ratio = len(overlap) / len(top_genes) if top_genes else 0
    return {
        "is_cancer": ratio >= threshold,
        "cancer_ratio": ratio,
        "cancer_count": len(overlap),
        "total_checked": len(top_genes),
        "cancer_genes": overlap,
    }


def check_marker_overlap(genes, marker_set, top_n=20, threshold=0.20):
    top_genes = genes[:top_n]
    overlap = [g for g in top_genes if g in marker_set]
    ratio = len(overlap) / len(top_genes) if top_genes else 0
    return {
        "hit": ratio >= threshold,
        "ratio": ratio,
        "count": len(overlap),
        "total_checked": len(top_genes),
        "genes": overlap,
    }


def marker_rank_score(genes, marker_set, top_n=20):
    top_genes = genes[:top_n]
    total_weight = sum(range(1, len(top_genes) + 1))
    hit_weight = 0
    hits = []
    for rank, gene in enumerate(top_genes):
        if gene in marker_set:
            weight = top_n - rank
            hit_weight += weight
            hits.append(gene)
    score = hit_weight / total_weight if total_weight else 0.0
    return {"score": score, "count": len(hits), "genes": hits}


def compare_breast_epithelial_neoplastic_score(genes, top_n=20, margin=0.10, min_tumor_hits=3):
    tumor = marker_rank_score(genes, BREAST_TUMOR_SCORE_GENES, top_n=top_n)
    normal = marker_rank_score(genes, BREAST_NORMAL_EPITHELIAL_SCORE_GENES, top_n=top_n)
    delta = tumor["score"] - normal["score"]
    return {
        "is_neoplastic": tumor["count"] >= min_tumor_hits and delta >= margin,
        "tumor_score": tumor["score"],
        "normal_score": normal["score"],
        "score_delta": delta,
        "tumor_count": tumor["count"],
        "normal_count": normal["count"],
        "tumor_genes": tumor["genes"],
        "normal_genes": normal["genes"],
        "top_n": top_n,
        "margin": margin,
        "min_tumor_hits": min_tumor_hits,
    }


def compare_breast_connective_subtype_score(genes, top_n=20, threshold=0.12, min_hits=3):
    subtype_sets = {
        "endothelial_vascular": BREAST_CONNECTIVE_ENDOTHELIAL_SCORE_GENES,
        "perivascular_smooth_muscle": BREAST_CONNECTIVE_PERIVASCULAR_SCORE_GENES,
        "stromal_fibroblast": BREAST_CONNECTIVE_STROMAL_SCORE_GENES,
    }
    scores = {name: marker_rank_score(genes, marker_set, top_n=top_n) for name, marker_set in subtype_sets.items()}
    best_subtype = max(scores, key=lambda name: (scores[name]["score"], scores[name]["count"]))
    best = scores[best_subtype]
    return {
        "is_connective": best["score"] >= threshold and best["count"] >= min_hits,
        "best_subtype": best_subtype,
        "best_score": best["score"],
        "best_count": best["count"],
        "best_genes": best["genes"],
        "subtype_scores": {
            name: {
                "score": value["score"],
                "count": value["count"],
                "genes": value["genes"],
            }
            for name, value in scores.items()
        },
        "top_n": top_n,
        "threshold": threshold,
        "min_hits": min_hits,
    }


def classify_cluster(genes, gene_votes, top_n=10, weight_by_rank=True, no_unlabeled=False):
    """Classify a single cluster from its ranked gene list."""
    total = np.zeros(len(VOTE_CATEGORIES), dtype=np.int64)
    for rank, gene in enumerate(genes[:top_n]):
        weight = (top_n - rank) if weight_by_rank else 1
        votes = gene_votes.get(gene)
        if votes is not None:
            total += votes * weight

    total_sum = int(total.sum())
    if total_sum == 0:
        if no_unlabeled:
            cls_idx = random.randint(0, len(VOTE_CATEGORIES) - 2)  # "Unknown" を除いたランダム選択
            return {"predicted_category": VOTE_CATEGORIES[cls_idx], "confidence": 0.0, "total_votes": {}}
        else:
            return {"predicted_category": "Unknown", "confidence": 0.0, "total_votes": {}}

    if no_unlabeled:
        best_idx = int(np.argmax(total[:-1]))  # "Unknown" を除外して最大値を取る
    else:
        best_idx = int(np.argmax(total))
    return {
        "predicted_category": VOTE_CATEGORIES[best_idx],
        "confidence": float(total[best_idx] / total_sum),
        "total_votes": {cat: int(total[idx]) for cat, idx in VOTE_IDX.items()},
    }


def classify_all_clusters(
    adata,
    groupby,
    gene_votes,
    neoplastic_set=None,
    cancer_marker_set=None,
    top_n=10,
    weight_by_rank=True,
    neoplastic_check_top_n=20,
    neoplastic_threshold=0.25,
    cancer_marker_threshold=0.25,
    neoplastic_check_scope="epithelial",
    lineage_refinement="none",
    epithelial_lineage_top_n=20,
    epithelial_lineage_threshold=0.20,
    epithelial_neoplastic_score_mode="none",
    epithelial_neoplastic_score_top_n=20,
    epithelial_neoplastic_score_margin=0.10,
    epithelial_neoplastic_min_tumor_hits=3,
    connective_subtype_score_mode="none",
    connective_subtype_score_top_n=20,
    connective_subtype_score_threshold=0.12,
    connective_subtype_min_hits=3,
    gene_votes_fallback=None,
    min_vote_threshold=5,
    no_unlabeled=False,
    enable_vis=True,
):

    # "unassigned" 細胞を除外して DEG を実行
    valid_mask = adata.obs[groupby] != "unassigned"
    n_unassigned = int((~valid_mask).sum())
    if n_unassigned > 0:
        print(f"  rank_genes_groups: excluding {n_unassigned} unassigned cells from {adata.n_obs}")
    adata_valid = adata[valid_mask].copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        sc.tl.rank_genes_groups(adata_valid, groupby=groupby, method="wilcoxon")
    df = sc.get.rank_genes_groups_df(adata_valid, group=None).copy()

    results = {}
    for group, group_df in df.groupby("group", sort=False, observed=True):
        group_genes = group_df["names"].tolist()
        result = classify_cluster(
            group_genes, gene_votes, top_n=top_n, weight_by_rank=weight_by_rank, no_unlabeled=no_unlabeled
        )

        total_sum = sum(result["total_votes"].values()) if result["total_votes"] else 0
        result["used_fallback"] = False
        if total_sum < min_vote_threshold and gene_votes_fallback is not None:
            result = classify_cluster(group_genes, gene_votes_fallback, top_n=top_n, weight_by_rank=weight_by_rank)
            result["used_fallback"] = True

        # Log: base classification
        votes_str = ", ".join(f"{k}={v}" for k, v in result.get("total_votes", {}).items())
        fb_tag = " [fallback]" if result["used_fallback"] else ""

        if enable_vis:
            print(
                f"    cluster {group}: {result['predicted_category']} "
                f"(conf={result['confidence']:.2f}, {votes_str}){fb_tag}"
            )

        result["epithelial_lineage_check"] = None
        if lineage_refinement == "breast_epithelial":
            lineage = check_marker_overlap(
                group_genes,
                BREAST_EPITHELIAL_LINEAGE_GENES,
                top_n=epithelial_lineage_top_n,
                threshold=epithelial_lineage_threshold,
            )
            result["epithelial_lineage_check"] = lineage
            if enable_vis:
                print(
                    f"      epithelial_lineage_check: ratio={lineage['ratio']:.2f} "
                    f"({lineage['count']}/{lineage['total_checked']}), hit={lineage['hit']}"
                )
            if lineage["hit"] and result["predicted_category"] in ("Others", "Unknown", "Inflammatory"):
                result["predicted_category"] = "Epithelial"
                if enable_vis:
                    print("      -> Epithelial (breast epithelial lineage refinement)")

        result["connective_subtype_score_check"] = None
        if connective_subtype_score_mode == "breast_v5" and result["predicted_category"] in ("Others", "Unknown"):
            connective = compare_breast_connective_subtype_score(
                group_genes,
                top_n=connective_subtype_score_top_n,
                threshold=connective_subtype_score_threshold,
                min_hits=connective_subtype_min_hits,
            )
            result["connective_subtype_score_check"] = connective
            if enable_vis:
                print(
                    "      connective_subtype_score: "
                    f"subtype={connective['best_subtype']}, "
                    f"score={connective['best_score']:.2f}, "
                    f"hits={connective['best_count']}, "
                    f"hit={connective['is_connective']}"
                )
            if connective["is_connective"]:
                result["predicted_category"] = "Connective_tissue"
                result["connective_subtype"] = connective["best_subtype"]
                if enable_vis:
                    print(f"      -> Connective_tissue ({connective['best_subtype']} subtype score)")

        check_scope = str(neoplastic_check_scope).lower()
        should_check_neoplastic = (
            result["predicted_category"] == "Epithelial"
            or check_scope == "all"
            or (check_scope == "non_connective" and result["predicted_category"] != "Connective_tissue")
        )

        if should_check_neoplastic:
            is_neoplastic = False
            reasons = []

            # Check 1: neoplastic gene list
            result["neoplastic_check"] = None
            if neoplastic_set is not None:
                neo = check_neoplastic(
                    group_genes, neoplastic_set, top_n=neoplastic_check_top_n, threshold=neoplastic_threshold
                )
                result["neoplastic_check"] = neo
                if neo["is_neoplastic"]:
                    is_neoplastic = True
                    reasons.append("neoplastic_genes")
                if enable_vis:

                    print(
                        f"      neoplastic_check: ratio={neo['neoplastic_ratio']:.2f} "
                        f"({neo['neoplastic_count']}/{neo['total_checked']}), hit={neo['is_neoplastic']}"
                    )

            # Check 2: CellMarker cancer cell markers (OR condition)
            result["cancer_marker_check"] = None
            if cancer_marker_set is not None:
                cancer = check_cancer_markers(
                    group_genes, cancer_marker_set, top_n=neoplastic_check_top_n, threshold=cancer_marker_threshold
                )
                result["cancer_marker_check"] = cancer
                if cancer["is_cancer"]:
                    is_neoplastic = True
                    reasons.append("cancer_markers")
                if enable_vis:
                    print(
                        f"      cancer_marker_check: ratio={cancer['cancer_ratio']:.2f} "
                        f"({cancer['cancer_count']}/{cancer['total_checked']}), hit={cancer['is_cancer']}"
                    )

            result["epithelial_neoplastic_score_check"] = None
            if (
                not is_neoplastic
                and epithelial_neoplastic_score_mode == "breast_v4"
                and result["predicted_category"] == "Epithelial"
            ):
                score_check = compare_breast_epithelial_neoplastic_score(
                    group_genes,
                    top_n=epithelial_neoplastic_score_top_n,
                    margin=epithelial_neoplastic_score_margin,
                    min_tumor_hits=epithelial_neoplastic_min_tumor_hits,
                )
                result["epithelial_neoplastic_score_check"] = score_check
                if enable_vis:
                    print(
                        "      epithelial_neoplastic_score: "
                        f"tumor={score_check['tumor_score']:.2f}, "
                        f"normal={score_check['normal_score']:.2f}, "
                        f"delta={score_check['score_delta']:.2f}, "
                        f"tumor_hits={score_check['tumor_count']}, "
                        f"normal_hits={score_check['normal_count']}, "
                        f"hit={score_check['is_neoplastic']}"
                    )
                if score_check["is_neoplastic"]:
                    is_neoplastic = True
                    reasons.append("epithelial_neoplastic_score")

            if is_neoplastic:
                result["predicted_category"] = "Epithelial_Neoplastic"
                if enable_vis:
                    print(f"      -> Epithelial_Neoplastic (reasons: {', '.join(reasons)})")
        else:
            result["neoplastic_check"] = None
            result["cancer_marker_check"] = None
            result["epithelial_neoplastic_score_check"] = None

        # no_unlabeled: "Others" / "Unknown" を別カテゴリに強制割り当て
        if no_unlabeled and result["predicted_category"] in ("Others", "Unknown"):
            votes = result.get("total_votes", {})
            non_others = {k: v for k, v in votes.items() if k != "Others"}
            if non_others and max(non_others.values()) > 0:
                best_cat = max(non_others, key=non_others.get)
                total_votes_sum = sum(votes.values())
                result["predicted_category"] = best_cat
                result["confidence"] = non_others[best_cat] / total_votes_sum if total_votes_sum > 0 else 0.0
            else:
                result["predicted_category"] = "Connective_tissue"
                result["confidence"] = 0.0
            if enable_vis:
                print(f"      -> no_unlabeled: reassigned to {result['predicted_category']}")

        results[group] = result

    return results


# ---------------------------------------------------------------------------
# Output suffix
# ---------------------------------------------------------------------------


def build_output_suffix(use_cellmarker: bool, use_neoplastic: bool, args) -> str:
    parts = ["bottomup"]
    if use_cellmarker:
        parts.append("cellmarker")
    if use_neoplastic:
        parts.append("neoplastic")
    if getattr(args, "no_unlabeled", False):
        parts.append("nounlabeled")
    if getattr(args, "neoplastic_check_scope", "epithelial") != "epithelial":
        parts.append(f"neoscope-{args.neoplastic_check_scope}")
    if getattr(args, "lineage_refinement", "none") != "none":
        parts.append(f"lineage-{args.lineage_refinement}")
        parts.append(f"linthr{args.epithelial_lineage_threshold:.2f}")
    if getattr(args, "epithelial_neoplastic_score_mode", "none") != "none":
        parts.append(f"episcore-{args.epithelial_neoplastic_score_mode}")
        parts.append(f"epimargin{args.epithelial_neoplastic_score_margin:.2f}")
    if getattr(args, "connective_subtype_score_mode", "none") != "none":
        parts.append(f"connscore-{args.connective_subtype_score_mode}")
        parts.append(f"connthr{args.connective_subtype_score_threshold:.2f}")
    if getattr(args, "neoplastic_threshold", 0.25) != 0.25:
        parts.append(f"neothr{args.neoplastic_threshold:.2f}")
    if getattr(args, "cancer_marker_threshold", 0.25) != 0.25:
        parts.append(f"cancerthr{args.cancer_marker_threshold:.2f}")
    if args.min_confidence is not None:
        parts.append(f"conf{args.min_confidence:.2f}")
    if args.groupby != "leiden_res_4.00":
        res = args.groupby.split("_")[-1]
        parts.append(f"leiden{res}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# AnnData labeling (bottom-up)
# ---------------------------------------------------------------------------


def load_or_label_adata(
    base_dir: Path, sample_id: str, adata, args, organ: str = "Unknown", enable_vis: bool = True
) -> sc.AnnData:
    """Load cached labeled AnnData or perform bottom-up cluster classification."""
    label_suffix = build_output_suffix(args.use_cellmarker, args.use_neoplastic, args)
    labeled_path = base_dir / "proc_adata" / f"{sample_id}_{label_suffix}.h5ad"

    # if labeled_path.exists():
    #     return sc.read_h5ad(labeled_path)

    if args.groupby not in adata.obs:
        res = args.groupby.split("_")[-1]
        sc.tl.leiden(adata, key_added=args.groupby, resolution=float(res), flavor="igraph")
        # raise ValueError(f"groupby key not found in adata.obs: {args.groupby}")

    db_args = dict(
        panglao_marker_file=args.panglao_marker_file,
        panglao_map_file=args.panglao_map_file,
        cellmarker_db_file=args.cellmarker_db_file,
        cellmarker_map_file=args.cellmarker_map_file,
        neoplastic_gene_file=args.neoplastic_gene_file,
        use_cellmarker=args.use_cellmarker,
        use_neoplastic=args.use_neoplastic,
    )

    # Organ-filtered databases
    panglao_db, cellmarker_db, panglao_mapping, cellmarker_mapping, neoplastic_set, cancer_marker_set = load_databases(
        **db_args,
        organ=organ,
    )
    gene_votes = build_gene_votes(
        panglao_db=panglao_db,
        panglao_mapping=panglao_mapping,
        cellmarker_db=cellmarker_db,
        cellmarker_mapping=cellmarker_mapping,
    )
    if cancer_marker_set is not None:
        print(f"  Cancer marker set: {len(cancer_marker_set)} genes (organ={organ})")

    # Unfiltered databases for fallback
    gene_votes_fallback = None
    if organ in PANGLAO_ORGAN_MAP or organ in CELLMARKER_TISSUE_MAP:
        panglao_all, cellmarker_all, panglao_map_all, cellmarker_map_all, _, _ = load_databases(
            **db_args,
            organ=None,
        )
        gene_votes_fallback = build_gene_votes(
            panglao_db=panglao_all,
            panglao_mapping=panglao_map_all,
            cellmarker_db=cellmarker_all,
            cellmarker_mapping=cellmarker_map_all,
        )
        print(f"  Organ-filtered gene votes: {len(gene_votes)} genes, fallback: {len(gene_votes_fallback)} genes")

    results = classify_all_clusters(
        adata,
        groupby=args.groupby,
        gene_votes=gene_votes,
        neoplastic_set=neoplastic_set,
        cancer_marker_set=cancer_marker_set,
        top_n=args.top_n,
        weight_by_rank=args.weight_by_rank,
        neoplastic_check_top_n=args.neoplastic_check_top_n,
        neoplastic_threshold=args.neoplastic_threshold,
        cancer_marker_threshold=args.cancer_marker_threshold,
        neoplastic_check_scope=getattr(args, "neoplastic_check_scope", "epithelial"),
        lineage_refinement=getattr(args, "lineage_refinement", "none"),
        epithelial_lineage_top_n=getattr(args, "epithelial_lineage_top_n", 20),
        epithelial_lineage_threshold=getattr(args, "epithelial_lineage_threshold", 0.20),
        epithelial_neoplastic_score_mode=getattr(args, "epithelial_neoplastic_score_mode", "none"),
        epithelial_neoplastic_score_top_n=getattr(args, "epithelial_neoplastic_score_top_n", 20),
        epithelial_neoplastic_score_margin=getattr(args, "epithelial_neoplastic_score_margin", 0.10),
        epithelial_neoplastic_min_tumor_hits=getattr(args, "epithelial_neoplastic_min_tumor_hits", 3),
        connective_subtype_score_mode=getattr(args, "connective_subtype_score_mode", "none"),
        connective_subtype_score_top_n=getattr(args, "connective_subtype_score_top_n", 20),
        connective_subtype_score_threshold=getattr(args, "connective_subtype_score_threshold", 0.12),
        connective_subtype_min_hits=getattr(args, "connective_subtype_min_hits", 3),
        gene_votes_fallback=gene_votes_fallback,
        min_vote_threshold=args.min_vote_threshold,
        no_unlabeled=getattr(args, "no_unlabeled", False),
        enable_vis=enable_vis,
    )

    n_fallback = sum(1 for v in results.values() if v.get("used_fallback"))
    if n_fallback > 0:
        print(f"  {n_fallback}/{len(results)} clusters used fallback (unfiltered) gene votes.")

    # Summary
    from collections import Counter

    cat_counts = Counter(v["predicted_category"] for v in results.values())
    summary = ", ".join(f"{k}={v}" for k, v in cat_counts.most_common())
    print(f"  Classification summary ({len(results)} clusters): {summary}")

    cluster_to_category = {str(k): v["predicted_category"] for k, v in results.items()}
    cluster_to_confidence = {str(k): v["confidence"] for k, v in results.items()}
    adata.obs["cell_category"] = adata.obs[args.groupby].map(cluster_to_category)
    adata.obs["classification_confidence"] = adata.obs[args.groupby].map(cluster_to_confidence).astype(float)
    adata.obs["coarse_cell_type"] = adata.obs["cell_category"].map(map_category_to_label)
    adata.obs["cell_type_id"] = [LABEL_MAP[lbl] for lbl in adata.obs["coarse_cell_type"]]

    # Confidence-aware filtering: low-confidence cells -> unlabeled
    min_confidence = getattr(args, "min_confidence", None)
    if min_confidence is not None and min_confidence > 0:
        low_conf_mask = adata.obs["classification_confidence"] < min_confidence
        n_filtered = low_conf_mask.sum()
        if n_filtered > 0:
            adata.obs.loc[low_conf_mask, "coarse_cell_type"] = "unlabeled"
            adata.obs.loc[low_conf_mask, "cell_type_id"] = LABEL_MAP["unlabeled"]
            print(
                f"  Confidence filtering: {n_filtered}/{adata.n_obs} cells "
                f"(conf < {min_confidence:.2f}) -> unlabeled"
            )

    adata.uns["bottomup_classification"] = {
        "groupby": args.groupby,
        "top_n": args.top_n,
        "weight_by_rank": bool(args.weight_by_rank),
        "use_cellmarker": bool(args.use_cellmarker),
        "use_neoplastic": bool(args.use_neoplastic),
        "neoplastic_check_top_n": args.neoplastic_check_top_n,
        "neoplastic_threshold": args.neoplastic_threshold,
        "organ": organ,
        "min_vote_threshold": args.min_vote_threshold,
        "cancer_marker_threshold": args.cancer_marker_threshold,
        "neoplastic_check_scope": getattr(args, "neoplastic_check_scope", "epithelial"),
        "lineage_refinement": getattr(args, "lineage_refinement", "none"),
        "epithelial_lineage_top_n": getattr(args, "epithelial_lineage_top_n", 20),
        "epithelial_lineage_threshold": getattr(args, "epithelial_lineage_threshold", 0.20),
        "epithelial_neoplastic_score_mode": getattr(args, "epithelial_neoplastic_score_mode", "none"),
        "epithelial_neoplastic_score_top_n": getattr(args, "epithelial_neoplastic_score_top_n", 20),
        "epithelial_neoplastic_score_margin": getattr(args, "epithelial_neoplastic_score_margin", 0.10),
        "epithelial_neoplastic_min_tumor_hits": getattr(args, "epithelial_neoplastic_min_tumor_hits", 3),
        "connective_subtype_score_mode": getattr(args, "connective_subtype_score_mode", "none"),
        "connective_subtype_score_top_n": getattr(args, "connective_subtype_score_top_n", 20),
        "connective_subtype_score_threshold": getattr(args, "connective_subtype_score_threshold", 0.12),
        "connective_subtype_min_hits": getattr(args, "connective_subtype_min_hits", 3),
        "min_confidence": min_confidence,
    }

    adata.write_h5ad(labeled_path)
    return adata


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired images/labels using bottom-up cluster classification."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("./data/hest_data"))
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--coords-file", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("proc_data/hest_data"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--groupby", type=str, default="leiden_res_4.00")
    parser.add_argument("--top-n", type=int, default=10)

    parser.add_argument("--weight-by-rank", action="store_true")
    parser.add_argument("--no-weight-by-rank", dest="weight_by_rank", action="store_false")

    parser.add_argument("--neoplastic-check-top-n", type=int, default=20)
    parser.add_argument("--neoplastic-threshold", type=float, default=0.25)
    parser.add_argument(
        "--min-vote-threshold",
        type=int,
        default=5,
        help="Minimum total votes before falling back to unfiltered markers.",
    )
    parser.add_argument(
        "--cancer-marker-threshold",
        type=float,
        default=0.25,
        help="Threshold for CellMarker cancer cell marker overlap ratio.",
    )
    parser.add_argument(
        "--neoplastic-check-scope",
        choices=["epithelial", "all", "non_connective"],
        default="epithelial",
        help="Which clusters should receive neoplastic/cancer-marker refinement.",
    )
    parser.add_argument(
        "--lineage-refinement",
        choices=["none", "breast_epithelial"],
        default="none",
        help="Optional lineage refinement before neoplastic refinement.",
    )
    parser.add_argument("--epithelial-lineage-top-n", type=int, default=20)
    parser.add_argument("--epithelial-lineage-threshold", type=float, default=0.20)
    parser.add_argument(
        "--epithelial-neoplastic-score-mode",
        choices=["none", "breast_v4"],
        default="none",
        help="Optional tumor-vs-normal epithelial score comparison inside epithelial lineage.",
    )
    parser.add_argument("--epithelial-neoplastic-score-top-n", type=int, default=20)
    parser.add_argument("--epithelial-neoplastic-score-margin", type=float, default=0.10)
    parser.add_argument("--epithelial-neoplastic-min-tumor-hits", type=int, default=3)
    parser.add_argument(
        "--connective-subtype-score-mode",
        choices=["none", "breast_v5"],
        default="none",
        help="Optional connective subtype score for rescuing unlabeled vascular/stromal clusters.",
    )
    parser.add_argument("--connective-subtype-score-top-n", type=int, default=20)
    parser.add_argument("--connective-subtype-score-threshold", type=float, default=0.12)
    parser.add_argument("--connective-subtype-min-hits", type=int, default=3)

    parser.add_argument("--use-cellmarker", action="store_true")
    parser.add_argument("--no-cellmarker", dest="use_cellmarker", action="store_false")
    parser.add_argument("--use-neoplastic", action="store_true")
    parser.add_argument("--no-neoplastic", dest="use_neoplastic", action="store_false")
    parser.set_defaults(use_cellmarker=True, use_neoplastic=True, weight_by_rank=True)

    parser.add_argument("--panglao-marker-file", type=Path, default=DEFAULT_PANGLAO_MARKER_PATH)
    parser.add_argument("--panglao-map-file", type=Path, default=DEFAULT_PANGLAO_MAP_PATH)
    parser.add_argument("--cellmarker-db-file", type=Path, default=DEFAULT_CELLMARKER_DB_PATH)
    parser.add_argument("--cellmarker-map-file", type=Path, default=DEFAULT_CELLMARKER_MAP_PATH)
    parser.add_argument("--neoplastic-gene-file", type=Path, default=DEFAULT_NEOPLASTIC_GENE_PATH)

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum classification confidence. Cells in clusters below this threshold "
        "are set to unlabeled and excluded from loss. e.g. 0.4",
    )

    parser.add_argument(
        "--no-unlabeled",
        dest="no_unlabeled",
        action="store_false",
        help="Others/Unknown クラスタを unlabeled に割り当てず、"
        "次点の非Others カテゴリ（票なし時は Connective_tissue）に強制割り当てる．",
    )

    parser.add_argument("--no-vis", dest="enable_vis", action="store_false")
    parser.add_argument(
        "--debug", dest="debug_mode", action="store_true", help="Disable skip logic; raise errors immediately."
    )

    parser.add_argument(
        "--use-img-adata",
        action="store_true",
        help="img_adata の morpho_cluster を groupby として使用する．"
        "保存先は proc_data/hest_data_image-based になる．",
    )
    parser.add_argument(
        "--img-adata-dir",
        type=Path,
        default=ROOT_DIR / "data" / "hest_data" / "img_adata",
        help="img_adata ディレクトリ",
    )

    parser.set_defaults(enable_vis=True, debug_mode=False, use_img_adata=False, no_unlabeled=False)

    return parser.parse_args()
