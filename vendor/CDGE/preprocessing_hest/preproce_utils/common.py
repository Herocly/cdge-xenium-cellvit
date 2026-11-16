"""Shared constants and patch-processing utilities for top-down / bottom-up pipelines."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from rasterio import features
from shapely.affinity import translate
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_TYPE_ID_TO_NAME: dict[int, str] = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
    11: "Unlabeled",
}
NAME_TO_CELL_TYPE_ID: dict[str, int] = {v.lower(): k for k, v in CELL_TYPE_ID_TO_NAME.items()}


LABEL_MAP: dict[str, int] = {
    "background": 0,
    "neoplastic": 1,
    "inflammatory": 2,
    "connective": 3,
    "dead": 4,
    "epithelial": 5,
    "unlabeled": 11,
    "low_count": 12,
}

NORM_TISSUE_NAMES: dict[str, str] = {
    "Bowel": "Colon",
    "Ovary": "Ovarian",
    "Pancreas": "Pancreatic",
}

CATEGORY_TO_LABEL: dict[str, str] = {
    "Epithelial": "epithelial",
    "Epithelial_Neoplastic": "neoplastic",
    "Inflammatory": "inflammatory",
    "Connective_tissue": "connective",
    "Others": "unlabeled",
    "Unknown": "unlabeled",
}

VIS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 0, 255),
    5: (255, 191, 0),
    11: (128, 128, 128),
    12: (0, 255, 255),
}

# ---------------------------------------------------------------------------
# Index / ID helpers
# ---------------------------------------------------------------------------


def normalize_index(idx) -> pd.Index:
    """Decode bytes and coerce to int if possible."""
    s = pd.Index(idx).map(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        return pd.Index(numeric.astype(int))
    return pd.Index(s.astype(str))


def normalize_ids(values) -> np.ndarray:
    """Same as normalize_index but returns ndarray."""
    s = pd.Index(values).map(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).to_numpy()
    return s.astype(str).to_numpy()


def resolve_cell_type_id(cell_id) -> int | None:
    """Convert mixed cell_type_id (int/str) to an integer label."""
    try:
        return int(cell_id)
    except (ValueError, TypeError):
        pass
    if isinstance(cell_id, str):
        key = cell_id.strip().lower()
        if key in NAME_TO_CELL_TYPE_ID:
            return NAME_TO_CELL_TYPE_ID[key]
    return None


def count_unique_genes(transcript_path: Path) -> int:
    transcript_df = pd.read_parquet(transcript_path, columns=["feature_name"])
    genes = transcript_df["feature_name"].map(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
    )
    return genes.nunique()


def map_category_to_label(category: str) -> str:
    if category is None or pd.isna(category):
        return "background"
    return CATEGORY_TO_LABEL.get(str(category), "background")


# ---------------------------------------------------------------------------
# Progress persistence (resume support)
# ---------------------------------------------------------------------------


def load_progress(save_dir: Path) -> tuple[dict, dict, dict]:
    types_records: dict[str, str] = {}
    cell_count_records: dict[str, dict] = {}
    unlabeled_records: dict[str, dict] = {}

    types_path = save_dir / "types.csv"
    if types_path.exists():
        df = pd.read_csv(types_path)
        types_records = {row["img"]: row["type"] for row in df.to_dict("records")}

    cell_count_path = save_dir / "cell_count.csv"
    if cell_count_path.exists():
        df = pd.read_csv(cell_count_path)
        cell_count_records = {row["Image"]: row for row in df.to_dict("records")}

    unlabeled_path = save_dir / "unlabeled_patches.csv"
    if unlabeled_path.exists():
        df = pd.read_csv(unlabeled_path)
        unlabeled_records = {row["img"]: row for row in df.to_dict("records")}

    return types_records, cell_count_records, unlabeled_records


def write_progress(save_dir: Path, types_records: dict, cell_count_records: dict) -> None:
    pd.DataFrame([{"img": k, "type": v} for k, v in types_records.items()]).to_csv(save_dir / "types.csv", index=False)
    pd.DataFrame(cell_count_records.values()).to_csv(save_dir / "cell_count.csv", index=False)


def write_unlabeled(save_dir: Path, unlabeled_records: dict) -> None:
    if unlabeled_records:
        pd.DataFrame(unlabeled_records.values()).to_csv(save_dir / "unlabeled_patches.csv", index=False)


# ---------------------------------------------------------------------------
# Segmentation mask preparation
# ---------------------------------------------------------------------------


_EXCLUDE_OBS_NAMES = {"UNASSIGNED", "UNKNOWN", "NA", "N/A", "NONE"}
UNLABELED_CELL_TYPE_ID = LABEL_MAP["unlabeled"]  # 11


def prepare_segmentation_for_patches(adata, seg_mask):
    """Attach cell_type_id and centroids to seg_mask; build spatial index.

    obs_names that match placeholder labels (UNASSIGNED, UNKNOWN, etc.)
    or have no mapping are assigned ``UNLABELED_CELL_TYPE_ID`` (11).
    """
    seg_mask = seg_mask.copy()
    seg_mask.index = normalize_index(seg_mask.index)

    cell_ids = normalize_ids(adata.obs_names)
    cell_type_series = adata.obs["cell_type_id"].to_numpy()

    # Build id_map, skipping placeholder obs_names
    id_map: dict[str, int] = {}
    for idx, cid in enumerate(cell_ids):
        key = str(cid)
        if key.strip().upper() in _EXCLUDE_OBS_NAMES:
            id_map[key] = UNLABELED_CELL_TYPE_ID
        else:
            id_map[key] = cell_type_series[idx]

    mapped = seg_mask.index.astype(str).map(id_map)
    # Cells in seg_mask with no matching obs_name → unlabeled
    seg_mask["cell_type_id"] = mapped.fillna(UNLABELED_CELL_TYPE_ID).astype(int)

    centroids = seg_mask.geometry.centroid
    seg_mask["center_x"] = centroids.x
    seg_mask["center_y"] = centroids.y

    sindex = seg_mask.sindex
    return seg_mask, sindex


# ---------------------------------------------------------------------------
# Single-patch processing
# ---------------------------------------------------------------------------


def _extract_roi(seg_mask, sindex, x0, y0, x1, y1):
    """Return GeoDataFrame of cells whose centroid lies inside the patch."""
    candidate_idx = list(sindex.intersection((x0, y0, x1, y1)))
    if not candidate_idx:
        return seg_mask.iloc[[]]
    return seg_mask.iloc[candidate_idx].copy()
    # gdf = seg_mask.iloc[candidate_idx]
    # return gdf[
    #     (gdf["center_x"] > x0) & (gdf["center_x"] <= x1) & (gdf["center_y"] > y0) & (gdf["center_y"] <= y1)
    # ].copy()


def _build_shapes(gdf_roi, x0, y0):
    """Build semantic / instance shape lists from a ROI GeoDataFrame."""
    gdf_roi["geom_crop"] = gdf_roi.geometry.apply(lambda g: translate(g, xoff=-x0, yoff=-y0))
    sem_shapes = []
    inst_shapes = []
    cell_count_dict: dict[str, int] = {}
    num_cells = 0

    for geom, cell_id in zip(gdf_roi["geom_crop"], gdf_roi["cell_type_id"]):
        resolved_id = resolve_cell_type_id(cell_id)
        if resolved_id is None:
            continue
        sem_shapes.append((geom, resolved_id))
        inst_shapes.append((geom, num_cells))
        num_cells += 1
        name = CELL_TYPE_ID_TO_NAME.get(resolved_id, str(cell_id))
        cell_count_dict[name] = cell_count_dict.get(name, 0) + 1

    return sem_shapes, inst_shapes, cell_count_dict


def _save_overlay(wsi_crop, sem_shapes, patch_size, vis_dir, out_name):
    """Render and save a boundary-overlay visualization."""
    alpha = 128
    base = wsi_crop.convert("RGBA")
    h, w = patch_size, patch_size
    overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)

    outline_shapes = []
    line_width = 6.0
    for geom, cid in sem_shapes:
        if cid == 0:
            cid = 11
        thick_line = geom.boundary.buffer(line_width / 2.0)
        outline_shapes.append((thick_line, cid))

    type_boundary = features.rasterize(
        outline_shapes,
        out_shape=(patch_size, patch_size),
        fill=0,
        dtype="int32",
        all_touched=True,
    )

    for cid, color in VIS_COLORS.items():
        if cid == 0:
            continue
        mask = type_boundary == cid
        if not mask.any():
            continue
        overlay_arr[mask, :3] = color
        overlay_arr[mask, 3] = alpha

    overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
    composed = Image.alpha_composite(base, overlay_img)
    composed.save(f"{vis_dir}/{out_name}_overlay.png")


def process_patch(
    patch_idx: int,
    x0: float,
    y0: float,
    patch_size: int,
    seg_mask,
    sindex,
    slide,
    save_dir: Path,
    organ: str,
    types_records: dict,
    cell_count_records: dict,
    unlabeled_records: dict,
    vis_dir: Path | None = None,
    enable_vis: bool = True,
) -> None:
    x1, y1 = x0 + patch_size, y0 + patch_size
    out_name = f"{patch_idx:05d}"
    img_name = f"{out_name}.png"
    image_path = save_dir / "images" / img_name
    label_path = save_dir / "labels" / f"{out_name}.npy"

    # Skip if already processed
    # if image_path.is_file() and label_path.is_file() and img_name in types_records and img_name in cell_count_records:
    #     write_progress(save_dir, types_records, cell_count_records)
    #     return
    # if img_name in unlabeled_records:
    #     return

    # Crop WSI
    wsi_crop = slide.read_region((int(x0), int(y0)), level=0, size=(int(patch_size), int(patch_size)))
    wsi_crop.save(image_path)

    # Extract cells in ROI
    gdf_roi = _extract_roi(seg_mask, sindex, x0, y0, x1, y1)

    if gdf_roi.empty:
        sem_shapes, inst_shapes, cell_count_dict = [], [], {}
    else:
        sem_shapes, inst_shapes, cell_count_dict = _build_shapes(gdf_roi, x0, y0)

    if not sem_shapes:
        unlabeled_records[img_name] = {
            "img": img_name,
            "patch_idx": patch_idx,
            "x0": int(x0),
            "y0": int(y0),
            "reason": "no_nucleus",
        }
        write_unlabeled(save_dir, unlabeled_records)
        return

    # Rasterize masks
    type_mask = features.rasterize(
        sem_shapes, out_shape=(patch_size, patch_size), fill=0, dtype="int32", all_touched=True
    )
    inst_mask = features.rasterize(
        inst_shapes, out_shape=(patch_size, patch_size), fill=0, dtype="int32", all_touched=True
    )
    np.save(label_path, {"inst_map": inst_mask, "type_map": type_mask})

    # Overlay visualization
    if enable_vis and vis_dir is not None:
        try:
            _save_overlay(wsi_crop, sem_shapes, patch_size, vis_dir, out_name)
        except Exception:
            pass

    # Record progress
    types_records[img_name] = organ
    cell_count_records[img_name] = {
        "Image": img_name,
        "Neoplastic": cell_count_dict.get("Neoplastic", 0),
        "Inflammatory": cell_count_dict.get("Inflammatory", 0),
        "Connective": cell_count_dict.get("Connective", 0),
        "Dead": cell_count_dict.get("Dead", 0),
        "Epithelial": cell_count_dict.get("Epithelial", 0),
    }
    write_progress(save_dir, types_records, cell_count_records)


# ---------------------------------------------------------------------------
# Patch-level loop
# ---------------------------------------------------------------------------


def generate_paired_sample(
    adata,
    seg_mask,
    sindex,
    slide,
    coords: np.ndarray,
    save_dir: Path,
    organ: str,
    patch_size: int = 256,
    vis_dir: Path | None = None,
    enable_vis: bool = True,
) -> None:
    types_records, cell_count_records, unlabeled_records = load_progress(save_dir)

    for patch_idx, (x0, y0) in enumerate(tqdm(coords, desc="Patches", unit="patch")):
        process_patch(
            patch_idx=patch_idx,
            x0=x0,
            y0=y0,
            patch_size=patch_size,
            seg_mask=seg_mask,
            sindex=sindex,
            slide=slide,
            save_dir=save_dir,
            organ=organ,
            types_records=types_records,
            cell_count_records=cell_count_records,
            unlabeled_records=unlabeled_records,
            vis_dir=vis_dir,
            enable_vis=enable_vis,
        )


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs(base_dir: Path, save_dir: Path, sample_id: str, output_sample_id: str) -> None:
    (base_dir / "proc_adata").mkdir(parents=True, exist_ok=True)
    (base_dir / "vis_mask" / output_sample_id).mkdir(parents=True, exist_ok=True)
    (save_dir / sample_id / "images").mkdir(parents=True, exist_ok=True)
    (save_dir / sample_id / "labels").mkdir(parents=True, exist_ok=True)


def cp_configs(src_dir: Path, dest_dir: Path) -> None:
    import shutil

    shutil.copyfile(f"{src_dir}/dataset_config.yaml", f"{dest_dir}/dataset_config.yaml")
    shutil.copyfile(f"{src_dir}/weight_config.yaml", f"{dest_dir}/weight_config.yaml")


# ---------------------------------------------------------------------------
# Data loading with error handling
# ---------------------------------------------------------------------------


def load_sample(base_dir: Path, sample_id: str):
    """Load a single sample, returning None on known-recoverable errors."""
    from preprocessing_hest import read_dataset

    try:
        meta, thumb_img, slide, adata, seg, coords = read_dataset(base_dir, sample_id)
    except IndexError as exc:
        if "Positions outside range of features" not in str(exc):
            raise
        transcript_path = base_dir / "transcripts" / f"{sample_id}_transcripts.parquet"
        try:
            n_vars = count_unique_genes(transcript_path)
            print(f"[SKIP] {sample_id}: calculate_qc_metrics failed (n_vars={n_vars}).")
        except Exception as count_exc:
            print(f"[SKIP] {sample_id}: calculate_qc_metrics failed ({count_exc}).")
        return None

    if coords.shape[0] == 0:
        print(f"[SKIP] {sample_id}: no patch coordinates found.")
        return None

    organ = meta.get("tissue", meta.get("organ", "Unknown"))
    if organ in NORM_TISSUE_NAMES:
        organ = NORM_TISSUE_NAMES[organ]

    return meta, thumb_img, slide, adata, seg, coords, organ
