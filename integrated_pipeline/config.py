"""Load and validate one experiment configuration; paths are relative to the YAML."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable: {value}")
    path = Path(expanded)
    # Preserve venv/bin/python symlinks. Path.resolve() follows them to the
    # shared uv interpreter and silently drops the virtual environment.
    return Path(os.path.abspath(base / path if not path.is_absolute() else path))


def load(path: Path) -> dict:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise ValueError("YAML requires a samples list")
    if not raw["samples"]:
        raise ValueError("No samples configured")
    root = path.parent
    for key in ("output_root", "hest_dir", "cdge_root", "image_python", "analysis_python", "cnv_python", "cellvit_python", "cellvit_config"):
        if key in raw:
            raw[key] = resolve(root, raw[key])
    for key in ("pretrained_checkpoint", "external_test_dataset"):
        if raw.get("cellvit", {}).get(key):
            raw["cellvit"][key] = resolve(root, raw["cellvit"][key])
    ids = set()
    for sample in raw["samples"]:
        sid = str(sample["sample_id"])
        if sid in ids or not sid.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Duplicate or unsafe sample_id: {sid}")
        ids.add(sid)
        if sample.get("split") not in ("train", "val", "test"):
            raise ValueError(f"Invalid split for {sid}")
        for key in ("input_adata", "source_proc_adata", "labels_csv", "teacher_labeled_h5ad",
                    "nucleus_segmentation", "patch_coordinates", "wsi", "metadata"):
            if key in sample:
                sample[key] = resolve(root, sample[key])
        sources = ("labels_csv", "input_adata", "source_proc_adata", "teacher_labeled_h5ad")
        if not any(sample.get(key) is not None for key in sources):
            if "hest_dir" not in raw:
                raise ValueError(f"{sid} needs hest_dir or an explicit molecular input")
            sample["source_proc_adata"] = raw["hest_dir"] / "proc_adata" / f"{sid}.h5ad"
        if sum(sample.get(key) is not None for key in sources) != 1:
            raise ValueError(f"{sid} requires exactly one of {sources}")
    return raw


def sample_files(config: dict, sample: dict) -> dict[str, Path]:
    """Resolve supported HEST layout or explicit per-sample file overrides."""
    sid = sample["sample_id"]
    root = config.get("hest_dir")
    if root is None:
        return {key: sample[key] for key in ("nucleus_segmentation", "patch_coordinates", "wsi", "metadata")
                if sample.get(key) is not None}
    defaults = {
        "nucleus_segmentation": root / "xenium_seg" / f"{sid}_xenium_nucleus_seg.parquet",
        "patch_coordinates": root / "coordinates" / "patches" / f"{sid}_patches.h5",
        "wsi": root / "wsis" / f"{sid}.tif",
        "metadata": root / "metadata" / f"{sid}.json",
    }
    return {key: sample.get(key) or value for key, value in defaults.items()}


def run_name(config_path: Path, raw: dict) -> str:
    # The input config is copied into each run. Changing any setting creates a
    # distinct run directory rather than silently reusing old patches.
    name = str(raw.get("experiment_id", "experiment"))
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("experiment_id must contain only letters, digits, '_' or '-'")
    # A reused filename may contain a revised label CSV. Include the bytes of
    # every supplied CSV so an edited labeling run cannot reuse an old dataset.
    inputs = []
    for sample in raw["samples"]:
        for key in ("labels_csv", "input_adata", "source_proc_adata", "teacher_labeled_h5ad",
                    "nucleus_segmentation", "patch_coordinates", "wsi", "metadata"):
            source = sample.get(key)
            if source is None:
                continue
            if not source.is_file():
                raise FileNotFoundError(source)
            stat = source.stat()
            inputs.append({"sample_id": sample["sample_id"], "kind": key,
                           "path": str(source), "size": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns,
                           "sha256": digest(source) if key == "labels_csv" else None})
        for key, source in sample_files(raw, sample).items():
            if source.is_file():
                stat = source.stat()
                inputs.append({"sample_id": sample["sample_id"], "kind": key,
                               "path": str(source), "size": stat.st_size,
                               "mtime_ns": stat.st_mtime_ns})
    fingerprint = hashlib.sha256(json.dumps({"config_sha256": digest(config_path),
                                             "inputs": inputs}, sort_keys=True).encode()).hexdigest()
    return f"{name}-{fingerprint[:12]}"


def jsonable(value):
    return json.loads(json.dumps(value, default=str))
