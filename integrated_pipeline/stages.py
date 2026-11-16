"""Stages keep research implementations in their original files."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import Counter

import yaml

from .audit import inspect_sample, write_report
from .config import digest, sample_files

HANDOFF = Path(__file__).resolve().parents[1]


def call(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def molecular(config: dict, sample: dict, run_dir: Path) -> Path:
    if sample.get("labels_csv") is not None:
        source = sample["labels_csv"]
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    sid = sample["sample_id"]
    sample_root = run_dir / "molecular" / sid
    predictions = sample_root / "run/18_full_input_predictions/all_input_predictions_no_official_labels.csv"
    export = sample_root / "labels.csv"
    if sample.get("teacher_labeled_h5ad") is not None:
        if not export.exists():
            call([str(config["image_python"]), str(HANDOFF / "tools/export_teacher_bottomup_labels.py"),
                  "--labeled-adata", str(sample["teacher_labeled_h5ad"]),
                  "--sample-id", sid, "--out", str(export)], sample_root / "teacher_export.log")
        return export
    if export.exists():
        return export
    sample_root.mkdir(parents=True, exist_ok=True)
    files = sample_files(config, sample)
    segmentation = files["nucleus_segmentation"]
    adata_path = sample.get("input_adata")
    if adata_path is None:
        adata_path = sample_root / "hest_label_free_input.h5ad"
        if not adata_path.exists():
            call([str(config["image_python"]), str(HANDOFF / "tools/prepare_hest_cell_input.py"),
                  "--source", str(sample["source_proc_adata"]), "--segmentation", str(segmentation),
                  "--wsi", str(files["wsi"]),
                  "--out", str(adata_path), "--min-match-fraction",
                  str(config.get("min_csv_match_fraction", 0.9))],
                 sample_root / "prepare_hest_cell_input.log")
    call([str(config["image_python"]), str(HANDOFF / "tools/check_hest_identity.py"),
          "--adata", str(adata_path), "--segmentation", str(segmentation),
          "--min-overlap", str(config.get("min_csv_match_fraction", 0.9))],
         sample_root / "identity_preflight.log")
    inner_config = {
        "dataset_id": sid, "organ": sample["organ"],
        "input_adata": str(adata_path),
        "output_dir": str(sample_root / "run"),
        "analysis_python": str(config["analysis_python"]),
        "cnv_python": str(config["cnv_python"]),
        "preparation_mode": sample.get("preparation_mode", "recompute"),
        "counts_layer": sample.get("counts_layer", "counts"),
    }
    internal = sample_root / "molecular_config.json"
    internal.write_text(json.dumps(inner_config, indent=2) + "\n")
    status = sample_root / "run/status.json"
    command = [str(config["analysis_python"]), str(HANDOFF / "run.py"), "run",
               "--variant", "latest", "--config", str(internal)]
    if status.exists():
        command.append("--resume")
    call(command, sample_root / "molecular.log")
    if not predictions.is_file():
        raise FileNotFoundError(predictions)
    call([str(config["analysis_python"]), str(HANDOFF / "tools/export_molecular_labels.py"),
          "--predictions", str(predictions), "--adata", str(adata_path),
          "--sample-id", sid, "--out", str(export)], sample_root / "export.log")
    return export


def convert(config: dict, sample: dict, labels_csv: Path, run_dir: Path) -> Path:
    sid = sample["sample_id"]
    scope = "all" if sample.get("patch_index") is None else f"patch{int(sample['patch_index']):05d}"
    source = f"{sid}-{digest(labels_csv)[:12]}-{scope}"
    result = run_dir / "patches" / source / sid
    if result.exists():
        audit = result / "csv_to_cellvit_audit.json"
        if not audit.exists() or json.loads(audit.read_text()).get("labels_csv_sha256") != digest(labels_csv):
            raise ValueError(f"Existing patch data cannot be reused: {result}")
        return result
    cmd = [str(config["image_python"]), str(HANDOFF / "tools/csv_to_cellvit_patches.py"),
           "--hest-dir", str(config["hest_dir"]), "--cdge-root", str(config["cdge_root"]),
           "--sample-id", sid, "--labels-csv", str(labels_csv),
           "--out-dir", str(run_dir / "patches"), "--run-name", sid,
           "--patch-size", str(config.get("patch_size", 256)),
           "--min-csv-match-fraction", str(config.get("min_csv_match_fraction", 0.9))]
    for key, flag in (("nucleus_segmentation", "--segmentation"),
                      ("patch_coordinates", "--coordinates"),
                      ("wsi", "--wsi"), ("metadata", "--metadata")):
        cmd.extend([flag, str(sample_files(config, sample)[key])])
    if sample.get("patch_index") is not None:
        cmd += ["--patch-index", str(sample["patch_index"])]
    call(cmd, run_dir / "logs" / f"convert_{sid}.log")
    if not result.exists():
        raise RuntimeError(f"Patch writer did not produce expected output: {result}")
    return result


def validate(config: dict, sample_dirs: dict[str, Path], run_dir: Path) -> dict:
    report = {sid: inspect_sample(path, config.get("patch_size", 256))
              for sid, path in sample_dirs.items()}
    write_report(run_dir / "audits/patch_compatibility.json", report)
    return report


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def assemble(config: dict, sample_dirs: dict[str, Path], run_dir: Path) -> Path:
    """Make fold0/train and fold1/val with globally unique image names."""
    cohorts = {split: {s.get("cohort") for s in config["samples"] if s["split"] == split}
               for split in ("train", "val", "test")}
    if None in (cohorts["train"] | cohorts["val"]):
        raise ValueError("Every train/val sample needs a cohort identifier")
    if cohorts["train"] & cohorts["val"]:
        raise ValueError("Train and validation cohorts overlap")
    if not cohorts["train"] or not cohorts["val"]:
        raise ValueError("Both Train and Validation need at least one cohort")
    dataset = run_dir / "cellvit_dataset"
    if dataset.exists():
        raise FileExistsError(dataset)
    teacher_dataset = config["cdge_root"] / "CellViT/configs/datasets/PanNuke"
    source = teacher_dataset / "dataset_config.yaml"
    if not source.is_file():
        raise FileNotFoundError(source)
    # A failed assembly must not leave a directory that later runs mistake for
    # a complete training dataset. Build beside the destination and publish it
    # only after both folds and all metadata have passed validation.
    staging = Path(tempfile.mkdtemp(prefix="cellvit_dataset_build_", dir=run_dir))
    try:
        _assemble_into(config, sample_dirs, staging, source, cohorts)
        for fold in (0, 1):
            inspect_sample(staging / f"fold{fold}", config.get("patch_size", 256))
        staging.rename(dataset)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dataset


def _assemble_into(config: dict, sample_dirs: dict[str, Path], dataset: Path,
                   source: Path, cohorts: dict) -> None:
    shutil.copy2(source, dataset / "dataset_config.yaml")
    tissue_counts = Counter()
    class_counts = Counter()
    for split, fold in (("train", 0), ("val", 1)):
        fold_dir = dataset / f"fold{fold}"
        (fold_dir / "images").mkdir(parents=True)
        (fold_dir / "labels").mkdir()
        types, cell_counts = [], []
        for sample in config["samples"]:
            if sample["split"] != split:
                continue
            sid = sample["sample_id"]
            directory = sample_dirs[sid]
            with (directory / "types.csv").open(newline="") as stream:
                sample_types = {r["img"]: r["type"] for r in csv.DictReader(stream)}
            with (directory / "cell_count.csv").open(newline="") as stream:
                sample_counts = {r["Image"]: r for r in csv.DictReader(stream)}
            for source in sorted((directory / "images").glob("*.png")):
                name = f"{sid}_{source.name}"
                label = directory / "labels" / (source.stem + ".npy")
                _link_or_copy(source, fold_dir / "images" / name)
                _link_or_copy(label, fold_dir / "labels" / (sid + "_" + label.name))
                types.append({"img": name, "type": sample_types[source.name]})
                count = dict(sample_counts[source.name]); count["Image"] = name
                cell_counts.append(count)
                if split == "train":
                    tissue_counts[sample_types[source.name]] += 1
                    for class_name in ("Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"):
                        class_counts[class_name] += int(count[class_name])
        if not types:
            raise ValueError(f"No image/label pairs for {split}")
        for filename, rows, columns in (("types.csv", types, ["img", "type"]),
                                         ("cell_count.csv", cell_counts, list(cell_counts[0]))):
            with (fold_dir / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)
    (dataset / "weight_config.yaml").write_text(yaml.safe_dump({
        "cell": dict(class_counts), "tissue": dict(tissue_counts)
    }, sort_keys=True))
    write_report(dataset / "split_manifest.json", {
        "cohorts": {k: sorted(v) for k, v in cohorts.items()},
        "samples": [{"sample_id": s["sample_id"], "split": s["split"], "cohort": s.get("cohort")}
                    for s in config["samples"]],
    })
