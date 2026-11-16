#!/usr/bin/env python3
"""Read-only preflight for the integrated pipeline on a new Linux server."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated_pipeline.config import load, sample_files


def probe(python: Path, names: list[str]) -> dict:
    code = (
        "import importlib,json,sys; "
        "names=" + repr(names) + "; "
        "r={}; "
        "\nfor n in names:\n"
        " try:\n  m=importlib.import_module(n); r[n]=getattr(m,'__version__','available')\n"
        " except Exception as e:\n  r[n]=type(e).__name__+': '+str(e)\n"
        "print(json.dumps({'python':sys.version.split()[0],'modules':r}))"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    if result.returncode:
        return {"error": result.stderr.strip() or result.stdout.strip()}
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deep", action="store_true", help="Inspect actual HEST files and IDs in image environment")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on any missing requirement")
    parser.add_argument("--require-train-val", action="store_true",
                        help="Require independent Train and Validation cohorts for the full pipeline")
    args = parser.parse_args()
    config = load(args.config)
    report = {"paths": {}, "environments": {}, "samples": [], "problems": [], "warnings": []}
    for key in ("hest_dir", "cdge_root", "output_root"):
        path = config[key]
        report["paths"][key] = {"path": str(path), "exists": path.exists()}
        if key != "output_root" and not path.exists():
            report["problems"].append(f"Missing {key}: {path}")
    for required in (config["cdge_root"] / "preprocessing_hest/preproce_utils/common.py",
                     config["cdge_root"] / "CellViT/models/segmentation/cell_segmentation/cellvit_shared.py"):
        if not required.is_file():
            report["problems"].append(f"Missing source file: {required}")
    for key, names in (
        ("analysis_python", ["anndata", "scanpy", "sklearn"]),
        ("cnv_python", ["insitucnv", "anndata"]),
        ("image_python", ["anndata", "geopandas", "openslide", "rasterio", "h5py"]),
        ("cellvit_python", ["torch", "cv2", "numba", "natsort", "einops"]),
    ):
        python = config.get(key)
        report["environments"][key] = probe(python, names) if python and python.is_file() else {"error": "Interpreter missing"}
        if "error" in report["environments"][key]:
            report["problems"].append(f"{key}: {report['environments'][key]['error']}")
        for name, value in report["environments"][key].get("modules", {}).items():
            if "Error:" in str(value):
                report["problems"].append(f"{key}: {name}: {value}")
    for sample in config["samples"]:
        sid = sample["sample_id"]
        hest = config["hest_dir"]
        files = sample_files(config, sample)
        sample_report = {
            "sample_id": sid, "split": sample["split"], "cohort": sample.get("cohort"),
            "input_adata": bool(sample.get("input_adata") and sample["input_adata"].is_file()),
            "source_proc_adata": bool(sample.get("source_proc_adata") and sample["source_proc_adata"].is_file()),
            "labels_csv": bool(sample.get("labels_csv") and sample["labels_csv"].is_file()),
            "files": {key: {"path": str(path), "exists": path.is_file()}
                      for key, path in files.items()},
        }
        report["samples"].append(sample_report)
        source = (sample.get("source_proc_adata") or sample.get("input_adata") or
                  sample.get("labels_csv") or sample.get("teacher_labeled_h5ad"))
        if source and not source.is_file():
            report["problems"].append(f"{sid}: missing molecular input: {source}")
        for key, value in sample_report["files"].items():
            if not value["exists"]:
                report["problems"].append(f"{sid}: missing {key}: {value['path']}")
        if args.deep and all(value["exists"] for value in sample_report["files"].values()):
            command = [str(config["image_python"]), str(Path(__file__).resolve().parents[1] / "tools/inspect_hest_input.py")]
            for key, flag in (("nucleus_segmentation", "segmentation"),
                              ("patch_coordinates", "coordinates"), ("wsi", "wsi"), ("metadata", "metadata")):
                command += ["--" + flag, str(files[key])]
            counts_source = sample.get("source_proc_adata") or sample.get("input_adata")
            if counts_source:
                command += ["--source", str(counts_source)]
            command += ["--minimum-match", str(config.get("min_csv_match_fraction", 0.9))]
            checked = subprocess.run(command, capture_output=True, text=True)
            if checked.returncode:
                report["problems"].append(f"{sid}: HEST format check: {checked.stderr.strip()}")
            else:
                sample_report["deep_check"] = json.loads(checked.stdout)
    train = {s.get("cohort") for s in config["samples"] if s["split"] == "train"}
    val = {s.get("cohort") for s in config["samples"] if s["split"] == "val"}
    if not train or not val or None in train | val or train & val:
        message = "Train and Validation need distinct, nonempty cohort identifiers before assembly or training"
        report["problems" if args.require_train_val else "warnings"].append(message)
    checkpoint = config.get("cellvit", {}).get("pretrained_checkpoint")
    if checkpoint and not checkpoint.is_file():
        report["problems"].append(f"Missing pretrained checkpoint: {checkpoint}")
    external = config.get("cellvit", {}).get("external_test_dataset")
    if external and not external.is_dir():
        report["problems"].append(f"Missing external test dataset: {external}")
    cellvit_python = config.get("cellvit_python")
    if cellvit_python and cellvit_python.is_file():
        gpu = subprocess.run([str(cellvit_python), "-c",
                              "import torch,json; print(json.dumps({'available':torch.cuda.is_available(), 'count':torch.cuda.device_count(), 'torch':torch.__version__}))"],
                             capture_output=True, text=True)
        report["gpu"] = json.loads(gpu.stdout.strip().splitlines()[-1]) if gpu.returncode == 0 else {"error": gpu.stderr.strip()}
    else:
        report["gpu"] = {"error": "CellViT interpreter missing"}
    if config.get("cellvit", {}).get("device", "cuda").startswith("cuda") and not report["gpu"].get("available"):
        report["problems"].append("CUDA requested, but the CellViT environment cannot see a GPU")
    report["ok"] = not report["problems"]
    print(json.dumps(report, indent=2))
    if args.strict and report["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
