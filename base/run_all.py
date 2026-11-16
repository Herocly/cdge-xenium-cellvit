#!/usr/bin/env python3
"""Run the complete two-pass B2.16 method with explicit inputs and checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_config(path):
    config = json.loads(path.read_text())
    required = {"dataset_id", "organ", "input_adata", "output_dir", "analysis_python", "cnv_python"}
    if required - config.keys():
        raise ValueError("Missing configuration fields: " + str(required - config.keys()))
    if not str(config["organ"]).strip():
        raise ValueError("Organ metadata must be supplied")
    config.setdefault("preparation_mode", "recompute")
    config.setdefault("counts_layer", "counts")
    if config["preparation_mode"] not in {"recompute", "existing"}:
        raise ValueError("Unknown preparation mode")
    if config["preparation_mode"] == "existing" and config["counts_layer"] != "counts":
        raise ValueError("Existing preparation requires raw counts in layers['counts']")
    for key in ("input_adata", "output_dir", "analysis_python", "cnv_python", "evaluation_reference"):
        if key in config:
            expanded = os.path.expandvars(str(config[key]))
            if "$" in expanded:
                raise ValueError(f"Unresolved environment variable in {key}: {expanded}")
            candidate = Path(expanded).expanduser()
            config[key] = str((path.parent / candidate).absolute() if not candidate.is_absolute() else candidate.absolute())
    for key in ("input_adata", "analysis_python", "cnv_python"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    if config.get("evaluation_reference") and not Path(config["evaluation_reference"]).is_file():
        raise FileNotFoundError(config["evaluation_reference"])
    return config


def build_plan(config):
    out = Path(config["output_dir"])
    analysis = out / "01_preparation/label_free_analysis.h5ad"
    cluster = out / "02_cluster_evidence/cluster_reliability_no_official_labels.csv"
    core = out / "03_bootstrap_normal_seeds/tiered_initial_seeds_no_official_labels.csv"
    old_cnv = out / "04_bootstrap_cnv/adata_cnv.h5ad"
    old_audit = out / "05_bootstrap_audit/cnv_candidate_audit_label_free.csv"
    old_evidence = out / "06_bootstrap_evidence/neoplastic_evidence_table_no_official_labels.csv"
    old_seeds = out / "07_bootstrap_neoplastic_seeds/neoplastic_seed_candidates.csv"
    multi = out / "08_multicluster_seeds/tiered_initial_seeds_no_official_labels.csv"
    qc = out / "09_seed_qc/qc_screened_tiered_seeds_no_official_labels.csv"
    normal = out / "10_original_seed_source/combined_initial_seeds_no_official_labels.csv"
    expanded = out / "11_normal_expansion/normal_reference_no_official_labels.csv"
    cnv = out / "12_strict_ci_cnv/adata_cnv.h5ad"
    audit = out / "13_cnv_audit/cnv_candidate_audit_label_free.csv"
    evidence = out / "14_neoplastic_evidence/neoplastic_evidence_table_no_official_labels.csv"
    neoplastic = out / "15_new_neoplastic_seeds/neoplastic_seed_candidates.csv"
    combined = out / "16_four_class_seeds/combined_initial_seeds_no_official_labels.csv"
    final = out / "17_four_class_iteration/iterative_all_cell_predictions_no_official_labels.csv"
    full_axis = out / "18_full_input_predictions/all_input_predictions_no_official_labels.csv"
    steps = []

    def stage(name, mode, arguments, outputs, python="cnv_python"):
        command = [config[python], str(ROOT / "run_stage.py"), mode, "--", *map(str, arguments)]
        steps.append({"name": name, "command": command, "outputs": list(map(str, outputs))})

    if config["preparation_mode"] == "recompute":
        args = ["--input", config["input_adata"], "--out", analysis]
        if config["counts_layer"] != "X":
            args += ["--expression-layer", config["counts_layer"]]
        stage("01_preparation", "prepare", args, [analysis], "analysis_python")
    else:
        steps.append({"name": "01_preparation", "copy_input": config["input_adata"], "outputs": [str(analysis)]})
    stage("02_cluster_evidence", "cluster-evidence", ["--adata", analysis, "--tissue", config["organ"], "--out-dir", cluster.parent], [cluster], "analysis_python")
    stage("03_bootstrap_normal_seeds", "normal-seeds", ["--adata", analysis, "--cluster-evidence", cluster, "--seed-policy", "single", "--out-dir", core.parent], [core], "analysis_python")
    stage("04_bootstrap_cnv", "bootstrap-cnv", ["--adata", analysis, "--non-tumor-seeds", core, "--out-dir", old_cnv.parent], [old_cnv])
    stage("05_bootstrap_audit", "bootstrap-audit", ["--cnv-adata", old_cnv, "--out-dir", old_audit.parent], [old_audit])
    stage("06_bootstrap_evidence", "bootstrap-evidence", ["--cnv-adata", old_cnv, "--expression-adata", analysis, "--cluster-table", cluster, "--cnv-audit", old_audit, "--out-dir", old_evidence.parent], [old_evidence])
    stage("07_bootstrap_neoplastic_seeds", "bootstrap-seeds", ["--evidence-table", old_evidence, "--out-dir", old_seeds.parent], [old_seeds])
    stage("08_multicluster_seeds", "normal-seeds", ["--adata", analysis, "--cluster-evidence", cluster, "--out-dir", multi.parent], [multi], "analysis_python")
    stage("09_seed_qc", "seed-qc", ["--adata", analysis, "--tiered-seeds", multi, "--out-dir", qc.parent], [qc], "analysis_python")
    stage("10_original_seed_source", "historical-seed-merge", ["--adata", analysis, "--tiered-seeds", qc, "--neoplastic-seeds", old_seeds, "--neoplastic-evidence", old_evidence, "--out-dir", normal.parent], [normal])
    stage("11_normal_expansion", "normal-expand", ["--adata", analysis, "--seed-file", normal, "--cluster-evidence", cluster, "--out-dir", expanded.parent], [expanded, expanded.parent / "generation_manifest_no_official_labels.json"], "analysis_python")
    stage("12_strict_ci_cnv", "cnv", ["--adata", analysis, "--non-tumor-seeds", expanded, "--out-dir", cnv.parent], [cnv])
    stage("13_cnv_audit", "cnv-audit", ["--cnv-adata", cnv, "--out-dir", audit.parent], [audit])
    stage("14_neoplastic_evidence", "neoplastic-evidence", ["--cnv-adata", cnv, "--expression-adata", analysis, "--cluster-table", cluster, "--cnv-audit", audit, "--out-dir", evidence.parent], [evidence])
    stage("15_new_neoplastic_seeds", "neoplastic-seeds", ["--evidence-table", evidence, "--out-dir", neoplastic.parent], [neoplastic])
    stage("16_four_class_seeds", "combine", ["--adata", analysis, "--normal-seeds", normal, "--neoplastic-seeds", neoplastic, "--out-dir", combined.parent], [combined])
    stage("17_four_class_iteration", "expand", ["--adata", analysis, "--seed-file", combined, "--cluster-evidence", cluster, "--neoplastic-evidence", evidence, "--out-dir", final.parent], [final, final.parent / "final_pseudolabels_no_official_labels.csv"])
    stage("18_full_input_predictions", "export-full-axis", ["--input-adata", config["input_adata"], "--predictions", final, "--out-dir", full_axis.parent], [full_axis])
    if config.get("evaluation_reference"):
        stage("19_evaluation_only", "evaluate", ["--predictions", full_axis, "--reference", config["evaluation_reference"], "--method-name", "B2.16-complete", "--out-dir", out / "19_evaluation_only"], [out / "19_evaluation_only/evaluation_summary_only.json"])
    return steps


def preflight(config):
    subprocess.run([sys.executable, str(ROOT / "tools/verify_bundle.py")], check=True)
    reports = {}
    for kind in ("analysis", "cnv"):
        command = [config[kind + "_python"], str(ROOT / "tools/preflight.py"), "--kind", kind]
        if kind == "analysis":
            command += ["--adata", config["input_adata"], "--counts-layer", config["counts_layer"],
                        "--preparation-mode", config["preparation_mode"], "--organ", config["organ"]]
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
        reports[kind] = json.loads(result.stdout)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.absolute())
    reports = preflight(config)
    plan = build_plan(config)
    if args.check_only:
        print(json.dumps({"preflight": reports, "plan": plan}, indent=2))
        return
    out = Path(config["output_dir"])
    signature = {"config": config, "input_sha256": sha(Path(config["input_adata"])),
                 "bundle_manifest_sha256": sha(ROOT / "BUNDLE_MANIFEST.json")}
    status_path = out / "status.json"
    if args.resume:
        state = json.loads(status_path.read_text())
        if signature != state["signature"]:
            raise ValueError("Cannot resume changed code, config or input")
    else:
        out.mkdir(parents=True, exist_ok=False)
        state = {"status": "running", "started": now(), "signature": signature, "steps": {}}
        atomic_json(out / "plan.json", plan)
        atomic_json(out / "environment_preflight.json", reports)
    state["status"] = "running"
    atomic_json(status_path, state)
    try:
        for step in plan:
            name = step["name"]
            recorded = state["steps"].get(name, {})
            if recorded.get("status") == "completed":
                for path, expected in recorded["outputs"].items():
                    if sha(Path(path)) != expected:
                        raise ValueError("Changed completed output: " + path)
                continue
            if any(Path(path).exists() for path in step["outputs"]):
                raise FileExistsError("Partial stage outputs require a separate fresh run: " + name)
            state["current_step"] = name
            state["updated"] = now()
            state["steps"][name] = {"status": "running", "started": now()}
            atomic_json(status_path, state)
            print(name, flush=True)
            if "copy_input" in step:
                destination = Path(step["outputs"][0])
                destination.parent.mkdir(parents=True)
                shutil.copy2(step["copy_input"], destination)
            else:
                with (out / (name + ".log")).open("a") as log:
                    subprocess.run(step["command"], stdout=log, stderr=subprocess.STDOUT, check=True)
            hashes = {path: sha(Path(path)) for path in step["outputs"]}
            state["steps"][name].update(status="completed", completed=now(), outputs=hashes)
            atomic_json(status_path, state)
        if sha(Path(config["input_adata"])) != signature["input_sha256"] or sha(ROOT / "BUNDLE_MANIFEST.json") != signature["bundle_manifest_sha256"]:
            raise ValueError("Input or bundle changed during the run")
        state["status"] = "completed"
        state["completed"] = now()
        print("Completed full pipeline:", out, flush=True)
    except Exception as error:
        state.update(status="failed", error=repr(error), updated=now())
        raise
    finally:
        atomic_json(status_path, state)


if __name__ == "__main__":
    main()
