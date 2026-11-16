#!/usr/bin/env python3
"""Run B2.16 with an E-only Binary-SPA-style gate before normal expansion.

The frozen base bundle is never edited. This runner inserts stage 10b and
changes only the stage-11 and stage-16 seed inputs to its output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base"
spec = importlib.util.spec_from_file_location("frozen_b216_runner", BASE / "run_all.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def replace_input(command: list[str], flag: str, original: str, replacement: str) -> None:
    position = command.index(flag)
    if command[position + 1] != original:
        raise ValueError(f"Unexpected {flag} source in frozen plan")
    command[position + 1] = replacement


def build_plan(config: dict) -> list[dict]:
    plan = base.build_plan(config)
    output = Path(config["output_dir"])
    source = str(output / "10_original_seed_source/combined_initial_seeds_no_official_labels.csv")
    gated_dir = output / "10b_binary_spa_e_gate"
    gated = str(gated_dir / "combined_initial_seeds_no_official_labels.csv")
    gate = {
        "name": "10b_binary_spa_e_gate",
        "command": [
            config["analysis_python"], str(HERE / "binary_spa_e_gate.py"),
            "--adata", str(output / "01_preparation/label_free_analysis.h5ad"),
            "--seed-file", source,
            "--tissue", config["organ"],
            "--method-dir", str(BASE / "methods/b2_16_ci_cnv_four_class_candidate_20261011"),
            "--out-dir", str(gated_dir),
        ],
        "outputs": [
            gated,
            str(gated_dir / "e_gate_decisions_no_official_labels.csv"),
            str(gated_dir / "marker_panel.json"),
            str(gated_dir / "gate_audit_no_official_labels.json"),
        ],
    }
    after = next(i for i, stage in enumerate(plan) if stage["name"] == "10_original_seed_source")
    plan.insert(after + 1, gate)
    expansion = next(stage for stage in plan if stage["name"] == "11_normal_expansion")
    combination = next(stage for stage in plan if stage["name"] == "16_four_class_seeds")
    replace_input(expansion["command"], "--seed-file", source, gated)
    replace_input(combination["command"], "--normal-seeds", source, gated)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = base.load_config(args.config.absolute())
    reports = base.preflight(config)
    plan = build_plan(config)
    if args.check_only:
        print(json.dumps({"preflight": reports, "plan": plan}, indent=2))
        return

    out = Path(config["output_dir"])
    signature = {
        "config": config,
        "input_sha256": base.sha(Path(config["input_adata"])),
        "base_manifest_sha256": base.sha(BASE / "BUNDLE_MANIFEST.json"),
        "runner_sha256": base.sha(Path(__file__)),
        "gate_sha256": base.sha(HERE / "binary_spa_e_gate.py"),
        "binary_rule_sha256": base.sha(HERE / "run_selection.py"),
    }
    status_path = out / "status.json"
    if args.resume:
        state = json.loads(status_path.read_text())
        if state["signature"] != signature:
            raise ValueError("Cannot resume changed code, config or input")
    else:
        out.mkdir(parents=True, exist_ok=False)
        state = {"status": "running", "started": base.now(),
                 "signature": signature, "steps": {}}
        base.atomic_json(out / "plan.json", plan)
        base.atomic_json(out / "environment_preflight.json", reports)
    state["status"] = "running"
    base.atomic_json(status_path, state)
    try:
        for stage in plan:
            name = stage["name"]
            recorded = state["steps"].get(name, {})
            if recorded.get("status") == "completed":
                for path, expected in recorded["outputs"].items():
                    if base.sha(Path(path)) != expected:
                        raise ValueError("Changed completed output: " + path)
                continue
            if any(Path(path).exists() for path in stage["outputs"]):
                raise FileExistsError("Partial stage outputs require a fresh run: " + name)
            state["current_step"] = name
            state["updated"] = base.now()
            state["steps"][name] = {"status": "running", "started": base.now()}
            base.atomic_json(status_path, state)
            print(name, flush=True)
            if "copy_input" in stage:
                destination = Path(stage["outputs"][0])
                destination.parent.mkdir(parents=True)
                shutil.copy2(stage["copy_input"], destination)
            else:
                with (out / (name + ".log")).open("a") as log:
                    subprocess.run(stage["command"], stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            hashes = {path: base.sha(Path(path)) for path in stage["outputs"]}
            state["steps"][name].update(status="completed", completed=base.now(),
                                        outputs=hashes)
            base.atomic_json(status_path, state)
        if (base.sha(Path(config["input_adata"])) != signature["input_sha256"]
                or base.sha(BASE / "BUNDLE_MANIFEST.json") != signature["base_manifest_sha256"]
                or base.sha(HERE / "binary_spa_e_gate.py") != signature["gate_sha256"]
                or base.sha(HERE / "run_selection.py") != signature["binary_rule_sha256"]):
            raise ValueError("Code or input changed during the run")
        state["status"] = "completed"
        state["completed"] = base.now()
        print("Completed gated pipeline:", out, flush=True)
    except Exception as error:
        state.update(status="failed", error=repr(error), updated=base.now())
        raise
    finally:
        base.atomic_json(status_path, state)


if __name__ == "__main__":
    main()
