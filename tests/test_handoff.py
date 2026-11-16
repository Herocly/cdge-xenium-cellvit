"""Fast tests for the handoff wrapper; no Xenium data or scientific packages."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load("handoff_base_runner", ROOT / "base/run_all.py")
variant = load("handoff_variant_runner", ROOT / "experimental_cell_gate/run_all_binary_spa.py")
comparison = load("handoff_compare", ROOT / "tools/compare_runs.py")
entry = load("handoff_entry", ROOT / "run.py")


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.config = {"dataset_id": "test", "organ": "Breast",
                       "input_adata": "/input/slide.h5ad", "output_dir": "/output/run",
                       "analysis_python": "/env/analysis/bin/python",
                       "cnv_python": "/env/cnv/bin/python",
                       "preparation_mode": "recompute", "counts_layer": "counts",
                       "evaluation_reference": "/input/reference.csv"}

    def test_frozen_stage_plan_matches_original_snapshot(self):
        actual = base.build_plan(self.config)
        for step in actual:
            if "command" in step:
                step["command"] = [part.replace(str(ROOT / "base"), "<BUNDLE>")
                                   for part in step["command"]]
        expected = json.loads((ROOT / "tests/fixtures/frozen_plan.json").read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 19)

    def test_marker_gate_changes_only_two_downstream_inputs(self):
        frozen = base.build_plan(self.config)
        gated = variant.build_plan(self.config)
        self.assertEqual(len(gated), 20)
        self.assertEqual(gated[10]["name"], "10b_binary_spa_e_gate")
        gate_command = gated[10]["command"]
        method_dir = Path(gate_command[gate_command.index("--method-dir") + 1])
        self.assertTrue((method_dir / "code/02_build_cluster_evidence.py").is_file())
        self.assertTrue((method_dir / "vendor/cdge/preprocessing_hest/preproce_utils/bottom_up_utils.py").is_file())
        self.assertTrue((method_dir / "resources/cell_ontology/cl.obo").is_file())
        for step in frozen:
            candidate = next(item for item in gated if item["name"] == step["name"])
            if step["name"] not in {"11_normal_expansion", "16_four_class_seeds"}:
                self.assertEqual(candidate, step)

    def test_default_run_enters_latest_marker_gated_pipeline(self):
        with patch.object(sys, "argv", ["run.py", "run", "--config", "config.json"]), \
             patch.object(entry.subprocess, "run") as launch:
            entry.main()
        command = launch.call_args.args[0]
        self.assertEqual(Path(command[1]), ROOT / "experimental_cell_gate/run_all_binary_spa.py")

    def test_existing_preparation_plan_matches_original_snapshot(self):
        config = dict(self.config, organ="Pancreas",
                      input_adata="/input/prepared.h5ad", preparation_mode="existing")
        config.pop("evaluation_reference")
        actual = base.build_plan(config)
        for step in actual:
            if "command" in step:
                step["command"] = [part.replace(str(ROOT / "base"), "<BUNDLE>")
                                   for part in step["command"]]
        expected = json.loads((ROOT / "tests/fixtures/frozen_existing_plan.json").read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 18)

    def test_cell_comparison_uses_cell_id_not_row_order(self):
        with TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old.csv", Path(tmp) / "new.csv"
            for path, values in ((old, [("b", "E"), ("a", "C")]),
                                 (new, [("a", "C"), ("b", "E")])):
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("cell_id", "label"))
                    writer.writerows(values)
            self.assertTrue(comparison.compare(old, new, ("label",))["exact_match"])

    def test_config_expands_paths_without_changing_method_settings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("input.h5ad", "analysis-python", "cnv-python"):
                (root / name).touch()
            config = dict(self.config, input_adata="${CDGE_TEST_ROOT}/input.h5ad",
                          output_dir="${CDGE_TEST_ROOT}/output",
                          analysis_python="${CDGE_TEST_ROOT}/analysis-python",
                          cnv_python="${CDGE_TEST_ROOT}/cnv-python")
            config.pop("evaluation_reference")
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with patch.dict(os.environ, {"CDGE_TEST_ROOT": str(root)}):
                loaded = base.load_config(path)
            self.assertEqual(loaded["input_adata"], str(root / "input.h5ad"))
            self.assertEqual(loaded["output_dir"], str(root / "output"))
            self.assertEqual(loaded["preparation_mode"], "recompute")

    def test_existing_preparation_requires_counts_layer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("input.h5ad", "analysis-python", "cnv-python"):
                (root / name).touch()
            config = dict(self.config, input_adata=str(root / "input.h5ad"),
                          output_dir=str(root / "output"),
                          analysis_python=str(root / "analysis-python"),
                          cnv_python=str(root / "cnv-python"),
                          preparation_mode="existing", counts_layer="X")
            config.pop("evaluation_reference")
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "layers"):
                base.load_config(path)


if __name__ == "__main__":
    unittest.main()
