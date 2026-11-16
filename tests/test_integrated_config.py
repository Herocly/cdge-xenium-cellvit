"""Small portability checks; real HEST image/gradient tests are documented separately."""
from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated_pipeline.config import load, run_name


class IntegratedConfigTests(unittest.TestCase):
    def test_venv_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "bin/base_python").write_text("placeholder")
            (root / "bin/python").symlink_to("base_python")
            (root / "labels.csv").write_text("sample_id,cell_id,predicted_class\nA,1,Epithelial\n")
            source = root / "experiment.yaml"
            source.write_text(yaml.safe_dump({"experiment_id": "test", "image_python": "bin/python",
                                              "samples": [{"sample_id": "A", "split": "train",
                                                           "labels_csv": "labels.csv"}]}))
            config = load(source)
            self.assertEqual(config["image_python"].name, "python")
            self.assertNotEqual(config["image_python"], (root / "bin/base_python").resolve())

    def test_edited_csv_or_sample_creates_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "labels.csv"
            csv.write_text("sample_id,cell_id,predicted_class\nA,1,Epithelial\n")
            source = root / "experiment.yaml"
            payload = {"experiment_id": "test", "samples": [{"sample_id": "A", "split": "train",
                       "labels_csv": "labels.csv"}]}
            source.write_text(yaml.safe_dump(payload))
            first = run_name(source, load(source))
            csv.write_text("sample_id,cell_id,predicted_class\nA,1,Neoplastic\n")
            second = run_name(source, load(source))
            self.assertNotEqual(first, second)
            payload["samples"][0]["sample_id"] = "B"
            source.write_text(yaml.safe_dump(payload))
            self.assertNotEqual(second, run_name(source, load(source)))


if __name__ == "__main__":
    unittest.main()
