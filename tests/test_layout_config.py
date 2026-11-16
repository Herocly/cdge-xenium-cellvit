"""New-server configuration tests with no patient data or HEST download."""
from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated_pipeline.config import load, run_name, sample_files


class LayoutConfigTests(unittest.TestCase):
    def test_standard_hest_layout_needs_only_root_and_sample_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hest = root / "hest"
            (hest / "proc_adata").mkdir(parents=True)
            (hest / "proc_adata/TENX_A.h5ad").write_bytes(b"placeholder")
            config_file = root / "run.yaml"
            config_file.write_text(yaml.safe_dump({
                "hest_dir": str(hest), "samples": [{"sample_id": "TENX_A", "split": "train"}]}))
            config = load(config_file)
            sample = config["samples"][0]
            self.assertEqual(sample["source_proc_adata"], hest / "proc_adata/TENX_A.h5ad")
            self.assertEqual(sample_files(config, sample)["wsi"], hest / "wsis/TENX_A.tif")
            self.assertTrue(run_name(config_file, config).startswith("experiment-"))

    def test_explicit_wsi_change_is_a_different_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "labels.csv").write_text("sample_id,cell_id,predicted_class\nA,1,Epithelial\n")
            wsi = root / "a.tif"
            wsi.write_bytes(b"first")
            config_file = root / "run.yaml"
            config_file.write_text(yaml.safe_dump({
                "hest_dir": str(root), "samples": [{"sample_id": "A", "split": "train",
                "labels_csv": "labels.csv", "wsi": "a.tif"}]}))
            first = run_name(config_file, load(config_file))
            wsi.write_bytes(b"a longer replacement")
            self.assertNotEqual(first, run_name(config_file, load(config_file)))


if __name__ == "__main__":
    unittest.main()
