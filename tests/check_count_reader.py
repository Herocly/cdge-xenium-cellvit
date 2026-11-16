"""Raw HEST counts must be preserved for dense and sparse input encodings."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_hest_cell_input import selected_raw_counts


class CountReaderTests(unittest.TestCase):
    def test_dense_and_csr_select_identical_counts(self):
        raw = np.array([[1, 0], [2, 3], [0, 4]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "counts.h5"
            with h5py.File(source, "w") as h5:
                dense = h5.create_dataset("dense", data=raw)
                group = h5.create_group("sparse")
                group.attrs["encoding-type"] = "csr_matrix"
                group.attrs["shape"] = raw.shape
                matrix = sparse.csr_matrix(raw)
                for name in ("data", "indices", "indptr"):
                    group.create_dataset(name, data=getattr(matrix, name))
                a = selected_raw_counts(dense, np.array([0, 2]), raw.shape)
                b = selected_raw_counts(group, np.array([0, 2]), raw.shape)
            np.testing.assert_array_equal(a, raw[[0, 2]].astype(np.int32))
            np.testing.assert_array_equal(b.toarray(), a)


if __name__ == "__main__":
    unittest.main()
