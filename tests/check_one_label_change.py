"""Compare outputs made from CSVs differing in one cell class."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def one(directory: Path):
    images = list((directory / "images").glob("*.png"))
    if len(images) != 1:
        raise ValueError("Provide directories with one identically named patch")
    image = images[0]
    mask = np.load(directory / "labels" / (image.stem + ".npy"), allow_pickle=True).item()
    return image, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    first_image, a = one(args.before)
    second_image, b = one(args.after)
    if first_image.name != second_image.name:
        raise AssertionError("Different patch names")
    sha = lambda p: hashlib.sha256(p.read_bytes()).digest()
    if sha(first_image) != sha(second_image):
        raise AssertionError("H&E image changed")
    if not np.array_equal(a["inst_map"], b["inst_map"]):
        raise AssertionError("Nucleus instance geometry changed")
    changed = a["type_map"] != b["type_map"]
    if not changed.any():
        raise AssertionError("No type-map change")
    nucleus_ids = np.unique(a["inst_map"][changed])
    if len(nucleus_ids) != 1 or int(nucleus_ids[0]) == 0:
        raise AssertionError(f"Change reached more than one nucleus: {nucleus_ids}")
    nucleus = a["inst_map"] == nucleus_ids[0]
    if not np.array_equal(changed, nucleus):
        raise AssertionError("Changed class did not cover exactly one whole nucleus")
    print(f"PASSED: only instance {int(nucleus_ids[0])} changed ({int(changed.sum())} pixels)")


if __name__ == "__main__":
    main()
