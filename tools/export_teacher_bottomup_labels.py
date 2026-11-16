#!/usr/bin/env python3
"""Export the teacher's already-labeled HEST AnnData to the shared cell CSV.

This is an interface for an existing Bottom-up result; it does not rerun or
change the teacher's labeling algorithm. Class IDs come from CDGE common.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py

CLASS_ID_TO_NAME = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective_tissue",
                    5: "Epithelial", 11: "Unknown"}


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def checksum(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-adata", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with h5py.File(args.labeled_adata) as handle:
        obs = handle["obs"]
        ids = [decode(x) for x in obs[decode(obs.attrs["_index"])][:]]
        class_ids = obs["cell_type_id"][:]
    if len(ids) != len(class_ids) or len(ids) != len(set(ids)):
        raise ValueError("Teacher AnnData has missing or duplicate cell IDs")
    unknown_ids = set(int(x) for x in class_ids) - set(CLASS_ID_TO_NAME)
    if unknown_ids:
        raise ValueError(f"Unsupported teacher type IDs: {sorted(unknown_ids)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.out.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "cell_id", "predicted_class"])
        for cell_id, class_id in zip(ids, class_ids):
            if cell_id == "UNASSIGNED":
                continue
            name = CLASS_ID_TO_NAME[int(class_id)]
            counts[name] += 1
            writer.writerow([args.sample_id, cell_id, name])
    report = {"source": str(args.labeled_adata.resolve()),
              "source_sha256": checksum(args.labeled_adata),
              "sample_id": args.sample_id, "label_source": "teacher_precomputed_bottom_up",
              "rows": sum(counts.values()), "class_counts": dict(counts)}
    args.out.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
