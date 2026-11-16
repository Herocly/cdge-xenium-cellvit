# Cell-label CSV to CDGE/CellViT patch masks

`tools/csv_to_cellvit_patches.py` joins **one HEST sample's** molecular-label
CSV to its nucleus polygons by `cell_id`. It then calls the original CDGE
patch writer to create paired `images/NNNNN.png` and `labels/NNNNN.npy`.
Each `.npy` contains an `inst_map` and a `type_map`.

Required CSV columns: `sample_id,cell_id,predicted_class`. Classes are
`Neoplastic`, `Inflammatory`, `Connective_tissue`, `Epithelial`, and `Unknown`.
Their CDGE IDs are 1, 2, 3, 5, and 11. Extra columns from the molecular export
are permitted. The script refuses duplicate IDs, mixed sample IDs, unexpected
classes, and a low CSV-to-segmentation ID match fraction. A nucleus absent from
the CSV is assigned ID 11 and reported in the audit JSON.

Example, first run only one patch:

```bash
python tools/csv_to_cellvit_patches.py \
  --hest-dir /path/to/hest_data \
  --cdge-root /path/to/original/CDGE \
  --sample-id TENX142 \
  --labels-csv /path/to/TENX142_molecular_labels.csv \
  --out-dir /path/to/output \
  --run-name ours \
  --patch-index 176
```

Omit `--patch-index` to generate all patches. Use the **same HEST sample,
segmentation, patch coordinates, and WSI** for Bottom-up and proposed labels.
Only the label CSV should change. Outputs are automatically isolated under
`<out-dir>/<run-name>-<first-12-characters-of-CSV-SHA256>-<scope>/<sample-id>/`.
If `--run-name` is omitted, the CSV filename stem is used. A changed CSV gets
a different directory and cannot silently overwrite an earlier label set.
An existing output for the same CSV is refused. The audit JSON records the
CSV path/hash, patch-writer hash, ID matching, and produced file counts.
The script does not generate a label CSV or verify H&E-to-nucleus registration;
those are upstream requirements.

## Verified smoke test, 2026-11-16

On server sample `TENX142`, patch 176, we made a seven-cell *demo CSV* from
the original `TENX142_bottomup_cellmarker.h5ad`. All seven CSV IDs matched
the HEST nucleus segmentation. Six polygon representative points were inside
the patch, and all six agreed with the existing type mask. The adapter's newly
generated H&E PNG, `inst_map`, and `type_map` were each identical to the
original Bottom-up output (PNG SHA equality; zero pixel differences for both
maps). The audit and example are in
`reviews/cellvit_csv_bridge_20261116/real_hest_probe/`.

**This verifies the CSV join and CDGE mask writer, not our molecular method on
HEST.** The existing Breast/Pancreas prediction CSVs came from different
samples and must never be joined to `TENX142`. Run both molecular methods on
the chosen HEST training samples before CellViT comparison.

For output isolation, a second synthetic CSV changed exactly one of these
seven cell labels from Connective tissue (ID 3) to Inflammatory (ID 2). The two
files produced different run directories. Their H&E PNGs were identical, their
instance maps had zero changed pixels, and their type maps differed at 173
pixels, all changing from ID 3 to ID 2. This is an interface check only; the
synthetic label is not a biological result. Both runs and manifests are under
`reviews/cellvit_csv_bridge_20261116/isolated_runs/` on the server.

## Required before CellViT training

- The original patch writer uses ID `11` for Unknown. The inspected CellViT
  trainer calls `F.one_hot(type_map, num_classes=6)` without masking ID 11.
  Thus the output format is correct for CDGE, but the present trainer cannot
  directly train on masks containing Unknown. Define a shared ignore-label
  policy for both arms and verify it in the loss and metrics. Do not silently
  turn Unknown nuclei into background.
- The original `_build_shapes` numbers instances starting at zero, which is
  also the background ID. Inspect and, if necessary, fix this shared writer for
  both arms before training. The one-patch equality check preserves that
  original behavior and does not validate its suitability as a training target.
- The persistent environment is specified by `envs/image/pyproject.toml` and
  `envs/image/uv.lock`. Run `uv sync --project envs/image --locked --python 3.10`
  and use `envs/image/.venv/bin/python` for this adapter. This environment was
  used to successfully regenerate one real HEST patch on the new server. The
  old CDGE virtualenv's Python symlink points to the previous account and is
  currently broken.
