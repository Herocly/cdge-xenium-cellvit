# HEST Xenium labels to CellViT: deployment and verification

This is the **integrated engineering interface**. The frozen molecular algorithms
remain in `base/methods/` and `experimental_cell_gate/`. The original CellViT
model, PanNuke loader, loss classes, and HEST polygon writer are copied with
their license to `vendor/CDGE/`; raw data and weights are not included.

## What runs, and what is not yet a result

The code accepts one or more HEST Xenium samples, generates the latest
molecular labels or imports a `labels_csv`, joins them by Xenium `cell_id` to
registered nucleus polygons, writes PNG/NPY pairs, audits the pairs, builds
cohort-separated folds, trains a CellViT-256 model, and can evaluate on an
external PanNuke-format dataset. Each arm needs its own configuration and
output directory. **No full two-arm CellViT experiment has been completed.**
The only real-data training check used two patches from the *same* TENX142
slide and a seven-cell **teacher-label demonstration CSV**. It establishes
loader and gradient compatibility, not image-level performance or our method's
HEST label quality.

The integrated trainer uses the teacher's model, dataset loader, and loss
classes, but has a necessary **Unknown-aware type-loss adapter** and a simpler
training loop. It follows the teacher configuration's 130-epoch, seed-19,
AdamW, exponential-scheduler, and epoch-25 encoder-unfreeze defaults, but is
**not numerically identical** to the original regimen: it does not yet use
the original augmentation, cell/tissue sampling, mixed precision, or exact
validation scheduling. These changes
must be agreed and held fixed in both arms before a paper comparison. The
original trainer crashes on class ID 11 (`Unknown`). This project never maps
Unknown to Background, Dead, or a normal cell class.

## Input contract

For each `sample_id`, the HEST root must contain:

```
proc_adata/<ID>.h5ad                       # per-cell raw counts, if generating labels
xenium_seg/<ID>_xenium_nucleus_seg.parquet # index = Xenium cell_id; H&E pixel polygons
coordinates/patches/<ID>_patches.h5       # H&E patch coordinates
wsis/<ID>.tif                             # registered H&E WSI, MPP metadata
metadata/<ID>.json                        # organ/tissue
```

The HEST `st/<ID>.h5ad` can be *spot-level*. Do not use it for cell labels.
`tools/prepare_hest_cell_input.py` builds label-free, cell-level AnnData from
`proc_adata` raw counts and nucleus polygon centroids. It converts WSI pixel
centroids to microns using the WSI MPP, copies no existing class/cluster
annotations, and records input hashes and match statistics. The identity
preflight rejects an h5ad whose cell IDs do not overlap the segmentation.
This conversion defines the spatial input and must be recorded in a study.

Alternatively, supply `input_adata` with raw `layers['counts']`, unique cell
and gene IDs, and finite two-column `obsm['spatial']`; or supply `labels_csv`
to skip molecular generation. For an existing teacher Bottom-up run, set
`teacher_labeled_h5ad` to its labeled per-cell AnnData; the exporter reads
`obs['cell_type_id']` without rerunning or changing the teacher's method.
This file is an **input artifact** and must come from a documented Bottom-up
run on the same HEST sample. The CSV columns are `sample_id,cell_id,predicted_class`
with one row per cell. Accepted names are `Connective_tissue`, `Inflammatory`,
`Epithelial`, `Neoplastic`, and `Unknown`. An external label CSV must name the
same HEST sample; a high ID match fraction is enforced. Unmatched nuclei
remain Unknown; unmatched CSV cells are counted and checked.

### Nucleus and type maps

Each `foldN/images/<name>.png` is a 256×256 uint8 H&E patch. The teacher
writer saves opaque RGBA; its CellViT loader converts it to RGB. The audit
rejects non-opaque alpha or other modes. Each
`foldN/labels/<name>.npy` is a dictionary with `inst_map` and `type_map`.
`inst_map` has **0 only for background** and positive integer instance IDs;
`type_map` is constant within each instance. Class IDs follow the teacher's
configuration: Background=0, Neoplastic=1, Inflammatory=2, Connective=3,
Dead=4, Epithelial=5; Unknown=11 is an **ignore label for the type loss**.
The original polygon writer began an instance at 0. The adapter adds 1 to
every instance ID without changing polygons or class assignments. The
integrated trainer still uses nuclei geometry for binary/HV segmentation
losses; Unknown pixels are excluded only from the six-class type loss.
These HEST nuclei are segmentation-derived, not manually verified ground truth.

## Install on a new Linux server

1. Clone the reviewed repository, preserving `vendor/CDGE` and `base/`. Place HEST
   data and model weights outside the code directory. Obtain authorized data
   access separately. No hard-coded account path is required.
2. Install `uv` and Python 3.10/3.11, then run `bash scripts/install.sh`.
   This installs the three locked scientific environments and a separate
   CellViT environment. Set `CDGE_TORCH_INDEX_URL` to the appropriate official
   PyTorch wheel index if the server needs a CUDA-specific build. OpenSlide
   needs a native shared library; `envs/image` pins `openslide-bin` for Linux.
3. Use `envs/cellvit/.venv/bin/python` (or another interpreter with PyYAML)
   for the top-level command. **Do not resolve** `venv/bin/python` symlinks to
   their base interpreters; that loses the installed packages.
4. Run `bash scripts/check_install.sh`, then copy `configs/integrated.example.yaml` (proposed labels) or
   `configs/integrated.bottomup.example.yaml` (teacher Bottom-up labels) to a
   private YAML file. Set HEST
   root, sample IDs, organ, **cohort**, train/val role, environment paths,
   an optional `source_proc_adata` or `labels_csv`, and a unique `experiment_id`.
   Standard HEST `proc_adata/<ID>.h5ad` is used when no per-sample molecular
   source is specified. A nonstandard layout may set per-sample
   `nucleus_segmentation`, `patch_coordinates`, `wsi`, and `metadata` paths.
   Paths are relative to the YAML unless absolute. `Ovary` is the molecular organ;
   the CellViT dataset uses the teacher's `Ovarian` tissue name.

```bash
cd /path/to/cdge-xenium-cellvit
envs/cellvit/.venv/bin/python -m integrated_pipeline --config configs/my_experiment.yaml --stage preflight
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage all
```

For stage control, replace `all` with `preflight`, `molecular`, `convert`, `check`,
`assemble`, `smoke`, `train`, or `evaluate`. `molecular` invokes the frozen
latest variant and exports the cell CSV. `convert` creates patches;
`check` verifies every written pair; `assemble` requires at least one train
and one validation cohort and refuses cohort overlap. `smoke` performs a
short training step only after assembly. `train` runs the configured epochs.
`evaluate` requires `cellvit.external_test_dataset` in PanNuke format.
The `all` stage runs strict HEST preflight and requires all dependencies,
including the external test set.
Stages reuse existing hashed outputs; edit the YAML or input CSV to create a
different run. A partial failed molecular run is retained for diagnosis.

## Outputs and provenance

```
<output_root>/<experiment_id>-<config-and-input-fingerprint>/
  source_config.yaml, resolved_config.json
  molecular/<sample_id>/run/{plan.json,status.json,01_...18_...}
  molecular/<sample_id>/labels.csv
  patches/<sample_id>-<csv-hash>-<scope>/<sample_id>/{images,labels,types.csv,cell_count.csv,csv_to_cellvit_audit.json}
  audits/patch_compatibility.json
  cellvit_dataset/{dataset_config.yaml,weight_config.yaml,split_manifest.json,fold0,fold1}
  cellvit_training/{smoke_checkpoint.pth,smoke_report.json,model_best.pth,model_last.pth,training_report.json}
  cellvit_evaluation/conditional_type_metrics.json
  logs/
```

The source config and input fingerprint separate experiment runs. The molecular
runner records source and input hashes. Patch audits record CSV and polygon
writer hashes. Keep patient identifiers and raw data on approved storage.

## Evaluation scope and limitations

The current evaluation adapter computes four-class F1 from model type scores
**at reference nucleus instances** in the external PanNuke dataset, excluding
Dead from the denominator. It uses the reference instance mask to pool scores,
so it is **conditional classification**, not end-to-end CellViT detection,
segmentation, or the teacher's exact reported metric. It must not be labeled
as Dice, AJI, PQ, Detection F1, or an independent full-image benchmark.
Independent PanNuke files are not included or present on the checked server.

The checked server has only TENX142 HEST sample. It cannot establish a
cohort-separated train/validation set; never use two patches of TENX142 as
study train/validation. Complete HEST cohorts, the agreed CellViT training
regimen, PanNuke dead-cell policy, and matched Bottom-up/ours label files are
required for the formal comparison. Do not mix the Breast/Pancreas development
slide metrics with a HEST-trained CellViT result.
The available TENX142 teacher Bottom-up artifact exported 237,762 matching
cells, but **zero Neoplastic labels**. This sample alone cannot demonstrate a
four-class image-training comparison. The proposed method has not produced
TENX142 labels yet.

See `docs/INTEGRATED_VERIFICATION_20261116.md` for exact real-data test status.
