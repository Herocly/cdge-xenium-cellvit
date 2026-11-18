# CDGE Xenium to CellViT

This project turns HEST Xenium measurements into cell-type training labels,
matches those labels to nuclei in registered H&E images, and prepares patches
for CellViT training and evaluation.



## Start here

To generate molecular labels and image patches, prepare a Linux server, `uv`,
Python 3.10 and 3.11, and your HEST Xenium files. Store HEST files outside
the repository. The installation script also sets up CellViT; GPU and model
settings matter when we reach training. If your server requires a particular
official PyTorch CUDA wheel, set `CDGE_TORCH_INDEX_URL` before installation.

```bash
git clone https://github.com/Herocly/cdge-xenium-cellvit.git cdge-xenium-cellvit
cd cdge-xenium-cellvit
bash scripts/install.sh
bash scripts/check_install.sh
cp configs/integrated.example.yaml configs/my_experiment.yaml
```

Open `configs/my_experiment.yaml` and set:

- `hest_dir`: where your HEST data lives.
- `output_root`: where results should be saved.
- `samples`: list every HEST sample you want to process, with its `sample_id`,
  real `organ`, and source `cohort`. Add one YAML entry per sample. The config
  also requires `split`; use `train` while preparing labels and patches, then
  set the agreed Train/Validation split when we configure CellViT.

Paths in the example are relative to the YAML file; absolute paths also work.
The `organ` setting selects marker database records and Cell Ontology context
for molecular annotation. CellViT reads its tissue name from HEST metadata.

## What we are running now

The current workflow has three steps: **preflight → molecular → convert**.
Each step works through the samples listed in your YAML. Use that same YAML
for all three commands:

```bash
envs/cellvit/.venv/bin/python run_pipeline.py \
  --config configs/my_experiment.yaml --stage preflight

envs/cellvit/.venv/bin/python run_pipeline.py \
  --config configs/my_experiment.yaml --stage molecular

envs/cellvit/.venv/bin/python run_pipeline.py \
  --config configs/my_experiment.yaml --stage convert
```

1. `preflight` checks the HEST files, cell IDs, and image information. Read
   the report before starting the long run.
2. `molecular` generates cell labels. Find them in
   `molecular/<ID>/labels.csv`.
3. `convert` matches those labels to nuclei and H&E patches. Find the PNG/NPY
   pairs and the matching report in `patches/`.

After these three steps, review the labels and patches for each sample. We
will configure CellViT together later: the Train/Validation cohort split,
model checkpoint, training settings, and test data.

## Stage reference

The first three stages are the current workflow. The remaining stages prepare
and run the later CellViT work:

| Stage | What it does | Main output |
| --- | --- | --- |
| `preflight` | Checks the environment, HEST files, IDs, image metadata, GPU, and cohorts. | Terminal report |
| `molecular` | Generates or imports cell labels for each sample. Computed labels pass through preparation, cluster evidence, seeds, CNV, and iterative fitting. Samples run in YAML order. | `molecular/<ID>/labels.csv` and stage logs |
| `convert` | Matches label IDs to nucleus polygons and cuts registered H&E patches. | PNG images, NPY labels, matching audit |
| `check` | Checks every PNG/NPY pair, instance IDs, class maps, and matching statistics. | `audits/patch_compatibility.json` |
| `assemble` | Collects Train patches in `fold0` and Validation patches in `fold1`. | `cellvit_dataset/` and split manifest |
| `smoke` | Runs a short CellViT training step, reloads a checkpoint, and tests inference. | `cellvit_training/smoke_report.json` |
| `train` | Trains on `fold0` and monitors `fold1`. | Checkpoints and training report |
| `evaluate` | Tests a checkpoint on a prepared PanNuke-format set. | `cellvit_evaluation/conditional_type_metrics.json` |
| `all` | Runs the complete sequence and strict checks. | All outputs above |

Results go to `<output_root>/<experiment_id>-<fingerprint>/`. Each run keeps
its YAML, logs, and input fingerprints. Changing the YAML or an input file
creates a separate run. Completed labels and patches are reused within the
same run. `convert`, `check`, and `assemble` call earlier data stages as
needed; `smoke`, `train`, and `evaluate` run when requested.

### CellViT training later

When we set up CellViT, keep each source cohort entirely in Train or
Validation. Add the agreed split and model settings before running `assemble`,
`smoke`, `train`, or `evaluate`. The `all` stage also uses a prepared
PanNuke-format test set.

Changing the YAML, including its `split` values, creates a new run directory.


## HEST inputs and label files

For sample `<ID>`, the standard layout is:

```text
<hest_dir>/proc_adata/<ID>.h5ad                       per-cell counts
<hest_dir>/xenium_seg/<ID>_xenium_nucleus_seg.parquet registered nucleus polygons; index = cell_id
<hest_dir>/coordinates/patches/<ID>_patches.h5       patch coordinates
<hest_dir>/wsis/<ID>.tif                             registered H&E image and MPP
<hest_dir>/metadata/<ID>.json                        tissue information
```

The molecular stage builds a per-cell input from `proc_adata` and registered
nucleus centroids. Some HEST `st/<ID>.h5ad` files describe spots; `preflight`
checks the file and its ID overlap. For another layout, specify
`nucleus_segmentation`, `patch_coordinates`, `wsi`, and `metadata` for
that sample in the YAML.

You may instead supply one existing molecular source per sample:

- `input_adata`: prepared label-free per-cell AnnData.
- `labels_csv`: previously generated labels.
- `teacher_labeled_h5ad`: a completed Bottom-up labeled AnnData to export.

A label CSV has `sample_id,cell_id,predicted_class`. Classes are
`Connective_tissue`, `Inflammatory`, `Epithelial`, `Neoplastic`, and
`Unknown`. Check per-class counts and the cell-to-nucleus match fraction
before training.

Converted patches have `images/<name>.png` and `labels/<name>.npy`.
The NPY contains `inst_map` for nucleus instances and `type_map` for cell
classes. `Unknown` uses class ID 11. The CellViT adapter excludes Unknown
pixels from the **cell-type loss** while retaining nucleus geometry for binary
and HV losses.

## Compare two label methods

Start from [our example](configs/integrated.example.yaml) and the
[Bottom-up example](configs/integrated.bottomup.example.yaml). Give both arms
the same HEST samples, cohorts, images, nuclear masks, CellViT model, initial
weights, training settings, and test set. Change the label source and
`experiment_id`, and record labeled-nucleus counts for each arm.

The Bottom-up example reads a completed teacher `cell_type_id` h5ad from the
same HEST samples. Agree on a compatible **full-model** CellViT checkpoint and
one training protocol for both arms. The current adapter uses the teacher
model and losses with a simpler loop; its augmentation, sampling, mixed
precision, and validation schedule differ from the teacher's original run.

The external evaluator currently reports four-class type F1 **at reference
nucleus instances**. It excludes PanNuke Dead cells and uses reference masks
to pool predictions. This is conditional cell-type classification.
Segmentation and detection need separate evaluations. HEST nucleus polygons
come from segmentation and have not been independently verified as manual
ground truth.



## Repository map

| Path | Purpose |
| --- | --- |
| `integrated_pipeline/`, `run_pipeline.py` | CLI, stage orchestration, audits, training adapter |
| `base/methods/current/`, `base/run_stage.py`, `experimental_cell_gate/`, `run.py` | Active molecular method and resumable runner |
| `tools/` | HEST checks, label export, patch conversion |
| `envs/`, `scripts/install.sh` | Environment setup |
| `tests/` | Unit checks and real-data smoke helpers |
| `docs/` | Detailed data contracts, outputs, verification records |
| `vendor/CDGE/` | CellViT and HEST processing code copied from the teacher project |


