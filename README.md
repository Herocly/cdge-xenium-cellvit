# CDGE Xenium to CellViT

This project turns HEST Xenium measurements into cell-type training labels,
matches those labels to nuclei in registered H&E images, and prepares patches
for CellViT training and evaluation.

The molecular method checks cluster evidence, selects C/I/E seeds, applies a
Binary-SPA-style marker check to Epithelial seeds, expands normal-cell labels,
and uses CNV and expression evidence to generate Neoplastic seeds. A four-class
classifier then expands labels for up to 20 rounds. Cells without enough
support remain `Unknown`.

## Start here

You need a Linux server, `uv`, Python 3.10 and 3.11, HEST Xenium files, and a
CUDA-capable GPU for CellViT training. Store HEST files and model weights
outside the repository. If your server requires a particular official PyTorch
CUDA wheel, set `CDGE_TORCH_INDEX_URL` before installation.

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
- `samples`: each sample's ID, real organ, source cohort, and `train` or
  `val` role.
- `cellvit.pretrained_checkpoint`: a compatible full CellViT checkpoint for
  a formal training comparison.

Paths in the example are relative to the YAML file; absolute paths also work.
The `organ` setting selects marker database records and Cell Ontology context
for molecular annotation. CellViT reads its tissue name from HEST metadata.

Put all samples in one YAML. Keep each source cohort entirely in Train or
Validation. Use cohort names from the source metadata or CSV.

Check the inputs before a long run:

```bash
envs/cellvit/.venv/bin/python run_pipeline.py \
  --config configs/my_experiment.yaml --stage preflight
```

The report shows paths, cell ID overlap, image metadata, environments, GPU
status, and split issues. You can begin with one sample for label generation
and patch checks; training requires an independent Validation cohort.

## Run the pipeline

Use this command and change the stage name as you progress:

```bash
envs/cellvit/.venv/bin/python run_pipeline.py \
  --config configs/my_experiment.yaml --stage molecular
```

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

### One sample first

With one HEST sample, run these commands in order:

```bash
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage molecular
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage convert
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage check
```

Look at `molecular/<ID>/labels.csv` and the patch audit. To train, add
samples from an independent Validation cohort, then run `assemble`,
`smoke`, and `train`. Set `cellvit.external_test_dataset` to a prepared
PanNuke-format dataset for `evaluate`. The `all` stage expects both
independent cohorts and the external test set.

Adding samples changes the run fingerprint. To reuse an earlier sample's
labels, set its `labels_csv` to the completed CSV in the new YAML. Keep that
as its single molecular source by removing any explicit `source_proc_adata`
or `input_adata` entry. Keep the CSV available at that path.

### Watching a long molecular run

Run `molecular` in `tmux` if you may disconnect. The useful files are:

```text
<run>/molecular/<ID>/run/status.json     current internal step
<run>/molecular/<ID>/run/plan.json       exact command, input, and output of each step
<run>/molecular/<ID>/run/<step>.log      detailed output from that step
```

A large slide may take hours, and the top-level terminal can stay quiet while
a calculation writes to its own log. If a step fails, keep the same YAML and
rerun `molecular` to resume completed internal steps.

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

## Verification status

As of 16 November 2026, source integrity, unit tests, one real HEST patch
conversion, CSV-to-nucleus matching, and a CellViT forward/backward/checkpoint
smoke test had passed. The complete HEST molecular-to-CellViT comparison
remains to be run. The smoke test checks software compatibility; it is not
an image-performance result. See the [validation record](docs/REPOSITORY_VALIDATION_20261116.md)
and [handoff checks](docs/INTEGRATED_VERIFICATION_20261116.md).

At that check, the server had one HEST sample, TENX142. A formal
Train/Validation comparison needs another independent cohort, matched
Bottom-up and proposed labels, an agreed training setup, and an external
test set. The available TENX142 Bottom-up artifact has zero Neoplastic
labels, so check class coverage when choosing samples.

## Repository map

| Path | Purpose |
| --- | --- |
| `integrated_pipeline/`, `run_pipeline.py` | CLI, stage orchestration, audits, training adapter |
| `base/`, `experimental_cell_gate/`, `run.py` | Molecular method and resumable runner |
| `tools/` | HEST checks, label export, patch conversion |
| `envs/`, `scripts/install.sh` | Environment setup |
| `tests/` | Unit checks and real-data smoke helpers |
| `docs/` | Detailed data contracts, outputs, verification records |
| `vendor/CDGE/` | CellViT and HEST processing code copied from the teacher project |

If a run stops, check `status.json` and the corresponding step log. Common
causes are mismatched sample and segmentation files, missing inputs,
overlapping cohorts, a missing cell class, or CUDA configuration.

For full file contracts and extra commands, see the
[pipeline guide](docs/INTEGRATED_CELLVIT_PIPELINE.md). Before public release,
review [third-party materials](THIRD_PARTY.md) and
[release blockers](RELEASE_BLOCKERS.md). Keep raw data, patient information,
weights, checkpoints, and generated runs outside Git.
