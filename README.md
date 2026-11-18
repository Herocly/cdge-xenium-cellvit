# CDGE Xenium to CellViT

## Set up your HEST samples

You need a Linux server, `uv`, Python 3.10 and 3.11, and your HEST files. The install script sets up the molecular, image, and CellViT environments. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) first if it is not on your server.

```bash
git clone https://github.com/Herocly/cdge-xenium-cellvit.git cdge-xenium-cellvit
cd cdge-xenium-cellvit
bash scripts/install.sh
bash scripts/check_install.sh
cp configs/integrated.example.yaml configs/my_experiment.yaml
```

Open `configs/my_experiment.yaml`. Set `hest_dir` to your HEST folder. List every HEST sample you want to use under `samples`, with one YAML entry per sample. Fill in its real `organ`, `cohort`, and `split`.

- `organ` tells the molecular method which marker resources to use.
- `cohort` is your name for the original data source. Samples from the same source stay together when you split Train and Validation.
- `split` is `train` or `val`. Put each cohort entirely in one split. Use separate cohorts for Train and Validation.

## Make labels and image patches

Run these commands with the same YAML file:

```bash
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage preflight
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage molecular
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage convert
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage check
```

1. `preflight` checks the HEST files, cell IDs, image coordinates, and environment. Read any items under `problems` before starting the long run.
2. `molecular` selects starting cells, calculates CNV, and expands cell labels over several rounds. This can take hours. If it stops, run the same command again to resume from the saved stages.
3. `convert` matches the cell labels to nuclei in the H&E image. It saves a PNG image and an NPY label file for each patch.
4. `check` reads those files back and checks the images, nuclei, classes, and matching results.

Results go under `<output_root>/<experiment_id>-<run ID>/`. Look for `molecular/<sample_id>/labels.csv`, the PNG/NPY files under `patches/`, and the check report at `audits/patch_compatibility.json`.

The run folder also keeps your config and logs. Run the same command again with the same config and inputs to reuse finished work. Changing the config or an input file creates a new run folder.

## HEST files

For each sample, the usual layout follows its sample ID. For example, `TENX142` uses:

```text
<hest_dir>/proc_adata/TENX142.h5ad
<hest_dir>/xenium_seg/TENX142_xenium_nucleus_seg.parquet
<hest_dir>/coordinates/patches/TENX142_patches.h5
<hest_dir>/wsis/TENX142.tif
<hest_dir>/metadata/TENX142.json
```

The `.h5ad` has cell expression data. The Parquet file has nucleus shapes and cell IDs. The H5 file has patch positions. The TIFF is the registered H&E image. The JSON has sample information.

## Train and test CellViT

After the Train and Validation samples have labels and patches, run:

```bash
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage assemble
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage smoke
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage train
```

`assemble` puts Train patches in `fold0` and Validation patches in `fold1`. `smoke` runs one training step, saves a model, and loads it again. `train` runs the full training and saves checkpoints.

For the final test, set `cellvit.external_test_dataset` in the YAML to a prepared PanNuke-format test set. Then run:

```bash
envs/cellvit/.venv/bin/python run_pipeline.py --config configs/my_experiment.yaml --stage evaluate
```

`evaluate` reports four-class F1 at the nucleus positions supplied by the test set. It skips PanNuke Dead nuclei and reports how many were skipped. Detection and segmentation have their own evaluations. After the split and test set are set up, `--stage all` runs the stages in order.

## Compare Bottom-up and our method

Use [the main example](configs/integrated.example.yaml) for our labels and [the Bottom-up example](configs/integrated.bottomup.example.yaml) for the teacher's labels. Use the same HEST samples, H&E images, nucleus shapes, Train/Validation split, CellViT checkpoint, and training settings in both runs. Give each run its own `experiment_id`.

Record the C/I/E/N and Unknown counts and how many nuclei enter training. Then test both models on the same dataset. The training loop in this repository is simpler than the teacher's original loop, so use this loop for both runs.

## Where to look

| Path | What is there |
| --- | --- |
| `run_pipeline.py`, `integrated_pipeline/` | The pipeline commands and CellViT training code |
| `base/methods/current/`, `experimental_cell_gate/`, `run.py` | Our molecular labeling method |
| `tools/` | HEST checks, label export, and patch conversion |
| `configs/` | YAML examples |
| `envs/`, `scripts/` | Installation and environment checks |
| `tests/` | Small code checks |
| `vendor/CDGE/` | The teacher's CellViT and HEST processing code |

For file formats and more detail on each stage, see [the pipeline guide](docs/INTEGRATED_CELLVIT_PIPELINE.md).
