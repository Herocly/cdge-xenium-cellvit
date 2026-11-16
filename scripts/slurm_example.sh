#!/usr/bin/env bash
# Edit the Slurm resource directives for the actual cluster before submission.
#SBATCH --job-name=cdge-xenium
#SBATCH --output=cdge-%j.out
set -euo pipefail

: "${CDGE_CONFIG:?Set CDGE_CONFIG to a configured JSON file}"
: "${CDGE_ANALYSIS_PYTHON:?Set CDGE_ANALYSIS_PYTHON}"
: "${CDGE_CNV_PYTHON:?Set CDGE_CNV_PYTHON}"
: "${CDGE_INPUT_ADATA:?Set CDGE_INPUT_ADATA}"
: "${CDGE_OUTPUT_DIR:?Set CDGE_OUTPUT_DIR}"

cd "$(dirname "$0")/.."
python3 run.py run --config "$CDGE_CONFIG" --variant "${CDGE_VARIANT:-frozen}"
