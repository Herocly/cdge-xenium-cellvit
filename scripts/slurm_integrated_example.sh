#!/usr/bin/env bash
#SBATCH --job-name=cdge-cellvit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# Adjust resource directives to the receiving cluster's limits.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch scripts/slurm_integrated_example.sh /path/to/experiment.yaml" >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="$1"
exec "$project_root/envs/cellvit/.venv/bin/python" "$project_root/run_pipeline.py" --config "$config_path" --stage all
