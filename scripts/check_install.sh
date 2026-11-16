#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="$repo_root/envs/cellvit/.venv/bin/python"
"$python_bin" -m integrated_pipeline --help >/dev/null
"$python_bin" -m unittest discover -s tests -p 'test_*.py' -q
"$python_bin" tests/test_unknown_mask.py
"$python_bin" run.py verify
echo "Code and dependency checks passed. HEST data and CellViT training are separate checks."
