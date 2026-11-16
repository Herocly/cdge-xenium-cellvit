#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. This installs only software, never HEST or weights.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

uv sync --project envs/analysis --frozen --python 3.10
uv sync --project envs/cnv --frozen --python 3.11
uv sync --project envs/image --frozen --python 3.10
uv venv envs/cellvit/.venv --python 3.10

# CUDA wheels differ between servers. Specify the official PyTorch wheel index
# for this machine when needed; otherwise use the standard index and let the
# strict preflight detect an unusable CUDA installation before training.
if [[ -n "${CDGE_TORCH_INDEX_URL:-}" ]]; then
  uv pip install --python envs/cellvit/.venv/bin/python \
    --index-url "$CDGE_TORCH_INDEX_URL" torch==2.1.1 torchvision==0.16.1
fi
uv pip install --python envs/cellvit/.venv/bin/python \
  -r envs/cellvit/requirements.txt
uv pip install --python envs/cellvit/.venv/bin/python -e .
echo "Installed. Run scripts/check_install.sh before adding HEST data."
