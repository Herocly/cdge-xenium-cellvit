# Environment status

The two `pyproject.toml` files transcribe package versions recorded in the
original B2.16 `ENVIRONMENT_LOCK.json`. `matplotlib` and `scvelo` are imported
by the executable source but were not versioned in that record. On the new
Linux server, the historical InSituCNV 0.1.1 installation recorded Git commit
`0f6176e9334bfe5309b6591e972b90353946e3f1` in its `direct_url.json`;
the CNV project now pins that exact source revision.
The old CNV environment used `umap-learn==0.5.12` and
`pynndescent==0.6.0`; both are pinned because their newer releases changed
the sparse neighbor graph representation and caused stage 04 to fail with
`AttributeError: 'csr_array' object has no attribute 'getnnz'`.

Both projects were resolved with `uv` on the new Linux server. Their
`uv.lock` files are included. Python 3.10.22 and 3.11.18 environments were
created and passed real Breast/Pancreas input preflight checks. This has not
yet established equality with the historical end-to-end outputs.

The separate `image/` project contains only dependencies needed for the HEST
CSV-to-patch adapter. It was locked and tested on the new server with Python
3.10.22. `openslide-bin` supplies the shared library missing from the migrated
historical environment. The test generated one real HEST H&E/label pair;
it did not train CellViT.

Recommended next steps on the Linux server:

1. Run `uv sync --project envs/analysis --locked --python 3.10`.
2. Run `uv sync --project envs/cnv --locked --python 3.11`.
3. For HEST patch export, run `uv sync --project envs/image --locked --python 3.10`.
4. Run `bash scripts/install.sh`, then `bash scripts/check_install.sh` from
   the repository root. Use `configs/integrated.example.yaml` for HEST inputs;
   the removed historical server wrapper is not needed on a new machine.
5. Compare a complete run with historical outputs before claiming parity.

The unversioned `matplotlib` and `scvelo` dependencies are fixed by the
included lockfiles. Do not substitute the newest InSituCNV Git HEAD.
