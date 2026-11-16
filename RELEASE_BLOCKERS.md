# Public release gate

**This is a local Git repository candidate, not an approved public release.**

The supported CLI and the preserved molecular bundle are intended to remain
scientifically unchanged. Public GitHub publication needs these decisions:

1. Obtain permission or a public license for the teacher's modified HEST
   writer and the bundled `base/methods/**/vendor` code.
2. Establish source, version, notice, and redistribution rights for marker
   database snapshots and Cell Ontology resources. Keep the CellViT license
   and required citation with any permitted code distribution.
3. Choose a license for the research team's own code with the relevant authors.
4. Review the path-only edits to seven historical source files and verify the
   updated source manifests against the original snapshots.
5. Test `scripts/install.sh` in a fresh Linux environment with the target CUDA
   driver. The current machine does not have `uv`, so local installation has
   not been tested from scratch.
6. Complete a true HEST cohort-separated Train/Validation experiment and an
   independent PanNuke evaluation. One real TENX142 patch has passed the
   CellViT loader/gradient smoke test; this is not a performance result.
7. Decide whether to match the teacher's augmentation, sampling, AMP, and
   validation schedule. The current adapter intentionally differs; use exactly
   the same adapter and settings in both label-source arms until resolved.
8. Confirm the external test metric: current `evaluate` is type classification
   at reference nuclei, not the original full detection/segmentation protocol.

No public push should occur while items 1–4 are unresolved. Items 5–8 are
scientific reproducibility gates; the README reports their current status.
