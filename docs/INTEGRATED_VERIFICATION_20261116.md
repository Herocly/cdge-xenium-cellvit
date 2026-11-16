# Integrated pipeline verification, 2026-11-16

This record separates **interface tests** from a scientific CellViT result.
The real-data checks below were run on a Linux research server against the
pre-publication handoff. The new Git checkout has separately passed source
integrity and 11 data-free unit tests; a fresh-clone CellViT training test is
still required. Machine-specific paths are omitted here.
The teacher source snapshot was taken from CDGE commit
`d97e0e2ab658b74903b42c7d95c16c29eec6f539`; SHA-256 of the vendored
patch writer and training YAML matches the teacher repository byte-for-byte.
Original logs and generated data remain in the private handoff; none are part
of this Git checkout.

| Check | Status | Evidence / limit |
| --- | --- | --- |
| Portable config and distinct runs | **Passed** | Nine server-side unit tests (`reviews/integrated_pipeline_20261116/unit_tests.log`), including two in `tests/test_integrated_config.py`; venv symlink is preserved, CSV content and sample changes alter run ID. |
| Environment preflight | **Passed on checked server** | All four configured interpreters imported their required modules; CellViT Python detected one CUDA GPU with torch 2.1.1+cu121. `reviews/integrated_pipeline_20261116/environment_report.json`. Other servers still need their own check. |
| HEST modality audit | **Passed** | `st/TENX142.h5ad` has 3,743 spot IDs and zero overlap with H&E nuclei; the pipeline rejects it. `proc_adata/TENX142.h5ad` has 237,763 rows, 237,762 matching nucleus IDs. |
| Label-free HEST cell input | **Passed** | 237,762 cells, 541 genes, raw counts, no copied annotations/clusters, registered nucleus centroids converted with WSI MPP 0.2737063915. See `runs_integrated/TENX142_cell_input.provenance.json`. This is an input construction check, not a biological label result. |
| Dense/sparse raw-count support | **Passed on synthetic matrices** | `tests/check_count_reader.py` showed identical selected counts from HDF5 dense and AnnData CSR encoding. Only the dense path has been exercised on real TENX142. |
| Input identity preflight | **Passed** | All 237,762 prepared cell IDs match registered nuclei; `tools/check_hest_identity.py` reported fraction 1.0. |
| Real H&E/CSV/NPY conversion | **Passed** | Teacher-label **demonstration** CSV for TENX142 patch 176 produced PNG, positive `inst_map`, and `type_map`; audit found 7 instances, 1,019 known pixels, 6,714 Unknown pixels. `run_pipeline.py --stage convert` and `--stage check` succeeded, including rerun/reuse. |
| Existing teacher Bottom-up output | **Passed for export and one patch** | The teacher's `TENX142_bottomup_cellmarker_neoplastic.h5ad` exported by `tools/export_teacher_bottomup_labels.py` to 237,762 cell rows. It has 76,708 I, 17,135 C, 1,474 E, 142,445 Unknown, and **0 N**; one real H&E patch passed CSV/PNG/NPY audit. This is precomputed teacher labeling, not a rerun of his Bottom-up algorithm. |
| One-label perturbation and isolation | **Passed** | A synthetic CSV changes one C label to I. Separate run directories; PNG bytes identical, 0 changed `inst_map` pixels, exactly 173 changed `type_map` pixels, all within nucleus instance 7. `tests/check_one_label_change.py` passed. |
| Actual CellViT loader + gradient | **Passed** | Vendored PanNukeDataset loaded real HEST patch PNG/NPY. `CellViT256Shared` ran forward, binary/HV/type/tissue losses, backward, finite gradients, parameter update, checkpoint save/load, and reloaded inference shape `[1,6,256,256]`. `reviews/integrated_pipeline_20261116/real_train_smoke_v2/real_data_smoke_summary.json`. Two patches are from **one slide**, so this is only a code smoke test. |
| Unknown type-loss exclusion | **Passed** | `tests/test_unknown_mask.py` used the real CellViT loss classes. Changing a class-11 pixel did not change type loss, and its type-loss gradient was exactly zero. |
| Full latest molecular run on HEST | **Running / not validated** | Initial attempt on spot-level input was rejected by cell/nucleus ID check. The corrected per-cell TENX142 run passed method preflight and entered stage 01. Scanpy 1.10.2/SciPy 1.14.1 hit `csr_array.getnnz`; the analysis environment was pinned to SciPy 1.11.4 (where that API exists) and the run resumed. Check `runs_integrated/TENX142_cell_latest/status.json` and `reviews/integrated_pipeline_20261116/TENX142_cell_latest_resume.log`. No final labels are claimed. |
| Cohort-separated HEST train/val assembly | **Not verified with real data** | Only TENX142 is present on this server. The code rejects missing/overlapping train and validation cohorts. A second independent HEST cohort is required. |
| Dataset assembly safety | **Passed with a real patch in a temporary test** | Two artificial cohort IDs using the same TENX142 patch were assembled only to exercise file layout; both folds passed PNG/NPY/metadata audit. Assembly now publishes the directory only after both folds pass checks. This does not establish independent scientific cohorts. |
| Full CellViT training and PanNuke test | **Not run** | No independent prepared PanNuke test dataset or two-arm HEST labels on the checked server. No image-level accuracy, F1, Dice, PQ, or detection result exists from this integration. |
| Teacher-regimen parity | **Not achieved** | The adapter shares the teacher's model, loader, and loss classes but omits original augmentation, weighted sampling, mixed precision, and exact validation scheduling. Unknown-aware type-loss masking is an explicit necessary change. A paper comparison must fix the regimen across arms and disclose the adaptation. |

## Scientific/data changes made by the integration

1. The original HEST nucleus writer assigns instance ID 0 to the first cell;
   CellViT reserves 0 for background. The adapter shifts **instance numbers**
   by +1 without changing cell polygons or semantic classes.
2. The HEST per-cell source contains no aligned H&E spatial coordinates in
   the form required by the molecular method. The converter joins each cell
   ID to the registered nucleus polygon centroid and multiplies H&E pixels by
   WSI MPP. This is the documented coordinate contract, not a new marker rule.
3. The original CellViT type one-hot loss cannot ingest class ID 11. The
   integrated trainer excludes only Unknown pixels from the type loss, while
   preserving their nucleus geometry in binary/HV losses. Both experimental
   arms must use the same handling.
4. HEST `Ovary` is normalized to the teacher CellViT tissue name `Ovarian`
   only in the image dataset. Molecular organ filtering retains `Ovary`.

## Next required acceptance gates

1. Finish the corrected HEST molecular run and inspect C/I/E/N seed counts,
   Unknown fraction, and final label CSV. Do not weaken thresholds merely to
   force this sample through.
2. Add at least one independent HEST cohort and verify the split manifest;
   no slide/patient/cohort may cross train and validation.
3. Produce **both** original Bottom-up and proposed label CSVs for the same
   HEST nuclei. Audit ID coverage and four-class distributions before training.
4. Agree one CellViT training protocol, especially augmentations, sampler,
   epochs, encoder schedule, weight initialization, and Unknown handling.
5. Prepare independent PanNuke test folds and agree the Dead-cell policy.
   Evaluate full detection/segmentation separately from nucleus-conditioned
   classification. Only then report a baseline-versus-ours image result.
