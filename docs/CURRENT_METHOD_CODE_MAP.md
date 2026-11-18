# Current molecular method: where each step runs

Start with `run.py run --variant latest`. The runner builds the stage plan in
`base/run_all.py`; `base/run_stage.py` maps each stage to a file in
`base/methods/current/`. The E-cell check between steps 10 and 11 lives in
`experimental_cell_gate/`.

| Steps | What runs | Main functions or evidence |
| --- | --- | --- |
| 01 | `code/01_prepare_label_free_analysis.py` | Counts check, PCA, neighbors, Leiden clusters |
| 02 | `code/02_build_cluster_evidence.py` | Original CDGE marker voting; structural and annotation scores |
| 03, 08 | `code/03_select_non_tumor_seeds.py` | Cluster selection, PCA-core cell selection |
| 04 | `code/05_run_insitucnv.py` | Bootstrap CNV reference |
| 05, 13 | `code/06_audit_cnv_candidates.py` | CNV candidate audit |
| 06, 14 | `code/09_build_neoplastic_evidence_ablation.py` | CNV and expression evidence |
| 07, 15 | `code/07_export_neoplastic_seeds.py` | Neoplastic seed candidates |
| 09 | `normal_seed_qc.py` → `screen_seeds` | Withhold low-information seed clusters |
| 10 | `code/10_combine_initial_seeds.py` | Original normal seed source |
| 10b | `experimental_cell_gate/binary_spa_e_gate.py` | E-cell marker check; C and I seeds retained |
| 11 | `normal_expansion.py` → `generate` | Temporary normal-cell classifier and expansion |
| 12 | `code/05_run_strict_ci_cnv.py` | CNV with expanded C/I reference |
| 16 | `code/seed_merge_metrics.py` → `merge_original_and_cnv` | Four-class initial seeds |
| 17 | `code/11_export_iterative_pseudolabels.py` | Balanced cell classifier, predict/select/retrain rounds |
| 18 | `base/run_stage.py` → `export_full_axis` | Restore QC-rejected cells as Unknown |

Both CNV scripts are kept because the bootstrap and C/I-reference passes use
different reference logic. Shared reliability functions live once under
`code/reliability/`. Marker databases, ontology and genomic coordinates live
once under `resources/`.

`SOURCE_MAP.json` records the original file and SHA-256 for every copied
source. The historical version directories are available in Git commit
`5a01e1f`; the current runner reads only `base/methods/current/`.
