# Repository validation, 2026-11-16

This record describes the **new local Git checkout**, separate from the
pre-existing handoff. No public push or scientific image-model comparison was
performed. The Linux checks used already installed environments from the
private handoff; a clean installation of `scripts/install.sh` remains untested.
After the smoke test, the local `--stage all` entry point was corrected to run
Smoke before Train. Local syntax, CLI help, source verification, and all 11
unit tests passed after that change. A final sync and rerun on Linux were
attempted but the private server SSH connection timed out; the Linux smoke
result therefore applies to the candidate before this final CLI ordering fix.

| Check | Result | Scope and evidence |
| --- | --- | --- |
| Molecular source integrity | Passed | `python3 run.py verify`: 156 packaged source hashes and syntax checks. Seven historical scripts have path-only changes recorded in the source maps. |
| Data-free tests | Passed | 11 unit tests in `tests/test_*.py` on the local checkout and Linux candidate. |
| Syntax and CLI | Passed locally | `compileall` for Python using a writable temporary bytecode cache, `bash -n` for installation/check scripts, and `python3 -m integrated_pipeline --help`. |
| Real HEST input inspection | Passed with reported QC issue | TENX142 `proc_adata`: 237,763 cells; 237,762 registered nucleus IDs matched. The segmentation contains 247,636 polygons, of which 1,317 (0.5318%) are invalid. Preflight reports this fraction and accepts it under the current 1% engineering cutoff; one patch conversion passed. This does not establish correctness for all patches. |
| Single-sample preflight | Passed on TENX142 | Standalone `--stage preflight` now reports the missing Train/Validation cohort split as a warning; the real file, ID, environment, and GPU checks passed. The stricter `--require-train-val` check still exits nonzero for this one-sample configuration. |
| Real CSV-to-CellViT conversion | Passed on one patch | TENX142 patch 176: 7 instances, 1,019 labeled pixels, 6,714 Unknown pixels; PNG/NPY audit passed. The demonstration CSV was derived from teacher output, not a newly completed proposed molecular run. |
| Unknown type-loss handling | Passed | The real CellViT loss test found zero type-loss gradient for Unknown/class-11 pixels. Nucleus geometry remains supervised in binary/HV losses. |
| CellViT training interface | Passed as smoke only | Two real patches from the **same** TENX142 slide were assembled into technical folds; one training step completed forward, binary/HV/type/tissue loss, backward, parameter update, checkpoint save/reload, and inference with shape `[1, 6, 256, 256]`. The smoke report has `status=passed`, `steps=1`. These folds are not independent cohorts. |
| Private/large asset scan | Passed for obvious assets | `scripts/release_audit.py --files-only` found no obvious raw data, model weights, private keys, or files over 50 MiB among tracked/unignored files. It does **not** clear third-party rights or semantic privacy review. |
| Fresh Linux install | Not verified | The local machine lacks `uv`; the Linux candidate reused previously installed dependencies. CUDA/PyTorch compatibility must be tested on each destination server. |
| Full proposed molecular labels on HEST | Not verified | The corrected TENX142 run reached method stage 01 in the private handoff; no final four-class CSV is claimed from that run. |
| Independent Train/Validation cohorts | Not verified | Only one real HEST cohort was available in the checked environment. The assembly rejects overlapping or absent cohort IDs. |
| Full CellViT training and external evaluation | Not verified | No complete two-arm training or independent prepared PanNuke evaluation was run. Current evaluator is nucleus-conditioned classification; it does not reproduce teacher detection/segmentation scoring. |
| Public release rights | Blocked | See `THIRD_PARTY.md` and `RELEASE_BLOCKERS.md` for teacher code, collaborator code, marker resources, and project license. |

## Acceptance before a research comparison

1. Complete proposed molecular labeling for at least two independent HEST
   cohorts and inspect four-class counts, Unknown coverage, and per-stage logs.
2. Generate Bottom-up and proposed labels on **the same registered nuclei**;
   run the same CellViT configuration and report labeled nuclei per arm.
3. Agree and implement the independent PanNuke evaluation protocol, including
   handling of Dead cells and matching the teacher's detection/classification
   metrics if those are to be compared.
4. Test a clean installation and full CLI run on the destination server.

Until these checks pass, the repository is a validated **interface and smoke
test candidate**, not a demonstrated end-to-end performance improvement.
