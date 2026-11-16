# Binary-SPA-style E gate in the seed-selection layer

The frozen `../base/` bundle remains unchanged. This variant inserts a
label-free stage **10b** after C/I/E seed assembly (stage 10) and before
normal expansion (stage 11). It withholds an existing E seed as `Unknown`
unless the Binary-SPA-style cell-level marker rule chooses E. C, I and the
bootstrap N seeds are untouched at stage 10b. No new seed is added.

The gate uses the same organ-filtered, top-20 marker panel and binary marker
count normalization as the earlier diagnostic. The highest class must be E,
its score must be positive, and its margin over the runner-up must be at least
0.15. It is a paper-inspired Python reproduction, not the full official
Binary-SPA pipeline.

The gated seed file feeds **both** stage 11 normal expansion and stage 16
four-class seed assembly. Because stage 11 feeds the strict C/I CNV reference,
stages 12-15 and the new N seeds must be recomputed for a full evaluation.
The earlier fixed-N 20-round E-only scores cannot be reported as this
integrated pipeline's final scores.

## Commands from this handoff folder

First set the four path variables described in the root `README.md` and set
the tissue and dataset ID in a copy of the example configuration. From this
folder's root:

```bash
python3 run.py run --config configs/slide.existing.example.json --variant binary-spa-e --check-only
python3 run.py run --config configs/slide.existing.example.json --variant binary-spa-e

# After an interruption, only when the config, code and input are unchanged:
python3 run.py run --config configs/slide.existing.example.json --variant binary-spa-e --resume
```

Use `recompute` instead of `existing` only when starting from raw counts.
Give each run a new `CDGE_OUTPUT_DIR`; the frozen B2.16 outputs are never
overwritten. The run records a status file, plan, environment preflight and
one log per stage.

To read progress:

```bash
cat "$CDGE_OUTPUT_DIR/status.json"
tail -30 "$CDGE_OUTPUT_DIR/10b_binary_spa_e_gate.log"
```

The selection output includes `marker_panel.json`, per-E decisions,
`gate_audit_no_official_labels.json`, and the full gated seed table.
Evaluation reference labels are read only by the final evaluation stage.

## Earlier development verification (not reproduced by this package)

On the frozen stage-10 inputs, the gate retained 842 of 848 Breast E seeds
and 971 of 2,708 Pancreas E seeds. Both selected E cell-ID sets match the
previous fixed-N E-only diagnostic exactly. C/I/bootstrap-N seed counts
were unchanged. Both slide configurations passed the full-plan preflight.
Pancreas stage 11 normal expansion completed all 20 rounds using the gated
stage-10 seed file. A stage-16 contract test with the *old* N-seed file also
completed and matched the earlier fixed-N E-only four-class seed table
cell-for-cell. That contract test does not substitute for regenerating N.
CNV, new N seeds, and final four-class evaluation have not yet been run for
this integrated variant, so no new end-to-end metric is claimed.
