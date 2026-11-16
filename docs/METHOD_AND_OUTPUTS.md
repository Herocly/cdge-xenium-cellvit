# Scientific stages and label meanings

This note describes the executable B2.16 lineage, not a new algorithm.

## Reliable C/I/E starting cells

The initial input is Xenium cell-by-gene counts and spatial coordinates.
Stage 01 computes a 30-dimensional expression PCA, a neighbor graph and
Leiden clusters. Stage 02 annotates each cluster using the fixed public
marker databases and Cell Ontology policy. It separately computes:

- **Structural reliability:** mean of within-slide percentile scores for
  higher silhouette, higher same-cluster neighbor consistency, higher
  expression cohesion, lower within-cluster dispersion, and lower dip
  statistic.
- **Annotation confidence:** mean of within-slide percentile scores for
  marker-vote confidence, vote margin and marker coverage. Source support
  from the two databases is a separate gate, not a calibrated probability.

Stage 03/08 selects clusters passing Tier A or fallback Tier B. Tier A requires
both scores at least 2/3, a positive vote margin and database source support.
Tier B is used only when a class lacks Tier A clusters; it requires both
scores at least 0.50, marker coverage at least 0.40, vote margin at least
0.25 and source support. From each selected cluster, the implementation ranks
cells by Euclidean distance to its 30-PC centroid and takes the nearest 30%
before applying the original per-class seed budget. The stage-08 policy may
use multiple clusters. Stage 09 withholds seed clusters with low total counts
**and** low detected-gene medians. These are cell selection and quality-control
steps; the cluster label is not assigned to every cluster member.

The optional `binary-spa-e` stage 10b adds a different kind of cell check:
for each original E seed, it scores detected marker genes in that **cell**.
Only cells whose top class is E with positive support and margin at least
0.15 remain E seeds; rejected E seeds become Unknown. This gate has only
been implemented for E in the provided candidate runner. It is not a
normal-versus-tumor proof, and the integrated end-to-end result is unverified.

## Neoplastic starting cells

The historical bootstrap stages 03–07 supply the original seed-source
lineage. Stage 11 trains a temporary three-lineage classifier from C/I/E
seeds and expands normal cells. Its strict C/I subset becomes the CNV normal
reference in stage 12; E is not a normal CNV reference. Stages 13–15 combine
CNV and tumor-related expression evidence to obtain new N candidate seeds.
Stage 16 combines the **original** C/I/E seeds with these **new** N seeds,
withholding contradictory cells. Expanded normal labels do not replace the
original C/I/E four-class starting set.

## Four-class iteration and evaluation

Stage 17 fits a balanced logistic-regression cell classifier to the first
30 PCA coordinates of accepted four-class training cells. It predicts
unlabeled cells, applies confidence, top-two margin, expression-neighbor
agreement and early-round cluster/evidence rules, accepts a bounded number
of pseudo-labels, and refits for at most 20 rounds. The first six rounds
retain stronger cluster/evidence restrictions; later rounds release those
hard restrictions but keep the cell-level confidence rules.

Two output concepts must remain separate:

1. `final_pseudolabels_no_official_labels.csv`: initial seeds plus labels
   actually accepted into training, with `seed_round`. Unaccepted cells are
   Unknown here.
2. `iterative_all_cell_predictions_no_official_labels.csv`: the final model's
   predictions for all processed cells, with confidence and a selective
   four-class/Unknown decision. Stage 18 restores preprocessing-rejected
   input cells as Unknown on the full input cell axis.

Stage 19, if given an external `cell_id,truth` reference, reports selective
precision among non-Unknown predictions, coverage on the reference cell
population, per-class F1 and macro F1. Unknown counts as a miss in F1.
The reference is not used by generation. A reference annotation is not
automatically an independent biological ground truth.

## Exact code pointers

- `base/methods/b1_9_coordinate_contract_support50_candidate_20261003/code/02_build_cluster_evidence.py`
- `base/methods/b1_9_coordinate_contract_support50_candidate_20261003/code/03_select_non_tumor_seeds.py`
- `base/methods/b2_7_qc_multicluster_normal_candidate_20261010/normal_reference.py`
- `base/methods/b2_14_teacher_constraint_ablation_candidate_20261011/normal_reference.py`
- `base/methods/b2_15_expanded_reference_cnv_candidate_20261011/code/05_run_insitucnv.py`
- `base/methods/b2_16_ci_cnv_four_class_candidate_20261011/code/11_export_iterative_pseudolabels.py`
- `experimental_cell_gate/binary_spa_e_gate.py`
