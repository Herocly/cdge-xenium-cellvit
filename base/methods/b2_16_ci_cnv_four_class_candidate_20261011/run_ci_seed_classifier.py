"""Expanded C/I CNV -> D seeds + original normal seeds -> selective classification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]  # legacy direct-run fallback; supported CLI passes explicit paths
CNV_RUN = PROJECT / 'outputs/b2_15_expanded_reference_cnv_20261011'
sys.path.insert(0, str(ROOT/'code'))
from reliability.data_contracts import aligned_cell_table, boolean_flags

CLASSES = ('Connective_tissue', 'Inflammatory', 'Epithelial', 'Neoplastic')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def merge_original_and_cnv(normal, neoplastic, cells):
    normal = aligned_cell_table(normal, cells)
    neoplastic = aligned_cell_table(neoplastic, cells)
    original = boolean_flags(normal.is_initial_seed)
    is_normal = original & normal.training_label.isin(CLASSES[:3]).to_numpy()
    is_neo = boolean_flags(neoplastic.is_neoplastic_seed)
    conflict = is_normal & is_neo
    labels = np.full(len(cells), 'Unknown', dtype=object)
    labels[is_normal & ~conflict] = normal.training_label.to_numpy()[is_normal & ~conflict]
    labels[is_neo & ~conflict] = 'Neoplastic'
    return pd.DataFrame({'cell_id': np.asarray(cells).astype(str),
        'training_label': labels, 'is_initial_seed': labels != 'Unknown',
        'seed_source': np.where(conflict, 'withheld_conflicting_seeds',
            np.where(labels == 'Neoplastic', 'expanded_ci_cnv_D',
                     np.where(labels != 'Unknown', 'original_normal_seed', ''))),
        'original_normal_candidate': is_normal, 'cnv_D_candidate': is_neo,
        'seed_conflict': conflict})


def per_class_metrics(predictions, reference, method):
    if not predictions.cell_id.is_unique or not reference.cell_id.is_unique:
        raise ValueError('Duplicate cell IDs')
    if not set(reference.cell_id).issubset(set(predictions.cell_id)):
        raise ValueError('Reference cells missing from predictions')
    if not set(reference.truth).issubset(CLASSES):
        raise ValueError('Unexpected evaluation reference classes')
    if not set(predictions.predicted_class).issubset((*CLASSES, 'Unknown')):
        raise ValueError('Unexpected predicted classes')
    d = reference.merge(predictions[['cell_id', 'predicted_class']], on='cell_id', validate='one_to_one')
    rows = []
    for label in CLASSES:
        t, p = d.truth.eq(label), d.predicted_class.eq(label)
        tp, called, total = int((t&p).sum()), int(p.sum()), int(t.sum())
        rows.append(dict(method=method, class_name=label, reference_cells=total,
            predicted_cells_all=int(predictions.predicted_class.eq(label).sum()),
            predicted_cells_evaluable=called, correct=tp,
            precision=tp/called if called else None, correct_class_coverage=tp/total if total else None,
            f1_all_reference_cells=2*tp/(called+total) if called+total else 0))
    return pd.DataFrame(rows), pd.crosstab(d.truth, d.predicted_class).reindex(
        index=CLASSES, columns=(*CLASSES, 'Unknown'), fill_value=0)


def check_hashes(items):
    for item in items.values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Changed fixed input: '+item['path'])


def wait_for_ci(slide, state, save):
    key = slide+'/expanded_strict_ci'
    started = time.monotonic()
    while True:
        status = json.loads((CNV_RUN/'status.json').read_text())
        if status.get('runs', {}).get(key) == 'completed':
            return CNV_RUN/slide/'expanded_strict_ci'
        if status.get('status') == 'failed':
            raise RuntimeError('Upstream CNV failed: '+str(status.get('error')))
        if time.monotonic()-started > 24*3600:
            raise TimeoutError('C/I CNV result not ready after 24 hours')
        state['slides'][slide] = 'waiting_for_expanded_ci_cnv'
        save()
        time.sleep(60)


def run_slide(slide, cnv, root):
    upstream = json.loads((cnv/'generation_manifest_no_official_labels.json').read_text())
    if set(upstream['reference_classes']) != set(CLASSES[:2]):
        raise ValueError('Only C/I-reference CNV is allowed')
    check_hashes(upstream['inputs'])
    check_hashes(upstream['outputs'])
    baseline = PROJECT/'outputs'/slide/'b2_12_feedback_audit_20261011_frozen_classifier/normal_reference'
    normal_manifest = json.loads((baseline/'generation_manifest_no_official_labels.json').read_text())
    inputs = normal_manifest['inputs']
    check_hashes({k:inputs[k] for k in ('adata', 'normal_seed_source', 'cluster_evidence')})
    out = root/slide
    out.mkdir(exist_ok=False)
    adata_path = Path(inputs['adata']['path'])
    adata = sc.read_h5ad(adata_path, backed='r')
    cells = adata.obs_names.astype(str)
    if any(any(x in str(c).lower() for x in ('annotation', 'ground_truth', 'official_label', 'cell_type', 'truth')) for c in adata.obs.columns):
        raise ValueError('Annotation-like column in generation AnnData')
    adata.file.close()
    normal_source = Path(inputs['normal_seed_source']['path'])
    neo_source = cnv/'neoplastic_seeds/neoplastic_seed_candidates.csv'
    merged = merge_original_and_cnv(pd.read_csv(normal_source, dtype={'cell_id':str}),
                                   pd.read_csv(neo_source, dtype={'cell_id':str}), cells)
    merged.to_csv(out/'combined_initial_seeds_no_official_labels.csv', index=False)
    counts = merged.loc[merged.is_initial_seed].training_label.value_counts().to_dict()
    (out/'seed_merge_audit_no_official_labels.json').write_text(json.dumps(dict(
        counts=counts, conflicting_cells=int(merged.seed_conflict.sum()),
        policy='original normal seeds only; old Neoplastic seeds ignored; new C/I CNV D only; contradictory seeds withheld'), indent=2))
    if set(counts) != set(CLASSES):
        raise RuntimeError('Four-class training requires a nonempty seed population in each class; counts='+str(counts))
    iteration = out/'iteration'
    with (out/'run.log').open('x') as log:
        subprocess.run([sys.executable, str(ROOT/'code/11_export_iterative_pseudolabels.py'),
            '--adata', str(adata_path), '--seed-file', str(out/'combined_initial_seeds_no_official_labels.csv'),
            '--cluster-evidence', inputs['cluster_evidence']['path'],
            '--neoplastic-evidence', str(cnv/'neoplastic_evidence/neoplastic_evidence_table_no_official_labels.csv'),
            '--out-dir', str(iteration), '--max-iterations', '20', '--cluster-release-rounds', '6',
            '--support-min-structural', '.5', '--support-min-annotation', '.5'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    fixed_inputs = [adata_path, normal_source, neo_source,
        Path(inputs['cluster_evidence']['path']),
        cnv/'generation_manifest_no_official_labels.json',
        cnv/'neoplastic_evidence/neoplastic_evidence_table_no_official_labels.csv',
        out/'combined_initial_seeds_no_official_labels.csv']
    fixed_outputs = list(iteration.glob('*.csv'))
    manifest = dict(official_labels_policy='not_loaded_during_generation',
        reference_policy='expanded strict C/I only; reuse completed label-free CNV',
        training_policy='original C/I/E seeds plus new D seeds, never expanded normal labels or old D',
        unknown_policy='classifier predicts all cells; confidence .8, margin .2, prediction-neighbor agreement .7; abstain otherwise',
        schedule='cluster ranking weights .15 -> 0 over six rounds, confidence .5 -> .65; cluster/C hard support removed from round seven; fixed cell gates',
        inputs={str(k):dict(path=str(p),sha256=sha(p)) for k,p in enumerate(fixed_inputs)},
        outputs={str(k):dict(path=str(p),sha256=sha(p)) for k,p in enumerate(fixed_outputs)})
    (out/'generation_manifest_no_official_labels.json').write_text(json.dumps(manifest, indent=2))
    # Evaluation is isolated until both prediction vectors and their manifest are fixed.
    provenance = json.loads((baseline/'evaluation_provenance_only.json').read_text())
    reference_path = Path(provenance['evaluation_only_source'])
    if sha(reference_path) != provenance['evaluation_only_source_sha256']:
        raise RuntimeError('Evaluation reference changed')
    check_hashes(manifest['inputs']); check_hashes(manifest['outputs'])
    reference = pd.read_csv(reference_path, usecols=['cell_id','truth'], dtype={'cell_id':str})
    sources = dict(seed_only_classifier=iteration/'seed_only_all_cell_predictions_no_official_labels.csv',
        iterative_classifier=iteration/'iterative_all_cell_predictions_no_official_labels.csv',
        original_cdge=PROJECT/f'outputs/original_cdge_commit_baseline_20261011/{slide}/predictions_no_official_labels.csv')
    metrics = []
    for name,path in sources.items():
        frame = pd.read_csv(path, dtype={'cell_id':str})
        if set(frame.cell_id) != set(cells):
            raise ValueError('Comparison cell population differs')
        rows, confusion = per_class_metrics(frame, reference, name)
        metrics.append(rows)
        confusion.to_csv(out/f'{name}_confusion_evaluation_only.csv')
    comparison = pd.concat(metrics, ignore_index=True)
    comparison.to_csv(out/'classifier_comparison_evaluation_only.csv', index=False)
    f1 = comparison.pivot(index='class_name', columns='method', values='f1_all_reference_cells')
    f1['iterative_minus_seed_only'] = f1.iterative_classifier-f1.seed_only_classifier
    f1['iterative_minus_original_cdge'] = f1.iterative_classifier-f1.original_cdge
    f1.to_csv(out/'f1_difference_evaluation_only.csv')
    (out/'evaluation_provenance_only.json').write_text(json.dumps(dict(
        reference=str(reference_path),reference_sha256=sha(reference_path),evaluated_cells=len(reference),
        prediction_hashes={k:sha(p) for k,p in sources.items()}, policy='same reference cells; Unknown counts as missed'), indent=2))
    print(slide, comparison.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=False)
    files = [p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts]
    code = {str(p):sha(p) for p in files}
    state = dict(status='running',slides={},method_hashes=code)
    def save():
        temp = root/'status.tmp'; temp.write_text(json.dumps(state,indent=2)); temp.replace(root/'status.json')
    save()
    try:
        for slide in ('rep1','pancreas'):
            cnv = wait_for_ci(slide,state,save)
            if any(sha(p)!=value for p,value in code.items()):
                raise RuntimeError('Running candidate modified')
            state['slides'][slide]='training_and_predicting'; save()
            run_slide(slide,cnv,root)
            state['slides'][slide]='completed'; save()
        state['status']='completed'
    except Exception as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
