"""Active four-class seed merger and evaluation metrics.

Functions copied verbatim from the final four-class research implementation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
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
