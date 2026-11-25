"""Validation-only metrics, definitions shared across the five cloud baselines."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

VERSION = 'full-valid-v1-20261117'
METRIC_KEYS = ('MAE', 'MSE', 'RMSE', 'Corr', 'Acc2_nonzero', 'F1_weighted_nonzero',
               'Acc2_haszero', 'F1_weighted_haszero', 'Acc5', 'Acc7')

def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def scores(y, pred):
    y, pred = np.asarray(y, dtype=np.float64).reshape(-1), np.asarray(pred, dtype=np.float64).reshape(-1)
    if not len(y) or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Empty, nonfinite, or misaligned predictions/labels')
    def binary(truth, guess):
        if len(truth)==0:
            return None, None
        f1 = 0.
        for label in (False, True):
            tp = int(((truth==label)&(guess==label)).sum())
            fp = int(((truth!=label)&(guess==label)).sum())
            fn = int(((truth==label)&(guess!=label)).sum())
            denominator = 2*tp+fp+fn
            f1 += (2*tp/denominator if denominator else 0.) * float((truth==label).mean())
        return float((truth==guess).mean()), f1
    nz = y != 0
    a0, f0 = binary(y>=0, pred>=0)
    an, fn = binary(y[nz]>0, pred[nz]>0)
    undefined = {}
    if np.std(y)==0 or np.std(pred)==0:
        corr = None
        undefined['Corr'] = 'Undefined: constant labels or constant predictions'
    else:
        corr = float(np.corrcoef(y,pred)[0,1])
    if not nz.any():
        undefined.update(Acc2_nonzero='No nonzero labels',F1_weighted_nonzero='No nonzero labels')
    mse = float(np.mean((pred-y)**2))
    metrics = dict(MAE=float(np.abs(pred-y).mean()), MSE=mse, RMSE=float(np.sqrt(mse)), Corr=corr,
                   Acc2_nonzero=an, F1_weighted_nonzero=fn, Acc2_haszero=a0, F1_weighted_haszero=f0,
                   Acc5=float((np.round(np.clip(pred,-2,2))==np.round(np.clip(y,-2,2))).mean()),
                   Acc7=float((np.round(np.clip(pred,-3,3))==np.round(np.clip(y,-3,3))).mean()))
    return dict(metrics=metrics, undefined_metrics=undefined, n_all=len(y),n_nonzero=int(nz.sum()),
                definitions={'MAE_MSE_RMSE_Corr':'All valid samples; unrounded continuous predictions',
                             'Acc2_nonzero_F1':'Exclude y==0; positive iff value>0; support-weighted F1',
                             'Acc2_haszero_F1':'Include y==0; positive iff value>=0; support-weighted F1',
                             'Acc5_Acc7':'Clip to [-2,2]/[-3,3], NumPy round-to-even, then exact accuracy',
                             'scale':'Acc/F1 are fractions 0..1, not percentage points'})

def replay_selection(history, key, patience, two_phase):
    minimize = key in ('Loss','MAE')
    best = 1e8 if minimize else 0
    best_epoch, saved_epoch, phase, saved_phase = 0, None, 1, None
    stopped = False
    for epoch, metrics in enumerate(history,1):
        value = metrics[key]
        improved = value<=best-1e-6 if minimize else value>=best+1e-6
        if improved:
            best,best_epoch,saved_epoch,saved_phase=value,epoch,epoch,phase
        if epoch-best_epoch>=patience:
            if two_phase and phase==1:
                phase=2;best_epoch=epoch;best=1e8 if minimize else 0
            else:
                if epoch!=len(history):
                    raise ValueError('History extends beyond reconstructed early stopping')
                stopped=True
    if not stopped or saved_epoch is None:
        raise ValueError('Cannot verify checkpoint selection from complete history')
    return saved_epoch,saved_phase

def expected_jobs():
    for protocol in ('whole','frame'):
        for method in ('EBMC','TLRA','API','LNLN','Self_MM'):
            for dataset in ('mosi','mosei'):
                for tag in (('A','V','T','AV','TA','TV') if protocol=='whole' else tuple(f'frame{i/10:.1f}' for i in range(1,8))):
                    yield f'{method}_{dataset}_{tag}', method, dataset, protocol, tag
