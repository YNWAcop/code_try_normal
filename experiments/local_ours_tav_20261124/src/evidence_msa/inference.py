"""Unified routing for the final model; no label is read during prediction."""
from pathlib import Path
import gc
import tempfile
import numpy as np
import joblib
import torch
from .common.checkpoint import loadck
from .common.metrics import digest

ALIASES = {'L':'T', 'LA':'TA', 'AT':'TA', 'LV':'TV', 'VT':'TV', 'VA':'AV'}


def canonical(condition):
    condition = ALIASES.get(condition.upper(), condition.upper())
    if condition not in ('T','TA','TV','TAV','A','V','AV'):
        raise ValueError('Supported conditions: T/A/V/TA/TV/AV/TAV (L/LA/LV aliases accepted)')
    return condition


def features(path):
    from .text_present.data import KEYS, _normalize
    with np.load(path, allow_pickle=False) as z:
        # Deliberately do not request y, even when the NPZ contains it.
        d = {k:np.array(z[k]) for k in KEYS+('ids','groups')}
    d['ids'] = _normalize(d['ids'])
    d['groups'] = _normalize(d['groups'])
    return d


def predict_text(d, model_dir, conditions=('TAV',)):
    from .text_present.protocol import configure, read
    from .text_present.runner import transform_test, repair, raw
    from .text_present.train import load_composite, composite, predict, FactorAnchor, blend_factor
    from .text_present.calibration import apply_postprocessors, load as loadpost
    from .text_present.protocol import CS, DEVICE
    from .input_views import select_inputs
    d=select_inputs(d,'enhanced')
    configure()
    out = Path(model_dir).resolve()
    sealed = read(out/'selection_lock.json')
    for name,h in sealed['artifacts'].items():
        if digest(out/name) != h: raise RuntimeError('Checkpoint hash mismatch: '+name)
    dims = read(out/'prep_main/data_complete.json')['dims']
    with tempfile.TemporaryDirectory(prefix='tav_inference_',dir=out.parent) as td:
        tmp = Path(td)
        hd = transform_test(d, out/'prep_main', tmp/'main')
        models = load_composite(dims,out/'main')
        z = composite(models,hd)
        del models
        gc.collect();torch.cuda.empty_cache()
        subpred = []
        for j in range(4):
            folder = out/f'sub_{j}'
            h = transform_test(d, folder/'prep', tmp/f'sub_{j}')
            model = load_composite(dims, folder/'models')
            zz = composite(model,h)
            subpred.append({c:zz['pred_'+c] for c in CS})
            del h, model
            gc.collect();torch.cuda.empty_cache()
        mean = {c:np.mean([v[c] for v in subpred],axis=0) for c in CS}
        post = loadpost(out/'post/complete.pt')
        base = apply_postprocessors(z,mean,post)
        factor = FactorAnchor(dims['T']).to(DEVICE)
        factor.load_state_dict(loadck(out/'factor/best.pt')['model'])
        adjusted = blend_factor(base,predict(factor,hd,'factor')['pred'])
        polarity = joblib.load(out/'polarity.joblib')
        final = {c:repair(adjusted[c],polarity[c].predict_proba(raw(d,slice(None),c))[:,1]) for c in conditions}
        del hd, factor
        gc.collect();torch.cuda.empty_cache()
    return final


def predict_absent(d, model_dir, condition, device='cuda'):
    """Use the preserved V1 checkpoint for exactly one A/V/AV route."""
    from .text_absent import v1
    condition = canonical(condition)
    if condition not in ('A','V','AV'): raise ValueError(condition)
    folder = Path(model_dir)
    import json
    sealed = json.loads((folder/'validation_lock.json').read_text(encoding='utf-8'))
    if sealed.get('condition') != condition: raise ValueError('Wrong V1 condition checkpoint')
    if digest(folder/'best.pt') != sealed['checkpoint_sha256']: raise RuntimeError('V1 checkpoint hash mismatch')
    scaled = {k:np.array(v,copy=True) for k,v in d.items()}
    v1.apply_scalers(scaled,joblib.load(folder/'scalers.joblib'))
    b = v1.arrays(scaled,'reliable')
    # The V1 no-text route never uses the text anchor or text-derived residual.
    b.update(anchor=np.zeros(len(d['ids']),dtype='float32'), A_res=b['A'], V_res=b['V'])
    ck = loadck(folder/'best.pt')
    model = v1.Evidence(ck['dims'],ck['appdim']).to(device)
    model.load_state_dict(ck['model']);model.eval()
    bits = torch.tensor([[m in condition for m in 'TAV']],dtype=torch.bool)
    parts=[]
    with torch.inference_mode():
        for start in range(0,len(d['ids']),256):
            batch = v1.tensors(b,slice(start,start+256),device)
            parts.append(model(batch,bits.expand(len(batch['T']),-1)).cpu().numpy())
    return np.concatenate(parts)
