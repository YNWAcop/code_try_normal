"""Predeclared supplemental grouped CV, never a fresh independent test claim."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from ..settings import RUN_ROOT as ROOT, DEVICE, SMOKE, INPUT_MODE, BACKBONE
from ..common.metrics import scores
from ..common.checkpoint import tsave,loadck,state_cpu,rng_state,restore_rng,lock
CS=('T','TA','TV','TAV')
METRICS=('Acc7','Acc5','Acc2_nonzero','F1_weighted_nonzero','MAE','Corr')
CONFIG=dict(distribution=True,polarity=True,shared=True,attention=True)
POLICY=dict(
    version=1, experiment='supplemental frozen-algorithm grouped cross-validation',
    historical_design_exposure=True, pristine_independent_test=False,
    datasets=['mosi','mosei'],source_splits=['train','valid'],original_test_opened=False,
    methods=['C5','D24','RoME','TMDC'],conditions=list(CS),metrics=list(METRICS),
    outer_folds=5,outer_split='GroupKFold without shuffle, video groups',
    inner_validation='first split of five-fold GroupKFold on each outer training pool',
    refit_after_selection=False, outer_scores='compute only after all 40 method/dataset/fold units complete',
    stopping='finish predetermined folds regardless of results; no outer-driven redesign',
    seed_policy='no manual seed or seed sweep; ordinary stochastic training; RNG state for resumption',
    preprocessing='fit every scaler, ridge, reliability weight within component training videos',
    no_old_learned_weights=True,no_distillation=True,
    C5=dict(epochs=40,batch=64,lr=.0003,weight_decay=.001,hidden=128),
    D24=dict(core='same fresh C5 as C5 comparator',D5_epochs=20,anchor_epochs=40,cell_epochs=40,
        crossfit_folds=4,crossfit_epochs='fixed from main inner-validation component selection',
        text_blend=.25,cell_strength=.5,cell_margin=.4,warp_penalty=.001,warp_strength=.5,
        increment_gate_strength=.5,consensus_strength=.5,intensity_strength=1.,factor_blend=.25,
        factor_epochs=40,ensemble_members=5),
    own_checkpoint_selection='weighted mean of six scaled inner-valid metrics; T:1 TA:2 TV:2; earliest tie',
    selection_scales=dict(classification=.02,MAE=.05,Corr=.05),
    baseline_selection='retain adapter protocol: internal-valid nonzero Acc2 then lower MAE then earliest',
    baseline_features='original official feature files required, not substituted with C5 features',
    frozen_A_V_AV='old V1 files retained; not part of this text-present CV',
    report='pooled OOF metrics + fold variation + LA/LV gain vs L; no seed-run mean or significance claim')

POLICY.update(experiment='frozen D24-R3 original-split fresh retraining', methods=['C5','D24-R3'], outer_folds=0, outer_split='original train/valid/test', inner_validation='original valid only', outer_scores='single original test after both dataset selection locks', stopping='complete both fixed dataset jobs once', R3=dict(radius=.1, confidence=.1, regularization=.001, epsilon=1e-4), report='original test benchmark; historical test exposure remains; not independent confirmation')

POLICY.update(experiment='local C5+complete ERU TAV extension v1',
    package='local_ours_tav_20261124',input_mode=INPUT_MODE,backbone=BACKBONE,synthetic_smoke=SMOKE,device=DEVICE,
    own_checkpoint_selection='TAV only: mean of six scaled validation metrics, earliest tie',
    report='TAV primary; T/TA/TV auxiliary diagnostics from this new run, never replacements for old six-condition results',
    tav_core='text query over [T,A_res,V_res,A_shared,V_shared]; arithmetic mean residual/shared slots preserve head width',
    tav_eru='raw evidence head projects concatenated A_res,V_res,Q,mean_timeline; all ERU stages retained',
    training_conditions=list(CS),primary_condition='TAV',loss_average='four visible conditions equally weighted')

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Windows readers may briefly hold status.json without delete sharing.
    for attempt in range(30):
        try:os.replace(tmp,p);break
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def status(out,stage,**kw):
    opened=bool(kw.pop('test_opened',False))
    value=dict(time=time.time(),pid=os.getpid(),stage=stage,original_test_opened=opened,**kw)
    save(Path(out)/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)

def utility(pred,y):
    values=[];weights=[]
    for c in ('TAV',):
        m=scores(y,pred[c])['metrics']
        v=[]
        for k in METRICS:
            # Undefined correlation on constant predictions is the least favorable correlation.
            x=m[k];x=(-1. if k=='Corr' else 0.) if x is None else float(x)
            assert np.isfinite(x),(c,k,x)
            v.append((-x if k=='MAE' else x)/(.05 if k in ('MAE','Corr') else .02))
        values.append(np.mean(v));weights.append(1 if c=='T' else 2)
    return float(np.average(values,weights=weights))

def configure():
    if DEVICE == 'cpu':
        torch.set_num_threads(2); return
    assert torch.cuda.is_available(),'CUDA required for formal cloud training'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)

def ensure_policy():
    p=ROOT/'protocol.json'
    if p.exists():assert read(p)==POLICY,'Frozen protocol changed'
    else:save(p,POLICY)
    return sha(p)

def move_opt(opt):
    for s in opt.state.values():
        for k,v in s.items():
            if torch.is_tensor(v):s[k]=v.to(DEVICE)

def local_unlink(p):
    # Only disposable files generated by this CV, never parent experiment artifacts.
    p=Path(p).resolve();assert ROOT in p.parents and p.name in ('last.pt',)
    if p.exists():p.unlink()
