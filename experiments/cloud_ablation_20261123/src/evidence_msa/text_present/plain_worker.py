"""A4 only: same loss/selection/40 epochs, no ERU fitting or inference."""
import argparse,gc
import numpy as np
import torch
from .protocol import *
from . import data as cv_data
from .runner import transform_test
from ..input_views import select_inputs
from .train import fresh_core,fit_component,predict,npz
def train(ds):
    from .supervisor import check_code
    check_code();configure();assert BACKBONE=='simple';ensure_policy()
    out=ROOT/'runs'/ds/'fold_0';out.mkdir(parents=True,exist_ok=True)
    with lock(out/'ours.lock'):
        if (out/'ours_done.json').exists():
            for n,h in read(out/'selection_lock.json')['artifacts'].items():assert sha(out/n)==h
            return
        m=read(ROOT/'manifests'/ds/'fold_0.json');data,dims=cv_data.prepare(ds,m['train_indices'],{'valid':m['valid_indices']},out/'prep_main')
        model,epoch=fit_component(fresh_core(dims),'C5',data['train'],data['valid'],out/'main/C5',1 if SMOKE else 40,lambda p:p)
        ps=predict(model,data['valid'],'C5');npz(out/'validation_predictions.npz',dict(ids=data['valid']['ids'],y=data['valid']['y'],**{'c5_'+c:ps[c] for c in CS}))
        save(out/'parameters.json',model.parameter_match)
        names=['main/C5/best.pt','prep_main/scalers.joblib','prep_main/A_shared.joblib','prep_main/V_shared.joblib']
        save(out/'selection_lock.json',dict(artifacts={n:sha(out/n) for n in names},selected_epoch=epoch,policy_sha=ensure_policy(),held_used=False))
        save(out/'ours_done.json',dict(files={'selection_lock.json':sha(out/'selection_lock.json')},test_opened=False))
        del model,data;gc.collect();cv_data.cleanup_arrays(out/'prep_main');status(out,'training_complete',dataset=ds,test_opened=False)
def evaluate():
    from .supervisor import check_code
    check_code();configure()
    for ds in ('mosi','mosei'):
        out=ROOT/'runs'/ds/'fold_0'
        for n,h in read(out/'selection_lock.json')['artifacts'].items():assert sha(out/n)==h
    for ds in ('mosi','mosei'):
        out=ROOT/'runs'/ds/'fold_0';dest=ROOT/'evaluation'/ds;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'done.json').exists():
            done=read(dest/'done.json');assert sha(dest/'predictions_test.npz')==done['predictions_sha'];continue
        source=cv_data.SOURCE/ds/'test.npz'
        with np.load(source,allow_pickle=False) as f:d=select_inputs({k:np.array(f[k]) for k in cv_data.KEYS+('ids','groups')})
        for k in ('ids','groups'):d[k]=cv_data._normalize(d[k])
        hd=transform_test(d,out/'prep_main',dest/'features');dims=read(out/'prep_main/data_complete.json')['dims'];model=fresh_core(dims);model.load_state_dict(loadck(out/'main/C5/best.pt')['model'])
        ps=predict(model,hd,'C5');npz(dest/'locked_predictions.npz',dict(ids=d['ids'],**{'pred_'+c:ps[c] for c in CS}))
        with np.load(source,allow_pickle=False) as f:y=np.array(f['y'])
        npz(dest/'predictions_test.npz',dict(ids=d['ids'],y=y,**{'pred_'+c:ps[c] for c in CS}))
        save(dest/'done.json',dict(predictions_sha=sha(dest/'predictions_test.npz'),test_sha=sha(source),selection_lock_sha=sha(out/'selection_lock.json'),historical_test_exposure=True,independent_confirmation=False))
        del model,hd,d;gc.collect()
        for p in (dest/'features').glob('*.npy'):p.unlink()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate']);p.add_argument('--dataset',choices=['mosi','mosei']);a=p.parse_args()
    train(a.dataset) if a.action=='train' else evaluate()
