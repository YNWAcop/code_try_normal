"""Fixed original-split fresh training and one post-lock test evaluation."""
import argparse,gc,sys,time
import numpy as np
import torch,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .protocol import *
from . import data as cv_data

def raw(d,ix,c):
    keys={'T':('T',),'TA':('T','A_native','A_words'),'TV':('T','V_native','V_expression','V_lip','V_words','V_quality')}[c]
    x=np.concatenate([np.asarray(d[k][ix],dtype=np.float32) for k in keys],axis=1)
    assert np.isfinite(x).all()
    return x

def fit_polarity(ds,fit,out):
    d=cv_data.pool(ds);mask=d['y'][fit]!=0;models={}
    with threadpool_limits(limits=4):
        for c in CS:
            status(out,'polarity_fit',dataset=ds,condition=c)
            m=make_pipeline(StandardScaler(),LogisticRegression(C=.001,solver='lbfgs',tol=1e-6,max_iter=1000))
            m.fit(raw(d,fit,c)[mask],(d['y'][fit][mask]>0).astype(int))
            assert m[-1].n_iter_[0]<1000
            models[c]=m
    joblib.dump(models,out/'polarity.joblib')

def repair(p,q):
    active=(np.abs(p)<.1)&(np.abs(q-.5)>=.1)&((p>0)!=(q>.5))
    result=p.copy();result[active]=np.where(q[active]>.5,1e-4,-1e-4)
    for clip in (2,3):assert np.array_equal(np.rint(p.clip(-clip,clip)),np.rint(result.clip(-clip,clip)))
    return result

def prepare():
    ensure_policy()
    for ds in ('mosi','mosei'):
        d=cv_data.pool(ds)
        with np.load(cv_data.SOURCE/ds/'train.npz') as f:n=len(f['y'])
        fit=np.arange(n);valid=np.arange(n,len(d['y']))
        assert not set(d['groups'][fit])&set(d['groups'][valid])
        m=dict(dataset=ds,train_indices=fit.tolist(),valid_indices=valid.tolist(),held_indices=[],original_test_opened=False)
        p=ROOT/'manifests'/ds/'fold_0.json'
        if p.exists():assert read(p)==m
        else:save(p,m)
        print(ds,'train',n,'valid',len(valid),flush=True)

def transform_test(d,prep,out):
    meta=read(prep/'data_complete.json')
    assert not set(map(str,d['groups']))&set(meta['fit_groups'])
    for n,h in meta['transforms'].items():assert sha(prep/n)==h
    ix=np.arange(len(d['ids']));sc=joblib.load(prep/'scalers.joblib')
    cv_data._output(d,ix,sc,out,False)
    for mod in ('A','V'):cv_data._apply_shared(d,ix,joblib.load(prep/(mod+'_shared.joblib')),out/(mod+'S.npy'))
    return cv_data.arrays(out,False)

def evaluate():
    from .supervisor import check_code
    from .train import load_composite,composite,predict,fresh_core,FactorAnchor,blend_factor,npz
    from .calibration import apply_postprocessors,load as loadpost
    check_code();configure()
    for ds in ('mosi','mosei'):
        out=ROOT/'runs'/ds/'fold_0';lockdata=read(out/'selection_lock.json')
        for n,h in lockdata['artifacts'].items():assert sha(out/n)==h
    reports=[]
    for ds in ('mosi','mosei'):
        dest=ROOT/'evaluation'/ds;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'done.json').exists():
            assert sha(dest/'results.json')==read(dest/'done.json')['results_sha'];reports.extend(read(dest/'results.json'));continue
        out=ROOT/'runs'/ds/'fold_0'
        status(ROOT,'evaluating_locked_test',dataset=ds,test_opened=True)
        source=cv_data.SOURCE/ds/'test.npz'
        # Features only until all fixed predictions have been produced.
        with np.load(source,allow_pickle=False) as f:d={k:np.array(f[k]) for k in cv_data.KEYS+('ids','groups')}
        d['ids']=cv_data._normalize(d['ids']);dims=read(out/'prep_main/data_complete.json')['dims']
        hd=transform_test(d,out/'prep_main',dest/'main_features')
        core=fresh_core(dims);core.load_state_dict(loadck(out/'main/C5/best.pt')['model']);c5=predict(core,hd,'C5');del core
        models=load_composite(dims,out/'main');z=composite(models,hd);del models;gc.collect();torch.cuda.empty_cache()
        subpred=[]
        for j in range(4):
            folder=out/f'sub_{j}';temp=dest/f'sub_{j}_features'
            h=transform_test(d,folder/'prep',temp);m=load_composite(dims,folder/'models');zz=composite(m,h)
            subpred.append({c:zz['pred_'+c] for c in CS});del m,h;gc.collect();torch.cuda.empty_cache()
            for p in temp.glob('*.npy'):p.unlink()
        mean={c:np.mean([v[c] for v in subpred],axis=0) for c in CS}
        post=loadpost(out/'post/complete.pt');b=apply_postprocessors(z,mean,post)
        factor=FactorAnchor(dims['T']).cuda();factor.load_state_dict(loadck(out/'factor/best.pt')['model'])
        p24=blend_factor(b,predict(factor,hd,'factor')['pred']);clfs=joblib.load(out/'polarity.joblib')
        final={c:repair(p24[c],clfs[c].predict_proba(raw(d,slice(None),c))[:,1]) for c in CS}
        npz(dest/'locked_predictions.npz',dict(ids=d['ids'],**{'pred_'+c:final[c] for c in CS}))
        with np.load(source,allow_pickle=False) as f:y=np.array(f['y'])
        result=[]
        for method,ps in [('C5_fresh',c5),('Ours_D24_R3_fresh',final)]:
            for c in CS:result.append(dict(dataset=ds,method=method,condition=c,metrics=scores(y,ps[c])['metrics']))
        npz(dest/'predictions_test.npz',dict(ids=d['ids'],y=y,**{'pred_'+c:final[c] for c in CS}))
        save(dest/'results.json',result);save(dest/'done.json',dict(results_sha=sha(dest/'results.json'),test_sha=sha(source),selection_lock_sha=sha(out/'selection_lock.json'),historical_test_exposure=True,independent_confirmation=False))
        reports.extend(result);del hd,d,factor;gc.collect();torch.cuda.empty_cache()
        for p in (dest/'main_features').glob('*.npy'):p.unlink()
    save(ROOT/'results.json',reports)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','evaluate']);v=a.parse_args()
    prepare() if v.action=='prepare' else evaluate()
