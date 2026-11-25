import numpy as np
from sklearn.metrics import accuracy_score, f1_score
def metrics(y,p):
    y=np.asarray(y).reshape(-1);p=np.asarray(p).reshape(-1)
    assert len(y)==len(p) and np.isfinite(p).all() and len(y)>0
    nz=y!=0
    r={'n':len(y),'n_nonzero':int(nz.sum()),'MAE':float(np.abs(y-p).mean()),
       'Corr':float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else None,
       'Acc7':float(100*np.mean(np.rint(np.clip(y,-3,3))==np.rint(np.clip(p,-3,3)))),
       'Acc5':float(100*np.mean(np.rint(np.clip(y,-2,2))==np.rint(np.clip(p,-2,2))))}
    for tag,mask,cmp in [('nonzero',nz,lambda a:a>0),('with_zero',np.ones(len(y),dtype=bool),lambda a:a>=0)]:
        if mask.any():
            r['Acc2_'+tag]=float(accuracy_score(cmp(y[mask]),cmp(p[mask]))*100)
            r['F1_'+tag]=float(f1_score(cmp(y[mask]),cmp(p[mask]),average='weighted',zero_division=0)*100)
        else:r['Acc2_'+tag]=r['F1_'+tag]=None
    if nz.any():
        r['MAE_nonzero']=float(np.abs(y[nz]-p[nz]).mean())
        r['Corr_nonzero']=float(np.corrcoef(y[nz],p[nz])[0,1]) if np.std(y[nz])>0 and np.std(p[nz])>0 else None
    return r
