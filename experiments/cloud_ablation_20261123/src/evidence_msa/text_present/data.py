"""Fold-local data preparation. Only original train/valid may enter this module."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import gc, json, hashlib
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from ..settings import RUN_ROOT as ROOT, DATA_ROOT
from ..input_views import select_inputs
SOURCE=DATA_ROOT/'ready'
KEYS=('T','A_native','A_words','V_native','V_lip','V_words','V_expression','V_quality','V_timeline')
MODEL_KEYS=('T','A','V','AS','VS','Q','timeline')
LABEL_KEYS=('ids','groups','y')
_VERIFIED_POOL=set()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def _normalize(a):return np.array([str(x).replace('$_$','_') for x in a])
def pool(ds):
    """Build train+valid cache locally; never read the test split here."""
    if ds not in ('mosi', 'mosei'):
        raise ValueError(ds)
    source = ROOT/'pool'/ds
    meta_path = source/'complete.json'
    if not meta_path.exists():
        source.mkdir(parents=True, exist_ok=True)
        splits = []
        sources = {}
        for split in ('train', 'valid'):
            path = SOURCE/ds/(split+'.npz')
            with np.load(path, allow_pickle=False) as z:
                splits.append(select_inputs({k:np.array(z[k]) for k in KEYS+LABEL_KEYS}))
            sources[split] = {'path':str(path), 'sha256':sha(path)}
        for k in ('ids', 'groups'):
            for d in splits: d[k] = _normalize(d[k])
        assert not set(splits[0]['groups']) & set(splits[1]['groups']), 'Video split overlap'
        for k in KEYS+LABEL_KEYS:
            a = np.concatenate([d[k] for d in splits], axis=0)
            if k not in ('ids', 'groups'): assert np.isfinite(a).all(), k
            with (source/(k+'.partial')).open('wb') as f:
                np.save(f, a, allow_pickle=False)
            (source/(k+'.partial')).replace(source/(k+'.npy'))
        save(meta_path, {'sources':sources, 'original_test_opened':False,
                        'files':{k:sha(source/(k+'.npy')) for k in KEYS+LABEL_KEYS}})
    meta = read(meta_path)
    if ds not in _VERIFIED_POOL:
        for k,h in meta['files'].items(): assert sha(source/(k+'.npy')) == h
        for split,v in meta['sources'].items(): assert sha(SOURCE/ds/(split+'.npz')) == v['sha256']
        _VERIFIED_POOL.add(ds)
    return {k:np.load(source/(k+'.npy'), mmap_mode='r', allow_pickle=False) for k in KEYS+LABEL_KEYS}

def manifests(ds):
    d=pool(ds);n=len(d['y']);counts=np.zeros(n,int);result=[]
    sources=read(ROOT/'pool'/ds/'complete.json')['sources']
    for fold,(outer,held) in enumerate(GroupKFold(5).split(np.zeros(n),groups=d['groups'])):
        fit0,valid0=next(GroupKFold(5).split(np.zeros(len(outer)),groups=d['groups'][outer]))
        splits=dict(train=outer[fit0],valid=outer[valid0],held=held)
        groups={s:set(map(str,d['groups'][ix])) for s,ix in splits.items()}
        assert all(not groups[a]&groups[b] for a,b in [('train','valid'),('train','held'),('valid','held')])
        v=dict(dataset=ds,fold=fold,sources=sources,original_test_opened=False)
        for s,ix in splits.items():
            v[s+'_indices']=ix.tolist();v[s+'_ids']=d['ids'][ix].tolist();v[s+'_groups']=sorted(groups[s])
        p=ROOT/'manifests'/ds/('fold_'+str(fold)+'.json')
        if p.exists():assert read(p)==v,'Immutable split changed'
        else:save(p,v)
        counts[held]+=1;result.append(v)
    assert (counts==1).all();return result
def crossfit_splits(ds,fit_indices):
    d=pool(ds);ix=np.asarray(fit_indices,int)
    return [(ix[a],ix[b]) for a,b in GroupKFold(4).split(np.zeros(len(ix)),groups=d['groups'][ix])]
def _scales(d,ix):
    sc={}
    for k in KEYS:
        s=StandardScaler()
        for start in range(0,len(ix),128):
            a=d[k][ix[start:start+128]];assert np.isfinite(a).all();s.partial_fit(a.reshape(-1,a.shape[-1]))
        sc[k]=s
    return sc
def _output(d,ix,sc,out,include_y=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fields={'T':('T',),'A':('A_native','A_words'),'V':('V_native','V_expression','V_lip','V_words'),'Q':('V_quality',),'timeline':('V_timeline',)}
    for dst,keys in fields.items():
        shape=(len(ix),)+d[keys[0]].shape[1:-1]+(sum(d[k].shape[-1] for k in keys),)
        m=np.lib.format.open_memmap(out/(dst+'.npy'),mode='w+',dtype='float32',shape=shape)
        for start in range(0,len(ix),128):
            parts=[]
            for k in keys:
                a=np.array(d[k][ix[start:start+128]],dtype='float32');parts.append(sc[k].transform(a.reshape(-1,a.shape[-1])).reshape(a.shape).clip(-8,8).astype('float32'))
            m[start:start+128]=np.concatenate(parts,axis=-1)
        m.flush();del m
    for k in LABEL_KEYS:
        if k!='y' or include_y:np.save(out/(k+'.npy'),d[k][ix],allow_pickle=False)
def arrays(p,include_y=True):
    keys=MODEL_KEYS+('ids','groups')+(('y',) if include_y else ())
    return {k:np.load(Path(p)/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in keys}
def _shared(d,fit,mod,keys):
    target=np.column_stack([d[k][fit] for k in keys]).astype('float32');x=np.asarray(d['T'][fit],dtype='float32');oof=np.empty_like(target)
    for a,b in GroupKFold(4).split(x,groups=d['groups'][fit]):
        xs=StandardScaler().fit(x[a]);ys=StandardScaler().fit(target[a]);r=Ridge(alpha=100.).fit(xs.transform(x[a]),ys.transform(target[a]))
        oof[b]=ys.inverse_transform(r.predict(xs.transform(x[b])))
    xs=StandardScaler().fit(x);ys=StandardScaler().fit(target)
    r=Ridge(alpha=100.).fit(xs.transform(x),ys.transform(target))
    gs=StandardScaler().fit(target);t=gs.transform(target).clip(-8,8).astype('float32');oo=gs.transform(oof).clip(-8,8).astype('float32')
    rho=np.clip(1-np.mean((oo-t)**2,axis=0)/np.maximum(t.var(0),1e-6),0,1)
    return dict(xscale=xs,yscale=ys,model=r,global_scale=gs,rho=rho),(oo*rho).astype('float32')
def _apply_shared(d,ix,r,out):
    dim=len(r['rho']);m=np.lib.format.open_memmap(out,mode='w+',dtype='float32',shape=(len(ix),dim))
    for start in range(0,len(ix),128):
        x=r['xscale'].transform(d['T'][ix[start:start+128]]);p=r['yscale'].inverse_transform(r['model'].predict(x))
        m[start:start+128]=(r['global_scale'].transform(p).clip(-8,8)*r['rho']).astype('float32')
    m.flush();del m
def prepare(ds,fit_indices,targets,destination):
    dest=Path(destination).resolve();assert ROOT in dest.parents;dest.mkdir(parents=True,exist_ok=True)
    d=pool(ds);fit=np.asarray(fit_indices,int);targets={k:np.asarray(v,int) for k,v in targets.items()}
    assert len(fit)==len(set(fit));assert len(set(d['groups'][fit]))>=4
    assert 'train' not in targets
    for k,ix in targets.items():
        assert k.isidentifier() and not k.startswith('_');assert not set(d['groups'][fit])&set(d['groups'][ix]),'Fit/target video overlap'
    identity=dict(dataset=ds,fit_indices=fit.tolist(),targets={k:v.tolist() for k,v in targets.items()},sources=read(ROOT/'pool'/ds/'complete.json')['sources'],original_test_opened=False)
    cp=dest/'data_protocol.json'
    if cp.exists():assert read(cp)==identity,'Preparation identity changed'
    else:save(cp,identity)
    dims=dict(T=d['T'].shape[-1],A=d['A_native'].shape[-1]+d['A_words'].shape[-1],V=sum(d[k].shape[-1] for k in ('V_native','V_expression','V_lip','V_words')),app=d['V_native'].shape[-1]+d['V_expression'].shape[-1],timeline=d['V_timeline'].shape[-1],Q=d['V_quality'].shape[-1])
    splits=dict(train=fit,**targets)
    complete=dest/'data_complete.json'
    if complete.exists() and all((dest/s/(k+'.npy')).exists() for s in splits for k in MODEL_KEYS+LABEL_KEYS):
        saved=read(complete);assert saved['protocol_sha256']==sha(cp)
        for n,h in saved['transforms'].items():assert sha(dest/n)==h,'Cached transform changed: '+n
        for n,h in saved.get('arrays',{}).items():assert sha(dest/n)==h,'Cached prepared array changed: '+n
        return {s:arrays(dest/s) for s in splits},dims
    sc=_scales(d,fit);joblib.dump(sc,dest/'scalers.joblib')
    for s,ix in splits.items():_output(d,ix,sc,dest/s)
    for mod,keys in [('A',('A_native','A_words')),('V',('V_native','V_expression','V_lip','V_words'))]:
        r,oo=_shared(d,fit,mod,keys);joblib.dump(r,dest/(mod+'_shared.joblib'));np.save(dest/'train'/(mod+'S.npy'),oo,allow_pickle=False)
        for s,ix in targets.items():_apply_shared(d,ix,r,dest/s/(mod+'S.npy'))
        del r,oo;gc.collect()
    save(complete,dict(protocol_sha256=sha(cp),dims=dims,transforms={n:sha(dest/n) for n in ('scalers.joblib','A_shared.joblib','V_shared.joblib')},arrays={s+'/'+k+'.npy':sha(dest/s/(k+'.npy')) for s in splits for k in MODEL_KEYS+LABEL_KEYS},fit_groups=sorted(set(map(str,d['groups'][fit]))),original_test_opened=False))
    return {s:arrays(dest/s) for s in splits},dims
def transform(ds,indices,prepdir,targetdir,include_y=False):
    prep=Path(prepdir).resolve();out=Path(targetdir).resolve();assert ROOT in prep.parents and ROOT in out.parents
    meta=read(prep/'data_protocol.json');assert meta['dataset']==ds
    complete=read(prep/'data_complete.json')
    for n,h in complete['transforms'].items():assert sha(prep/n)==h
    d=pool(ds);ix=np.asarray(indices,int);assert not set(d['groups'][ix])&set(complete['fit_groups'])
    sc=joblib.load(prep/'scalers.joblib');_output(d,ix,sc,out,include_y)
    for mod in ('A','V'):_apply_shared(d,ix,joblib.load(prep/(mod+'_shared.joblib')),out/(mod+'S.npy'))
    save(out/'transform_audit.json',dict(prep_protocol_sha256=sha(prep/'data_protocol.json'),indices=ix.tolist(),original_test_opened=False,labels_returned=include_y))
    return arrays(out,include_y)
def cleanup_arrays(dest):
    dest=Path(dest).resolve();assert ROOT in dest.parents
    identity=read(dest/'data_protocol.json')
    for split in ['train']+list(identity['targets']):
        p=(dest/split).resolve();assert dest==p.parent
        for k in MODEL_KEYS+LABEL_KEYS:
            f=p/(k+'.npy')
            if f.exists():f.unlink()
if __name__=='__main__':
    import sys
    for ds in sys.argv[1:] or ['mosi','mosei']:
        m=manifests(ds);print(ds,'manifests ready',len(m),flush=True)
