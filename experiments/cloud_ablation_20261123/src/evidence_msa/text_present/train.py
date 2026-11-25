"""Fresh C5 and frozen D24 construction inside one outer fold.

No historical weights/predictions are loaded. Outer labels are excluded from the
trainer; only the final aggregation program computes outer metrics.
"""
import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from .protocol import *
from .model import TextPresent,objective
from .raw_adapters import RawEvidenceAdapter
from .text_anchor import TextAnchor
from .cell_text_anchor import CellTextAnchor,cell_objective
from .factorized_anchor import FactorAnchor,objective as factor_loss
from .calibration import cells,signature
from . import data as cv_data

KEYS=('T','A','AS','V','VS','timeline','Q')
def batch(d,ix):return {k:torch.tensor(np.array(d[k][ix]),device=DEVICE,dtype=torch.float32) for k in KEYS}

@torch.no_grad()
def predict(model,d,kind):
    model.eval();out={c:[] for c in CS} if kind in ('C5','D5') else {'pred':[],'prob':[]}
    for i in range(0,len(d['T']),64):
        if kind in ('C5','D5'):
            b=batch(d,slice(i,i+64))
            for c in CS:out[c].append(model(b,c)['pred'].cpu().numpy().astype('float64'))
        else:
            o=model(torch.tensor(np.array(d['T'][i:i+64]),device=DEVICE))
            out['pred'].append(o['pred'].cpu().numpy().astype('float64'))
            if 'logits' in o:out['prob'].append(o['logits'].softmax(-1).cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items() if len(v)}

def blend_anchor(base,a):return {c:(base[c]+.25*(a-base['T'])).clip(-3,3) for c in CS}
def blend_cell(base,o):
    sp=np.sort(o['prob'],-1);active=(o['prob'].argmax(-1)!=3)&((sp[:,-1]-sp[:,-2])>=.4)
    return {c:(base[c]+.5*active*(o['pred']-base['T'])).clip(-3,3) for c in CS}
def blend_factor(base,q):
    out={}
    for c in CS:
        _,lo,hi=cells(base[c]);out[c]=np.clip(base[c]+.25*(q-base['T']),lo,hi)
        assert np.array_equal(signature(out[c]),signature(base[c]))
    return out

def fit_component(model,kind,tr,va,folder,epochs,callback=None,fixed=False):
    """Epoch0 candidate for residual/anchor components, earliest tie. Atomic resume."""
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():
        done=read(folder/'done.json');assert sha(folder/'best.pt')==done['sha256']
        ck=loadck(folder/'best.pt');model.load_state_dict(ck['model']);model.eval();return model,ck['epoch']
    lr=.0001 if kind=='D5' else .0003;bs=32 if kind=='D5' else 64
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=lr,weight_decay=.001)
    contract=dict(kind=kind,epochs=epochs,fixed_epochs=fixed,batch=bs,lr=lr,policy=ensure_policy(),
        train_ids_sha=__import__('hashlib').sha256('\n'.join(map(str,tr['ids'])).encode()).hexdigest(),
        valid_ids_sha=None if fixed else __import__('hashlib').sha256('\n'.join(map(str,va['ids'])).encode()).hexdigest())
    if (folder/'contract.json').exists():assert read(folder/'contract.json')==contract
    else:save(folder/'contract.json',contract)
    start=0;best_state=None;best_rank=-1e300;best_epoch=0;history=[]
    if (folder/'last.pt').exists():
        ck=loadck(folder/'last.pt');assert ck['contract']==contract
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);move_opt(opt);restore_rng(ck['rng'])
        start=ck['epoch'];best_state=ck['best'];best_rank=ck['best_rank'];best_epoch=ck['best_epoch'];history=ck['history']
    elif not fixed and kind!='C5':
        p=callback(predict(model,va,kind));best_rank=utility(p,va['y']);best_state=state_cpu(model)
    for epoch in range(start,epochs):
        began=time.time();model.train();total=0.
        for ix in torch.randperm(len(tr['T'])).split(bs):
            ix=ix.numpy();y=torch.tensor(np.array(tr['y'][ix]),device=DEVICE,dtype=torch.float32);opt.zero_grad(set_to_none=True)
            if kind in ('C5','D5'):
                b=batch(tr,ix);value=0.
                for c in CS:
                    loss=objective(model(b,c),y,CONFIG)/3
                    assert torch.isfinite(loss);loss.backward();value+=float(loss.detach())
            else:
                o=model(torch.tensor(np.array(tr['T'][ix]),device=DEVICE))
                loss=cell_objective(o,y) if kind=='cell' else (factor_loss(o,y) if kind=='factor' else objective(o,y,CONFIG))
                assert torch.isfinite(loss);loss.backward();value=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(parameters,5.);assert torch.isfinite(norm)
            opt.step();total+=value*len(ix)
        record=dict(epoch=epoch+1,loss=total/len(tr['T']),seconds=time.time()-began)
        if fixed:best_state=state_cpu(model);best_epoch=epoch+1
        else:
            p=callback(predict(model,va,kind));rank=utility(p,va['y']);record['inner_utility']=rank
            if rank>best_rank+1e-12:best_state=state_cpu(model);best_rank=rank;best_epoch=epoch+1
        history.append(record)
        tsave(folder/'last.pt',dict(contract=contract,epoch=epoch+1,model=state_cpu(model),optimizer=opt.state_dict(),rng=rng_state(),best=best_state,best_rank=best_rank,best_epoch=best_epoch,history=history))
        save(folder/'history.json',history)
        status(folder,'training',component=kind,epoch=epoch+1,epochs=epochs,seconds=record['seconds'],selected_epoch=best_epoch)
    if best_state is None:best_state=state_cpu(model) # zero fixed epochs is valid for zero residuals
    model.load_state_dict(best_state);model.eval()
    tsave(folder/'best.pt',dict(model=best_state,epoch=best_epoch,contract=contract))
    save(folder/'done.json',dict(sha256=sha(folder/'best.pt'),selected_epoch=best_epoch,inner_utility=None if fixed else best_rank))
    local_unlink(folder/'last.pt');return model,best_epoch

def fresh_core(dims):
    from ..settings import BACKBONE
    from .simple_fusion import SimpleFusion
    return (SimpleFusion(dims,CONFIG,128) if BACKBONE=='simple' else TextPresent(dims,CONFIG,128)).to(DEVICE)
def d5_from(core,dims):return RawEvidenceAdapter(dict(model=state_cpu(core),dims=dims,config=CONFIG,hidden=128),'D5_bilinear').to(DEVICE)

@torch.no_grad()
def composite(models,d):
    m,a,c=models
    for x in models:x.eval()
    chunks=[]
    for i in range(0,len(d['T']),64):
        b=batch(d,slice(i,i+64));o={k:m(b,k) for k in CS};base={k:o[k]['pred'].cpu().numpy().astype('float64') for k in CS}
        aq=a(b['T'])['pred'].cpu().numpy().astype('float64');co=c(b['T']);q=co['pred'].cpu().numpy().astype('float64');pr=co['logits'].softmax(-1).cpu().numpy()
        sp=np.sort(pr,-1);margin=sp[:,-1]-sp[:,-2];active=(pr.argmax(-1)!=3)&(margin>=.4)
        old=blend_anchor(base,aq);p={k:(old[k]+.5*active*(q-old['T'])).clip(-3,3) for k in CS};z={}
        for k in CS:
            pc=o[k]['probabilities'].cpu().numpy();pt=o['T']['probabilities'].cpu().numpy()
            small=np.column_stack((p['T'],p[k],base['T'],base[k],aq,q,margin,base[k]-base['T'],q-old['T']))
            z['f_'+k]=np.concatenate((small,pc,pt,pr,pc-pt),-1).astype('float32');z['pred_'+k]=p[k]
        chunks.append(z)
    result={k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]}
    for k in ('ids','groups','y'):
        if k in d:result[k]=np.array(d[k])
    return result

def train_composite(dims,data,folder,budgets=None):
    tr=data['train'];va=data.get('valid');fixed=budgets is not None
    budgets=(dict(C5=1,D5=1,anchor=1,cell=1) if SMOKE else dict(C5=40,D5=20,anchor=40,cell=40)) if budgets is None else budgets
    core,e0=fit_component(fresh_core(dims),'C5',tr,va,folder/'C5',budgets['C5'],lambda x:x,fixed)
    model,e1=fit_component(d5_from(core,dims),'D5',tr,va,folder/'D5',budgets['D5'],lambda x:x,fixed)
    # Persist C5 separately as the comparator; D5 has its own frozen copy.
    del core;gc.collect();torch.cuda.empty_cache()
    base=None if fixed else predict(model,va,'D5')
    a,e2=fit_component(TextAnchor(dims['T']).to(DEVICE),'anchor',tr,va,folder/'anchor',budgets['anchor'],None if fixed else lambda x:blend_anchor(base,x['pred']),fixed)
    b2=None if fixed else blend_anchor(base,predict(a,va,'anchor')['pred'])
    c,e3=fit_component(CellTextAnchor(dims['T']).to(DEVICE),'cell',tr,va,folder/'cell',budgets['cell'],None if fixed else lambda x:blend_cell(b2,x),fixed)
    return (model,a,c),dict(C5=e0,D5=e1,anchor=e2,cell=e3)

def load_composite(dims,folder):
    core=fresh_core(dims);core.load_state_dict(loadck(folder/'C5/best.pt')['model'])
    m=d5_from(core,dims);m.load_state_dict(loadck(folder/'D5/best.pt')['model']);del core
    a=TextAnchor(dims['T']).to(DEVICE);a.load_state_dict(loadck(folder/'anchor/best.pt')['model'])
    c=CellTextAnchor(dims['T']).to(DEVICE);c.load_state_dict(loadck(folder/'cell/best.pt')['model'])
    for x in (m,a,c):x.eval()
    return m,a,c

def npz(path,z):
    tmp=Path(str(path)+'.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**z)
    tmp.replace(path)

def run(ds,fold):
    from .supervisor import check_code
    check_code();configure();ensure_policy();out=ROOT/'runs'/ds/('fold_'+str(fold));out.mkdir(parents=True,exist_ok=True)
    with lock(out/'ours.lock'):
        if (out/'ours_done.json').exists():
            for n,h in read(out/'ours_done.json')['files'].items():assert sha(out/n)==h
            return
        manifest=read(ROOT/'manifests'/ds/('fold_'+str(fold)+'.json'))
        fit=np.array(manifest['train_indices']);valid=np.array(manifest['valid_indices']);held=np.array(manifest['held_indices'])
        status(out,'preparing_fresh_C5',dataset=ds,fold=fold)
        data,dims=cv_data.prepare(ds,fit,{'valid':valid},out/'prep_main')
        models,budgets=train_composite(dims,data,out/'main')
        zv=composite(models,data['valid']);npz(out/'inner_full.npz',zv);save(out/'budgets.json',budgets)
        del models;gc.collect();torch.cuda.empty_cache()
        tr=data['train'];n=len(fit);oof={k:np.empty((n,37),np.float32) if k.startswith('f_') else np.empty(n,np.float64) for k in zv if k.startswith(('f_','pred_'))}
        for k in ('ids','groups','y'):oof[k]=np.array(tr[k])
        counts=np.zeros(n,int);foldpred=[];audits=[]
        for j,(subfit,subheld) in enumerate(GroupKFold(4).split(np.zeros(n),groups=tr['groups'])):
            dest=out/('sub_'+str(j));dest.mkdir(exist_ok=True)
            sp=dict(train_indices=fit[subfit].tolist(),oof_indices=fit[subheld].tolist(),valid_indices=valid.tolist())
            if (dest/'split.json').exists():assert read(dest/'split.json')==sp
            else:save(dest/'split.json',sp)
            if (dest/'inference_done.json').exists():
                for name,h in read(dest/'inference_done.json')['files'].items():assert sha(dest/name)==h
                with np.load(dest/'oof.npz') as f:zo={k:f[k] for k in f.files}
                with np.load(dest/'valid.npz') as f:zs={k:f[k] for k in f.files}
            else:
                status(out,'crossfit_training',dataset=ds,fold=fold,subfold=j)
                sd,dd=cv_data.prepare(ds,fit[subfit],{'oof':fit[subheld],'valid':valid},dest/'prep')
                sm,_=train_composite(dd,sd,dest/'models',budgets)
                zo=composite(sm,sd['oof']);zs=composite(sm,sd['valid']);npz(dest/'oof.npz',zo);npz(dest/'valid.npz',zs)
                save(dest/'inference_done.json',dict(files={name:sha(dest/name) for name in ('oof.npz','valid.npz')}))
                del sm,sd;gc.collect();torch.cuda.empty_cache()
            assert np.array_equal(zo['ids'],tr['ids'][subheld]);assert np.array_equal(zs['ids'],data['valid']['ids'])
            for k in oof:
                if k.startswith(('pred_','f_')):oof[k][subheld]=zo[k]
            counts[subheld]+=1;foldpred.append({c:zs['pred_'+c] for c in CS})
            audits.append(dict(subfold=j,fit_groups=len(set(tr['groups'][subfit])),oof_groups=len(set(tr['groups'][subheld])),overlap=0))
        assert np.all(counts==1);oof['count']=counts;npz(out/'inner_oof.npz',oof);save(out/'crossfit_audit.json',audits)
        foldmean={c:np.mean([p[c] for p in foldpred],axis=0) for c in CS}
        from .calibration import fit_postprocessors,apply_postprocessors
        def report_post(record):
            record=dict(record);stage=record.pop('stage','postprocessing');record.pop('time',None)
            status(out,stage,dataset=ds,fold=fold,**record)
        state=fit_postprocessors(oof,zv,foldmean,out/'post',utility,report_post)
        base=apply_postprocessors(zv,foldmean,state)
        factor,_=fit_component(FactorAnchor(dims['T']).to(DEVICE),'factor',tr,data['valid'],out/'factor',1 if SMOKE else 40,lambda x:blend_factor(base,x['pred']))
        locked={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('best.pt'))}
        locked.update({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*shared.joblib'))})
        locked.update({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('scalers.joblib'))})
        locked['post/complete.pt']=sha(out/'post/complete.pt')
        save(out/'selection_lock.json',dict(artifacts=locked,manifest_sha=sha(ROOT/'manifests'/ds/('fold_'+str(fold)+'.json')),policy_sha=ensure_policy(),held_used=False))
        # Fresh polarity classifier fitted only to original training labels.
        from .runner import fit_polarity
        fit_polarity(ds,fit,out)
        # Paired ablation: the no-ERU row is exactly this full run's frozen C5.
        from .runner import repair,raw
        import joblib
        core=fresh_core(dims);core.load_state_dict(loadck(out/'main/C5/best.pt')['model'])
        pc=predict(core,data['valid'],'C5');q=blend_factor(base,predict(factor,data['valid'],'factor')['pred'])
        clfs=joblib.load(out/'polarity.joblib');source=cv_data.pool(ds)
        final={c:repair(q[c],clfs[c].predict_proba(raw(source,valid,c))[:,1]) for c in CS}
        npz(out/'validation_predictions.npz',dict(ids=data['valid']['ids'],y=data['valid']['y'],**{'c5_'+c:pc[c] for c in CS},**{'final_'+c:final[c] for c in CS}))
        save(out/'parameters.json',dict(c5_parameters=sum(p.numel() for p in core.parameters()),eru_enabled=True,crossfit_members=4))
        del core,source
        locked.update({'polarity.joblib':sha(out/'polarity.joblib')})
        save(out/'selection_lock.json',dict(artifacts=locked,manifest_sha=sha(ROOT/'manifests'/ds/('fold_'+str(fold)+'.json')),policy_sha=ensure_policy(),held_used=False))
        save(out/'ours_done.json',dict(files={'selection_lock.json':sha(out/'selection_lock.json')},test_opened=False))
        del tr,data,factor;gc.collect();torch.cuda.empty_cache()
        for prep in [out/'prep_main']+[out/('sub_'+str(j))/'prep' for j in range(4)]:cv_data.cleanup_arrays(prep)
        status(out,'training_complete',dataset=ds,test_opened=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mosi','mosei'],required=True);p.add_argument('--fold',type=int,default=0);a=p.parse_args()
    run(a.dataset,a.fold)
