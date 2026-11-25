"""Protocol adaptation of public RoME. Original architecture and train loss retained.
Validation-only epoch selection, clip metrics, no fixed seed, resumable once/config.
Never invoke the upstream main: it selects epochs using test metrics.
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'text_decision_v3_20261120'))
from metrics_common import scores, atomic_json, digest
sys.path.insert(0,str(HERE/'RoME'/'Rome'))
from dataloader_cmumosi import CMUMOSIDataset
from utils import build_model, generate_inputs
from train_Rome import train_or_eval_model
from loss import MaskedMSELoss

CONDITIONS=['T','TA','TV','A','V','AV','TAV']
FEATURES=['wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT']

def state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def load(path): return torch.load(path,map_location='cpu',weights_only=False)
def rng():
    return dict(py=random.getstate(),np=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore(r):
    random.setstate(r['py']);np.random.set_state(r['np']);torch.set_rng_state(r['torch'])
    if r['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])

def source_hashes():
    paths=list((HERE/'RoME').rglob('*.py'))+[Path(__file__),HERE/'text_decision_v3_20261120/metrics_common.py']
    return {p.relative_to(HERE).as_posix():digest(p) for p in sorted(paths)}

def audit_data(folder,out):
    # Full hash once; later runs require unchanged path/size/mtime inventory.
    files=sorted(p for p in folder.rglob('*') if p.is_file())
    inventory=[dict(path=p.relative_to(folder).as_posix(),size=p.stat().st_size,mtime=p.stat().st_mtime_ns) for p in files]
    dest=out/'data_audit.json'
    if dest.exists():
        audit=json.loads(dest.read_text());assert inventory==audit['inventory'],'Input data changed; do not mix runs'
    else:
        audit=dict(inventory=inventory,sha256={r['path']:digest(folder/r['path']) for r in inventory})
        atomic_json(dest,audit)
    return digest(dest)

def datasets(a):
    name='CMU'+a.dataset.upper();folder=a.data_root/name
    label=folder/(name+'_features_raw_2way.pkl')
    if not label.exists():raise FileNotFoundError(f'Place official labels/features under {folder}')
    for f in FEATURES:assert (folder/'features'/f).is_dir(),f'Missing official feature directory: {folder}/features/{f}'
    d=CMUMOSIDataset(str(label),*[str(folder/'features'/f) for f in FEATURES])
    vids=[d.trainVids,d.valVids,d.testVids]
    ids=[{str(uid) for v in vs for uid in d.videoIDs[v]} for vs in vids]
    for i in range(3):
        assert len(ids[i])==sum(len(d.videoIDs[v]) for v in vids[i]),'Duplicate clip ID'
        for j in range(i):assert not set(vids[i])&set(vids[j]) and not ids[i]&ids[j],'Split overlap'
    loaders={}
    for split,vs in zip(['train','valid','test'],vids):
        ix=[d.vids.index(v) for v in sorted(vs)]
        assert ix,'Empty split'
        if a.smoke:ix=ix[:2]
        loaders[split]=DataLoader(Subset(d,ix),batch_size=2 if a.smoke else 32,shuffle=split=='train',collate_fn=d.collate_fn,num_workers=0)
    return d,loaders,folder

@torch.no_grad()
def evaluate(model,loader,d,device,condition):
    model.eval();ids=[];ys=[];pred=[];groups=[]
    for b in loader:
        feats=[x.to(device) for x in b[:6]];q,u,y=[x.to(device) for x in b[6:9]]
        masks=[torch.full((*x.shape[:2],1),int(c in condition),device=device,dtype=x.dtype) for x,c in zip(feats,['A','T','V']*2)]
        masked=[x*m for x,m in zip(feats,masks)]
        f=generate_inputs(*masked,q)[0];m=generate_inputs(*masks,q)[0]
        out=model(f,m,u,first_stage=False)[1].squeeze(-1)
        assert out.shape==y.shape==u.shape
        for i,v in enumerate(b[-1]):
            n=int(u[i].sum());assert n==len(d.videoIDs[v])
            ids.extend(map(str,d.videoIDs[v]));groups.extend([str(v)]*n)
            ys.extend(y[i,:n].cpu().numpy().tolist());pred.extend(out[i,:n].cpu().numpy().tolist())
    ys=np.asarray(ys);pred=np.asarray(pred)
    assert np.isfinite(pred).all() and np.isfinite(ys).all()
    return dict(ids=np.asarray(ids),groups=np.asarray(groups),y=ys,pred=pred),scores(ys,pred)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','test','smoke'])
    p.add_argument('--dataset',choices=['mosi','mosei'],required=True);p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/rome_data'))
    p.add_argument('--output-root',type=Path,default=HERE/'rome_results');p.add_argument('--device',default='cuda');a=p.parse_args()
    a.smoke=a.stage=='smoke';torch.set_num_threads(4)
    if a.device.startswith('cuda'):assert torch.cuda.is_available(),'GPU unavailable'
    out=a.output_root/('smoke' if a.smoke else 'main')/f'{a.dataset}_{a.condition}';out.mkdir(parents=True,exist_ok=True)
    # Prevent duplicate jobs. Windows smoke runs are single-process tests only.
    lock=(out/'run.lock').open('a')
    if os.name=='posix':
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    epochs,stage1=(100,50) if a.dataset=='mosi' else (60,30)
    if a.smoke:epochs,stage1=2,1
    args=SimpleNamespace(device=torch.device(a.device),hidden=16 if a.smoke else 256,depth=1 if a.smoke else 4,num_heads=2,
        n_classes=1,n_speakers=1,drop_rate=.5,attn_drop_rate=0.,no_cuda=a.device=='cpu',
        use_cit=1,use_mrr=1,use_rmc=1,use_cmdae=0 if a.condition=='A' else 1,
        use_huber=False,lambda_ccc=0.,lambda_m=0. if a.condition=='A' else .1,grad_clip=1.,eval_pool='token',test_condition='atv' if a.condition=='TAV' else a.condition.lower())
    d,ls,folder=datasets(a)
    auditdir=a.output_root/('smoke' if a.smoke else 'main')/a.dataset;auditdir.mkdir(exist_ok=True)
    data_hash=audit_data(folder,auditdir)
    config=dict(dataset=a.dataset,condition=a.condition,epochs=epochs,stage1=stage1,smoke=a.smoke,
        source=source_hashes(),data_audit_sha256=data_hash,args={k:str(v) if k=='device' else v for k,v in vars(args).items()},
        selection='stage2 valid Acc2_nonzero highest; tie MAE lowest; tie earliest',stage1_selection='training accuracy; retained original adapter rule',feature_protocol='official RoME features; not V1 enhanced inputs',
        manual_seed=False,stage1_all_modalities=True,test_used_for_selection=False)
    cp=out/'protocol.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Protocol/source/data changed; inspect before resume'
    else:atomic_json(cp,config)
    model=build_model(args,*d.get_featDim()).to(args.device)
    selection=out/'selection_lock.json'
    if a.stage=='test':
        assert selection.exists(),'Train and freeze validation selection first'
        sel=json.loads(selection.read_text());assert sel['protocol_sha256']==digest(cp)
        assert sel['checkpoint_sha256']==digest(out/'best.pt')
        done=out/'test_done.json'
        if done.exists():
            old=json.loads(done.read_text());assert old['selection_sha256']==digest(selection)
            assert old['predictions_sha256']==digest(out/'predictions_test.npz');print('Test already complete; unchanged');return
        model.load_state_dict(load(out/'best.pt')['model'])
        check,_=evaluate(model,ls['valid'],d,args.device,a.condition)
        with np.load(out/'predictions_valid.npz') as ref:
            np.testing.assert_array_equal(check['ids'],ref['ids']);np.testing.assert_allclose(check['pred'],ref['pred'],atol=2e-5,rtol=0)
        pack,metric=evaluate(model,ls['test'],d,args.device,a.condition)
        np.savez_compressed(out/'predictions_test.npz',**pack);atomic_json(out/'metrics_test.json',metric)
        atomic_json(done,dict(selection_sha256=digest(selection),predictions_sha256=digest(out/'predictions_test.npz')));return
    if selection.exists():print('Validation selection already frozen; skip training');return
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5);reg=MaskedMSELoss()
    epoch0=0;best_train=-1.;best_stage1=None;best_rank=(-1.,-1e9);best_epoch=None
    if (out/'last.pt').exists():
        ck=load(out/'last.pt');assert ck['protocol_sha256']==digest(cp)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);epoch0=ck['epoch']+1
        best_train=ck['best_train'];best_stage1=ck['best_stage1'];best_rank=ck['best_rank'];best_epoch=ck['best_epoch'];restore(ck['rng'])
    for epoch in range(epoch0,epochs):
        if epoch==stage1:
            current=model.state_dict()
            for k in current:
                if 'Transformer' in k and any(c in k for c in 'atv'):current[k]=best_stage1[k]
            model.load_state_dict(current)
        train_m,_,_=train_or_eval_model(args,model,reg,None,ls['train'],optimizer=opt,train=True,first_stage=epoch<stage1)
        assert np.isfinite(train_m['total_loss']),'Nonfinite training loss'
        if epoch<stage1:
            if train_m['acc']>best_train:best_train=train_m['acc'];best_stage1=state(model)
            valid=None
        else:
            pack,valid=evaluate(model,ls['valid'],d,args.device,a.condition);m=valid['metrics'];rank=(m['Acc2_nonzero'],-m['MAE'])
            if rank>best_rank:
                best_rank=rank;best_epoch=epoch+1
                save(out/'best.pt',dict(model=state(model),epoch=best_epoch))
                np.savez_compressed(out/'predictions_valid.npz',**pack);atomic_json(out/'metrics_valid.json',valid)
        save(out/'last.pt',dict(model=state(model),optimizer=opt.state_dict(),epoch=epoch,best_train=best_train,
            best_stage1=best_stage1,best_rank=best_rank,best_epoch=best_epoch,rng=rng(),protocol_sha256=digest(cp)))
        status=dict(stage='train',epoch=epoch+1,epochs=epochs,stage_number=1 if epoch<stage1 else 2,train=train_m,valid=valid,time=time.time())
        atomic_json(out/'status.json',status);print(json.dumps(status),flush=True)
    assert best_epoch is not None
    atomic_json(selection,dict(best_epoch=best_epoch,checkpoint_sha256=digest(out/'best.pt'),protocol_sha256=digest(cp),test_used_for_selection=False))
    atomic_json(out/'status.json',dict(stage='validation_complete',best_epoch=best_epoch))

if __name__=='__main__':main()
