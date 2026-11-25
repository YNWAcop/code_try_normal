"""Official TMDC architecture/loss; validation-selected protocol adaptation.
No upstream test-selected entrypoint, no fixed seed, no benchmark test in profiling.
"""
import argparse, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor/TMDC'))
from dataloader_cmumosi import CMUMOSIDataset
from utils_TMDC import build_model,generate_inputs
from loss import MaskedMSELoss
from metrics_common import atomic_json,digest,scores
GIB=1024**3
CONDITIONS=['T','TA','TV','A','V','AV','TAV']
FEATURES=['wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT']

def snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def rng():
    return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore(r):
    random.setstate(r['py']);np.random.set_state(r['np']);torch.set_rng_state(r['cpu'])
    if r['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])
def code_hash():
    files=[Path(__file__),BASE/'metrics_common.py']+sorted((BASE/'vendor/TMDC').rglob('*.py'))
    return {p.relative_to(BASE).as_posix():digest(p) for p in files}
def lock(path):
    f=path.open('a')
    if os.name=='posix':
        import fcntl
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f
def config(a):
    mosi=a.dataset=='mosi'
    first={'TAV':35,'T':35,'TA':35,'TV':35,'A':82,'V':107,'AV':107} if mosi else {'TAV':45,'T':45,'TA':45,'TV':45,'A':45,'V':38,'AV':45}
    return dict(epochs=2 if a.stage=='smoke' else (300 if mosi else 200),
                stage1=1 if a.stage=='smoke' else first[a.condition],
                batch=2 if a.stage=='smoke' else (a.batch_size or (32 if mosi else 64)),
                hidden=16 if a.stage=='smoke' else 256,depth=1 if a.stage=='smoke' else 4,
                dropout=.3 if mosi else .6,condition={'TAV':'atv','TA':'at','AV':'av','TV':'tv'}.get(a.condition,a.condition.lower()))
def dataset(a,c):
    name='CMU'+a.dataset.upper();folder=a.data_root/name
    label=folder/(name+'_features_raw_2way.pkl')
    if not label.is_file():raise FileNotFoundError(label)
    d=CMUMOSIDataset(str(label),*[str(folder/'features'/f) for f in FEATURES])
    split_vids=[d.trainVids,d.valVids,d.testVids];seen_v=set();seen_u=set();indices={}
    for split,vs in zip(['train','valid','test'],split_vids):
        assert vs and len(vs)==len(set(vs)) and not seen_v.intersection(vs),'Duplicate/overlapping videos'
        ids=[str(u) for v in vs for u in d.videoIDs[v]]
        assert len(ids)==len(set(ids)) and not seen_u.intersection(ids),'Duplicate/overlapping clips'
        seen_v.update(vs);seen_u.update(ids)
        indices[split]=[d.vids.index(v) for v in sorted(vs)]
        if a.stage=='smoke':indices[split]=indices[split][:2]
    # test labels are structurally checked; the test loader is never iterated in training/profile.
    ls={s:DataLoader(Subset(d,ix),batch_size=c['batch'],shuffle=s=='train',num_workers=0,collate_fn=d.collate_fn) for s,ix in indices.items()}
    return d,ls,folder,indices
def audit(folder,out):
    files=sorted(p for p in folder.rglob('*') if p.is_file())
    inv=[(p.relative_to(folder).as_posix(),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    inv=json.loads(json.dumps(inv));path=out/'data_audit.json'
    if path.exists():assert json.loads(path.read_text())['inventory']==inv,'Data changed; inspect before resume'
    else:atomic_json(path,dict(inventory=inv,sha256={p.relative_to(folder).as_posix():digest(p) for p in files}))
    return digest(path)
def forward(model,b,device,condition,first):
    features=[x.to(device) for x in b[:6]];q,u,y=[x.to(device) for x in b[6:9]]
    masks=[torch.full((*x.shape[:2],1),int(first or m in condition),dtype=x.dtype,device=device) for x,m in zip(features,['A','T','V']*2)]
    f=generate_inputs(*[x*m for x,m in zip(features,masks)],q)[0]
    mask=generate_inputs(*masks,q)[0]
    return model(f,mask,u,first_stage=first),u,y
def train_batch(model,b,device,condition,first,opt):
    model.train();opt.zero_grad(set_to_none=True)
    out,u,y=forward(model,b,device,condition,first);reg=MaskedMSELoss()
    if first:
        # Sum the same three modality objectives before one backward pass.
        loss=sum(reg(x,y,u) for x in out[2]+out[3]+out[4])+sum(out[5])*.01
    else:loss=reg(out[1],y,u)
    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
    loss.backward();opt.step();return float(loss.detach())
@torch.no_grad()
def evaluate(model,loader,d,device,condition,first=False):
    model.eval();ids=[];groups=[];truth=[];pred=[[],[],[]] if first else [[]]
    for b in loader:
        out,u,y=forward(model,b,device,condition,first)
        values=out[2] if first else [out[1]]
        for i,v in enumerate(b[-1]):
            n=int(u[i].sum());assert n==len(d.videoIDs[v])
            ids.extend(map(str,d.videoIDs[v]));groups.extend([str(v)]*n)
            truth.extend(y[i,:n].cpu().tolist())
            for j,value in enumerate(values):pred[j].extend(value[i,:n,0].cpu().tolist())
    if first:return [scores(truth,p)['metrics'] for p in pred]
    pack=dict(ids=np.asarray(ids),groups=np.asarray(groups),y=np.asarray(truth),pred=np.asarray(pred[0]))
    return pack,scores(truth,pred[0])
def profile(model,opt,d,indices,c,a):
    # Pack the longest real TRAIN videos together; includes both stages and VALID forward.
    train_ix=sorted(indices['train'],key=lambda i:len(d.videoIDs[d.vids[i]]),reverse=True)[:c['batch']]
    valid_ix=sorted(indices['valid'],key=lambda i:len(d.videoIDs[d.vids[i]]),reverse=True)[:c['batch']]
    batch=d.collate_fn([d[i] for i in train_ix]);vb=d.collate_fn([d[i] for i in valid_ix])
    started=time.time()
    for first in (True,False):
        train_batch(model,batch,a.device,a.condition,first,opt)
        model.eval()
        with torch.no_grad():forward(model,vb,a.device,a.condition,first)
    if a.device=='cuda':torch.cuda.synchronize()
    return dict(stage='profile_complete',dataset=a.dataset,condition=a.condition,
                batch=c['batch'],config=c,source=code_hash(),gpu_cap_gib=a.gpu_cap_gib,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB if a.device=='cuda' else None,
                train_max_length=max(len(d.videoIDs[d.vids[i]]) for i in train_ix),
                seconds=time.time()-started,test_used=False,time=time.time())
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','profile','train','test'])
    p.add_argument('--dataset',choices=['mosi','mosei'],required=True);p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/rome_data'))
    p.add_argument('--output-root',type=Path,default=BASE/'results');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=0);p.add_argument('--gpu-cap-gib',type=float,default=8.)
    a=p.parse_args();torch.set_num_threads(4)
    if a.device=='cuda':
        assert torch.cuda.is_available(),'CUDA unavailable'
        total=torch.cuda.get_device_properties(0).total_memory/GIB
        torch.cuda.set_per_process_memory_fraction(min(a.gpu_cap_gib/total,.9),0)
        torch.cuda.reset_peak_memory_stats()
    c=config(a);out=a.output_root/('smoke' if a.stage=='smoke' else 'main')/f'{a.dataset}_{a.condition}'
    if a.stage=='profile':out=a.output_root/'profiles'/f'{a.dataset}_{a.condition}'
    out.mkdir(parents=True,exist_ok=True);held=lock(out/'run.lock')
    try:
        atomic_json(out/'status.json',dict(stage='loading_data',time=time.time(),pid=os.getpid()))
        d,ls,folder,indices=dataset(a,c)
        args=SimpleNamespace(device=torch.device(a.device),hidden=c['hidden'],depth=c['depth'],num_heads=2,
                             n_classes=1,drop_rate=c['dropout'],attn_drop_rate=0.,no_cuda=a.device=='cpu',test_condition=c['condition'])
        model=build_model(args,*d.get_featDim()).to(a.device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
        if a.stage=='profile':
            report=profile(model,opt,d,indices,c,a);atomic_json(out/'profile.json',report);atomic_json(out/'status.json',report);print(json.dumps(report),flush=True);return
        auditdir=a.output_root/('smoke' if a.stage=='smoke' else 'main')/a.dataset;auditdir.mkdir(exist_ok=True)
        protocol=dict(method='TMDC',dataset=a.dataset,condition=a.condition,config=c,data_sha256=audit(folder,auditdir),
                      source=code_hash(),seed_policy='one run; no manual seed',selection='validation nonzero Acc2; tie MAE; earliest',
                      stage1_selection='per-intra-attention branch validation Acc2; real tensor snapshots',
                      features='official TMDC/RoME feature names; distinct from V1 enhanced inputs',test_used_for_selection=False,
                      adaptations=['validation split retained','snapshot bug fixed','sum equivalent stage1 losses in one backward'],gpu_cap_gib=a.gpu_cap_gib)
        cp=out/'protocol.json'
        if cp.exists():assert json.loads(cp.read_text())==protocol,'Protocol/source/data changed; refusing resume'
        else:atomic_json(cp,protocol)
        selection=out/'selection_lock.json'
        if a.stage=='test':
            sel=json.loads(selection.read_text());assert sel['protocol_sha256']==digest(cp) and sel['checkpoint_sha256']==digest(out/'best.pt')
            if (out/'test_done.json').exists():
                done=json.loads((out/'test_done.json').read_text());assert done['selection_sha256']==digest(selection) and done['predictions_sha256']==digest(out/'predictions_test.npz')
                atomic_json(out/'status.json',dict(stage='test_complete',already_done=True));return
            ck=load(out/'best.pt');model.load_state_dict(ck['model']);restore(ck['eval_rng'])
            pack,_=evaluate(model,ls['valid'],d,a.device,a.condition)
            with np.load(out/'predictions_valid.npz') as ref:
                np.testing.assert_array_equal(pack['ids'],ref['ids']);np.testing.assert_allclose(pack['pred'],ref['pred'],atol=2e-5,rtol=1e-5)
            pack,metrics=evaluate(model,ls['test'],d,a.device,a.condition)
            np.savez_compressed(out/'predictions_test.npz',**pack);atomic_json(out/'metrics_test.json',metrics)
            atomic_json(out/'test_done.json',dict(selection_sha256=digest(selection),predictions_sha256=digest(out/'predictions_test.npz')))
            atomic_json(out/'status.json',dict(stage='test_complete',time=time.time()));return
        if selection.exists():
            sel=json.loads(selection.read_text());assert sel['checkpoint_sha256']==digest(out/'best.pt') and sel['protocol_sha256']==digest(cp)
            atomic_json(out/'status.json',dict(stage='validation_complete',already_done=True));return
        epoch0=0;best_rank=(-1.,-float('inf'));best_epoch=None;branch_rank=[-1.]*3;branch_state=[None]*3
        if (out/'last.pt').exists():
            ck=load(out/'last.pt');assert ck['protocol_sha256']==digest(cp)
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore(ck['rng'])
            epoch0=ck['epoch'];best_rank=ck['best_rank'];best_epoch=ck['best_epoch'];branch_rank=ck['branch_rank'];branch_state=ck['branch_state']
        for epoch in range(epoch0,c['epochs']):
            if epoch==c['stage1']:
                current=model.state_dict()
                for saved in branch_state:
                    assert saved;current.update(saved)
                model.load_state_dict(current)
            first=epoch<c['stage1'];started=time.time();losses=[]
            for b in ls['train']:losses.append(train_batch(model,b,a.device,a.condition,first,opt))
            if first:
                validation=evaluate(model,ls['valid'],d,a.device,a.condition,True)
                for i,prefix in enumerate(['intra_attn_as.','intra_attn_ts.','intra_attn_vs.']):
                    value=validation[i]['Acc2_nonzero']
                    if value is not None and value>branch_rank[i]:
                        branch_rank[i]=value;branch_state[i]={k:v for k,v in snapshot(model).items() if k.startswith(prefix)}
            else:
                eval_rng=rng();pack,validation=evaluate(model,ls['valid'],d,a.device,a.condition)
                m=validation['metrics'];rank=(m['Acc2_nonzero'],-m['MAE'])
                if rank>best_rank:
                    best_rank=rank;best_epoch=epoch+1;save(out/'best.pt',dict(model=snapshot(model),epoch=best_epoch,eval_rng=eval_rng))
                    np.savez_compressed(out/'predictions_valid.npz',**pack);atomic_json(out/'metrics_valid.json',validation)
            save(out/'last.pt',dict(model=snapshot(model),optimizer=opt.state_dict(),rng=rng(),epoch=epoch+1,
                 best_rank=best_rank,best_epoch=best_epoch,branch_rank=branch_rank,branch_state=branch_state,protocol_sha256=digest(cp)))
            status=dict(stage='train',dataset=a.dataset,condition=a.condition,epoch=epoch+1,epochs=c['epochs'],stage_number=1 if first else 2,
                        loss=float(np.mean(losses)),seconds=time.time()-started,valid=validation,time=time.time(),
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB if a.device=='cuda' else None)
            atomic_json(out/'status.json',status);print(json.dumps(status),flush=True)
        assert best_epoch is not None
        atomic_json(selection,dict(best_epoch=best_epoch,checkpoint_sha256=digest(out/'best.pt'),protocol_sha256=digest(cp),test_used_for_selection=False))
        atomic_json(out/'status.json',dict(stage='validation_complete',best_epoch=best_epoch,time=time.time()))
    except BaseException as e:
        atomic_json(out/'status.json',dict(stage='failed',error=repr(e),time=time.time()));raise

if __name__=='__main__':main()
