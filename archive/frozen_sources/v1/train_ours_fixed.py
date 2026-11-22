"""Portable, fresh-task-head consolidation of the existing research framework.

Frozen author/ASR/VSR features -> text anchor + visible A/V correction.
Original-video OOF nuisance reliability, visual dual stream and temporal pooling.
This is not a bit-for-bit reproduction of historic ensembles. No fixed RNG seed.
"""
from pathlib import Path
import argparse,json,time,copy,hashlib,platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import joblib
from metrics import metrics
ROOT=Path(__file__).resolve().parents[1]
MASKS=('T','A','V','TA','TV','AV','TAV')
BITS=torch.tensor([[m in mask for m in 'TAV'] for mask in MASKS],dtype=torch.bool)
KEYS=['T','A_native','A_words','V_native','V_lip','V_words','V_expression','V_quality','V_timeline']
def save(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(ds,s):
    with np.load(ROOT/'data/ready'/ds/(s+'.npz')) as z:return {k:z[k] for k in z.files}
def scaling(train,other):
    scalers={}
    for k in KEYS:
        sc=StandardScaler().fit(train[k].reshape(-1,train[k].shape[-1]));scalers[k]=sc
        for d in [train]+other:d[k]=np.clip(sc.transform(d[k].reshape(-1,d[k].shape[-1])).reshape(d[k].shape),-8,8).astype('float32')
    return scalers
def apply_scalers(d,scalers):
    for k,sc in scalers.items():d[k]=np.clip(sc.transform(d[k].reshape(-1,d[k].shape[-1])).reshape(d[k].shape),-8,8).astype('float32')
def loss(r,y):
    # Same scalar is used for all reported metrics; continuous and ordinal supervision.
    ordinal=F.binary_cross_entropy_with_logits((r[:,None]-torch.arange(-2.5,3,device=r.device))/.6,(y[:,None]>torch.arange(-2.5,3,device=r.device)).float())
    return F.smooth_l1_loss(r,y,beta=.55)+.3*ordinal
def mlp(d,h=128,out=None):
    return nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(.2),nn.Linear(h,h if out is None else out))
class Text(nn.Module):
    def __init__(self,dim):super().__init__();self.net=mlp(dim,out=1)
    def forward(self,x):return 3*torch.tanh(self.net(x).squeeze(-1)/3)
def text_fit(x,y,xv,yv,args,epochs=None):
    model=Text(x.shape[1]).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    best=float('inf');state=None;chosen=0
    for ep in range(epochs or args.epochs):
        model.train()
        for ix in torch.randperm(len(y)).split(args.batch_size):
            ix=ix.numpy();opt.zero_grad();r=model(torch.as_tensor(x[ix],device=args.device));l=loss(r,torch.as_tensor(y[ix],device=args.device));l.backward();opt.step()
        if epochs is None:
            pred=text_predict(model,xv,args);score=np.abs(pred-yv).mean()
            if score<best:best=score;state=copy.deepcopy(model.state_dict());chosen=ep+1
    if state is not None:model.load_state_dict(state)
    return model,chosen if epochs is None else epochs
def text_predict(model,x,args):
    model.eval()
    with torch.no_grad():return np.concatenate([model(torch.as_tensor(x[i:i+256],device=args.device)).cpu().numpy() for i in range(0,len(x),256)])
def reconstruction(t,x,groups):
    oof=np.empty_like(x)
    for tr,va in GroupKFold(4).split(t,groups=groups):
        oof[va]=Ridge(alpha=100.).fit(t[tr],x[tr]).predict(t[va])
    rho=np.clip(1-np.mean((oof-x)**2,axis=0)/np.maximum(x.var(0),1e-6),0,1)
    return dict(model=Ridge(alpha=100.).fit(t,x),mean=x.mean(0),rho=rho),oof
def arrays(d,variant):
    a=np.column_stack([d['A_native']]+([] if variant in ('base_features','no_asr') else [d['A_words']]))
    app=np.column_stack([d['V_native'],d['V_expression']])
    lex=np.column_stack([d['V_lip'],d['V_words']])
    if variant in ('base_features','no_vsr'):lex=np.zeros_like(lex)
    timeline=d['V_timeline'].copy()
    if variant in ('base_features','no_vsr'):timeline[:]=0
    return dict(T=d['T'],A=a,V=np.column_stack([app,lex]),timeline=timeline,quality=d['V_quality'])
class Evidence(nn.Module):
    def __init__(self,dims,appdim):
        super().__init__();h=64;self.appdim=appdim
        self.t=mlp(dims['T'],h);self.a=mlp(dims['A'],h)
        self.app=mlp(appdim,h);self.lex=mlp(dims['V']-appdim,h)
        self.time=nn.Sequential(nn.Conv1d(768,h,3,padding=1),nn.GELU());self.query=nn.Linear(h,h)
        self.visual_gate=nn.Sequential(nn.Linear(2*h+5,h),nn.Sigmoid())
        self.delta=mlp(3*h+4,h,1);self.no_text=mlp(2*h+2,h,1)
    def visual(self,v,timeline,q):
        app=self.app(v[:,:self.appdim]);lex=self.lex(v[:,self.appdim:]);states=self.time(timeline.transpose(1,2)).transpose(1,2)
        att=(self.query(lex)[:,None,:]*states).sum(-1).softmax(1)
        lex=lex+(states*att[:,:,None]).sum(1);w=self.visual_gate(torch.cat([app,lex,q],1))
        return app*(1-w)+lex*w
    def forward(self,b,bits):
        bits=bits.to(b['T'].device);t,a,v=[bits[:,j:j+1].float() for j in range(3)]
        # Mask raw inputs too: unavailable sources cannot affect any computation.
        tx=self.t(b['T']*t)*t
        ax=self.a(torch.where(t.bool(),b['A_res'],b['A'])*a)*a
        vx=self.visual(torch.where(t.bool(),b['V_res'],b['V'])*v,b['timeline']*v[:,:,None],b['quality']*v)*v
        anchor=b['anchor']*t[:,0]
        correction=self.delta(torch.cat([tx,ax,vx,anchor[:,None],bits.float()],1)).squeeze(-1)
        present=anchor+torch.tanh(correction)*(a[:,0]+v[:,0]).clamp_max(1)
        missing=3*torch.tanh(self.no_text(torch.cat([ax,vx,bits[:,1:].float()],1)).squeeze(-1)/3)
        return torch.where(t[:,0].bool(),present,missing).clamp(-3,3)
def tensors(d,ix,device):return {k:torch.as_tensor(v[ix],device=device,dtype=torch.float32) for k,v in d.items()}
def predict(model,x,args):
    model.eval();result={}
    with torch.no_grad():
        for j,m in enumerate(MASKS):
            result[m]=np.concatenate([model(tensors(x,slice(i,i+256),args.device),BITS[j].expand(min(256,len(x['T'])-i),-1)).cpu().numpy() for i in range(0,len(x['T']),256)])
    return result
def visibility_check(model,x,args):
    model.eval();b=tensors(x,slice(0,8),args.device);n=len(b['T'])
    with torch.no_grad():
        for j,m in enumerate(MASKS):
            alt={k:v.clone() for k,v in b.items()}
            if 'T' not in m:
                for k in ('T','anchor','A_res','V_res'):alt[k][:]=99
            if 'A' not in m:
                for k in ('A','A_res'):alt[k][:]=99
            if 'V' not in m:
                for k in ('V','V_res','timeline','quality'):alt[k][:]=99
            torch.testing.assert_close(model(b,BITS[j].expand(n,-1)),model(alt,BITS[j].expand(n,-1)),atol=1e-6,rtol=1e-6)
def train(args):
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    tr,va=load(args.dataset,'train'),load(args.dataset,'valid')
    if args.smoke:
        chosen=np.unique(tr['groups'])[:8];ix=np.concatenate([np.flatnonzero(tr['groups']==g)[:12] for g in chosen])
        tr={k:v[ix] for k,v in tr.items()};va={k:v[:32] for k,v in va.items()};args.epochs=1
    # Text OOF folds fit their own scaling; no historic trained scores are consumed.
    text_oof=np.empty(len(tr['y']),dtype='float32')
    full_sc=StandardScaler().fit(tr['T']);full_t=full_sc.transform(tr['T']).clip(-8,8).astype('float32')
    val_t=full_sc.transform(va['T']).clip(-8,8).astype('float32')
    text,epochs=text_fit(full_t,tr['y'],val_t,va['y'],args)
    for fold,(fit,held) in enumerate(GroupKFold(4).split(tr['T'],groups=tr['groups'])):
        sc=StandardScaler().fit(tr['T'][fit]);x=sc.transform(tr['T'][fit]).clip(-8,8).astype('float32');xh=sc.transform(tr['T'][held]).clip(-8,8).astype('float32')
        model,_=text_fit(x,tr['y'][fit],None,None,args,epochs=epochs)
        text_oof[held]=text_predict(model,xh,args);print('text_oof_fold',fold,'complete',flush=True)
    scalers=scaling(tr,[va]);joblib.dump(scalers,out/'scalers.joblib')
    x,z=arrays(tr,args.variant),arrays(va,args.variant)
    x['anchor']=text_oof;z['anchor']=text_predict(text,va['T'],args)
    nuisances={}
    for k in ('A','V'):
        if args.variant in ('reliable','no_asr','no_vsr'):
            remaining=np.empty_like(x[k])
            for fit,held in GroupKFold(4).split(x['T'],groups=tr['groups']):
                fitb,_=reconstruction(x['T'][fit],x[k][fit],tr['groups'][fit])
                remaining[held]=x[k][held]-(fitb['model'].predict(x['T'][held])-fitb['mean'])*fitb['rho']
            b,_=reconstruction(x['T'],x[k],tr['groups']);nuisances[k]=b
            x[k+'_res']=remaining;z[k+'_res']=z[k]-(b['model'].predict(z['T'])-b['mean'])*b['rho']
        else:x[k+'_res']=x[k].copy();z[k+'_res']=z[k].copy()
    joblib.dump(nuisances,out/'nuisances.joblib')
    dims={k:x[k].shape[-1] for k in ('T','A','V')};appdim=tr['V_native'].shape[-1]+tr['V_expression'].shape[-1]
    model=Evidence(dims,appdim).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    best=float('inf');history=[]
    for ep in range(args.epochs):
        model.train();total=0.
        for ix in torch.randperm(len(tr['y'])).split(args.batch_size):
            ix=ix.numpy();bits=BITS[torch.randint(len(MASKS),(len(ix),))]
            opt.zero_grad();pred=model(tensors(x,ix,args.device),bits);l=loss(pred,torch.as_tensor(tr['y'][ix],device=args.device));l.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=l.item()*len(ix)
        pred=predict(model,z,args);score=float(np.mean([np.abs(pred[m]-va['y']).mean() for m in MASKS[:6]]))
        history.append({'epoch':ep+1,'train_loss':total/len(tr['y']),'valid_condition_MAE':score});print(history[-1],flush=True)
        save(out/'history.json',history)
        if score<best:
            best=score;torch.save({'model':model.state_dict(),'text':text.state_dict(),'dims':dims,'appdim':appdim},out/'best.pt');chosen_epoch=ep+1
        save(out/'heartbeat.json',{'time':time.time(),'epoch':ep+1})
    ck=torch.load(out/'best.pt',map_location=args.device,weights_only=False);model.load_state_dict(ck['model']);visibility_check(model,z,args)
    protocol={'dataset':args.dataset,'variant':args.variant,'epochs':args.epochs,'batch_size':args.batch_size,'lr':args.lr,'seed_policy':'no manual seed, no repeats or ensemble','selection':'minimum validation MAE for this fixed retained-modality condition','selected_epoch':chosen_epoch,'text_selected_epoch':epochs,'valid_score':best,'smoke':args.smoke,'status':'validation_locked_no_test_evaluated','runtime':{'torch':torch.__version__,'python':platform.python_version()},'scope':'fresh task heads over frozen upstream author/ASR/VSR features; portable consolidation, not historical ensemble reproduction'}
    save(out/'validation_lock.json',protocol)
    np.savez_compressed(out/'validation_predictions.npz',ids=va['ids'],y=va['y'],**predict(model,z,args))
    print('VALIDATION LOCKED; evaluate separately only after configurations are final.',flush=True)
def evaluate(args):
    lock=json.loads((args.out/'validation_lock.json').read_text());assert not lock['smoke'],'Smoke cannot produce main-table results'
    if (args.out/'test_metrics.json').exists():raise RuntimeError('Test already evaluated for this run; preserve the existing record')
    args.dataset=lock['dataset'];args.variant=lock['variant'];d=load(args.dataset,'test');apply_scalers(d,joblib.load(args.out/'scalers.joblib'))
    ck=torch.load(args.out/'best.pt',map_location=args.device,weights_only=False)
    text=Text(d['T'].shape[-1]).to(args.device);text.load_state_dict(ck['text']);x=arrays(d,args.variant);x['anchor']=text_predict(text,d['T'],args)
    nuisance=joblib.load(args.out/'nuisances.joblib')
    for k in ('A','V'):
        if k in nuisance:
            b=nuisance[k];x[k+'_res']=x[k]-(b['model'].predict(x['T'])-b['mean'])*b['rho']
        else:x[k+'_res']=x[k].copy()
    model=Evidence(ck['dims'],ck['appdim']).to(args.device);model.load_state_dict(ck['model']);visibility_check(model,x,args);pred=predict(model,x,args)
    np.savez_compressed(args.out/'test_predictions.npz',ids=d['ids'],y=d['y'],**pred)
    save(args.out/'test_metrics.json',{'protocol':'whole_modality_fixed_masks','dataset':args.dataset,'method':'cloud_'+args.variant,'metrics':{m:metrics(d['y'],p) for m,p in pred.items()},'checkpoint_sha256':hashlib.sha256((args.out/'best.pt').read_bytes()).hexdigest()})

if __name__=='__main__':
    import os
    from metrics_common import scores, digest, atomic_json
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',choices=['mosi','mosei'],required=True)
    p.add_argument('--condition',choices=['A','V','T','AV','TA','TV'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=3e-4)
    a=p.parse_args();a.variant='reliable';ROOT=a.root.resolve()
    MASKS=(a.condition,);BITS=torch.tensor([[m in a.condition for m in 'TAV']],dtype=torch.bool)
    torch.set_num_threads(4)
    train(a)
    lock=json.loads((a.out/'validation_lock.json').read_text())
    lock.update(method='OURS_reliable',protocol='whole',condition=a.condition,checkpoint=str(a.out/'best.pt'),checkpoint_sha256=digest(a.out/'best.pt'))
    atomic_json(a.out/'validation_lock.json',lock)
    with np.load(a.out/'validation_predictions.npz',allow_pickle=False) as z:
        ids,y,pred=z['ids'],z['y'],z[a.condition]
    report=scores(y,pred)
    assert abs(report['metrics']['MAE']-lock['valid_score'])<1e-4
    np.savez_compressed(a.out/'predictions_full_valid.npz',ids=ids,y=y,pred=pred)
    report.update(schema_version='priority-whole-v1',status='complete',job=a.out.name,method='OURS_reliable',dataset=a.dataset,protocol='whole',condition=a.condition,split='valid',selected_epoch=lock['selected_epoch'],checkpoint_sha256=lock['checkpoint_sha256'],validation_lock_sha256=digest(a.out/'validation_lock.json'),smoke=a.smoke,training_protocol='fixed condition; frozen upstream author/ASR/VSR features')
    atomic_json(a.out/'metrics_full_valid.json',report)
