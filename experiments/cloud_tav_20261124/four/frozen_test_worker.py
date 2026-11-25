"""Frozen-checkpoint inference only. One process per job/split, no optimization."""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
from common import ROOT, WHOLE, HAND, METHODS, read, digest, atomic_json, scores, result_dir

class Config(dict):
    def __getattr__(self,k):
        try:return self[k]
        except KeyError:raise AttributeError(k)
    def __setattr__(self,k,v):self[k]=v

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,file);obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj);return obj

def load_state(model,weight,device):
    import torch
    state=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True);del state
    return model.to(device).eval()

def collect(loader,forward,device):
    import torch
    ps=[];ys=[];ids=[]
    with torch.inference_mode():
        for b in loader:
            p=forward(b).detach().cpu().numpy().reshape(-1)
            y=b['labels']['M'].numpy().reshape(-1)
            if len(p)!=len(y):raise RuntimeError('Prediction/label batch size differs')
            ps.append(p);ys.append(y);ids.extend(map(str,b['id']))
    return np.asarray(ids),np.concatenate(ys),np.concatenate(ps)

def infer_ours(src,ds,condition,split,device):
    import torch,joblib
    mod=module('frozen_reliable',WHOLE/'train_ours_fixed.py');mod.ROOT=ROOT
    mod.MASKS=(condition,);mod.BITS=torch.tensor([[m in condition for m in 'TAV']],dtype=torch.bool)
    a=argparse.Namespace(device=str(device),variant='reliable')
    d=mod.load(ds,split);mod.apply_scalers(d,joblib.load(src/'scalers.joblib'))
    ck=torch.load(src/'best.pt',map_location='cpu',weights_only=True,mmap=True)
    text=mod.Text(d['T'].shape[-1]).to(device);text.load_state_dict(ck['text'],strict=True)
    x=mod.arrays(d,'reliable');x['anchor']=mod.text_predict(text,d['T'],a)
    nuisance=joblib.load(src/'nuisances.joblib')
    for k in ('A','V'):
        b=nuisance[k];x[k+'_res']=x[k]-(b['model'].predict(x['T'])-b['mean'])*b['rho']
    model=mod.Evidence(ck['dims'],ck['appdim']).to(device);model.load_state_dict(ck['model'],strict=True)
    model.eval();mod.visibility_check(model,x,a)
    return d['ids'].astype(str),d['y'].reshape(-1),mod.predict(model,x,a)[condition]

def infer_four(src,method,ds,condition,split,device):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from mmsa_slim_pickle import load_fields
    from mmsa_corruption import CorruptedDataset
    fields=('text_bert','audio','vision','audio_lengths','vision_lengths','id','regression_labels')
    blob=load_fields(ROOT/f'data/mmsa/{ds}/unaligned_50.pkl',fields,splits=(split,))
    if method in ('MISA','ALMT'):
        adapter=module('native_misa_almt',WHOLE/'adapter_misa_almt.py')
        args=Config(read(src/'run_manifest.json'));args.device=device;args.pretrained=str(ROOT/'models/bert-base-uncased')
        base=adapter.InputDataset(blob[split]);del blob
        dsobj=CorruptedDataset(base,WHOLE,ds,split,'whole',condition,0.)
        cls=__import__(f'mmsa.models.singleTask.{method}',fromlist=[method])
        model=adapter.ModelWrapper(getattr(cls,method)(args),args)
        model=load_state(model,src/'best.pt',device)
        def forward(b):
            if method=='MISA':
                model.Model.visible_audio_lengths=b['audio_lengths'].cpu().long().clamp(min=1)
                model.Model.visible_vision_lengths=b['vision_lengths'].cpu().long().clamp(min=1)
            pred=model(b['text'].to(device),b['audio'].to(device),b['vision'].to(device))
            return pred['M'] if method=='MISA' else pred
        batch=int(args.batch_size)
    else:
        vendor=WHOLE/'vendor/emt_dlfr'
        if not (vendor/'PATCH_MANIFEST.json').is_file():raise RuntimeError('Missing trained EMT/TFR vendor snapshot')
        sys.path.insert(0,str(vendor));os.environ['EMT_BERT_PATH']=str(ROOT/'models/bert-base-uncased')
        from models.AMIO import AMIO
        # TFR's training Storage returns False for absent optional fields.
        # Preserve that native behavior (including weight_sim_loss), not a new default.
        if method=='TFR_NET':
            from utils.functions import Storage
            args=Storage(read(src/'config.json')['native_config'])
        else:args=Config(read(src/'config.json')['native_config'])
        args.device=device
        args.model_save_path=str(src/'best.pt');args.protocol_condition=condition
        d=blob[split]
        class Base(Dataset):
            def __init__(self):
                self.text=np.asarray(d['text_bert'],dtype=np.float32);self.audio=d['audio'];self.vision=d['vision']
                self.ids=d['id'];self.labels=np.asarray(d['regression_labels'],dtype=np.float32).reshape(-1,1)
                self.al=np.clip(np.asarray(d['audio_lengths'],dtype=np.int64),1,self.audio.shape[1]);self.vl=np.clip(np.asarray(d['vision_lengths'],dtype=np.int64),1,self.vision.shape[1])
            def __len__(self):return len(self.ids)
            def __getitem__(self,i):
                return dict(text=torch.from_numpy(self.text[i].copy()),audio=torch.from_numpy(np.nan_to_num(np.asarray(self.audio[i],dtype=np.float32),nan=0,posinf=0,neginf=0)),vision=torch.from_numpy(np.nan_to_num(np.asarray(self.vision[i],dtype=np.float32),nan=0,posinf=0,neginf=0)),audio_lengths=int(self.al[i]),vision_lengths=int(self.vl[i]),labels={'M':torch.from_numpy(self.labels[i].copy())},id=str(self.ids[i]),index=i)
        class Visible(Dataset):
            def __init__(self):self.base=Base();self.corrupt=CorruptedDataset(self.base,WHOLE,ds,split,'whole',condition,0.)
            def __len__(self):return len(self.base)
            def __getitem__(self,i):
                visible=self.corrupt[i];item=dict(visible)
                for m,key in [('T','text'),('A','audio'),('V','vision')]:
                    item[key+'_m']=visible[key]
                    if m=='T':available=visible[key][1].clone()
                    else:
                        target=(torch.arange(item[key].shape[0])<item[key+'_lengths']).float()
                        item[key+'_mask']=target;available=target.clone()
                    if m not in condition:available.zero_()
                    item[key+'_missing_mask']=available
                return item
        dsobj=Visible();model=load_state(AMIO(args),src/'best.pt',device);batch=int(args.batch_size)
        def forward(b):
            x={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items() if k!='labels'}
            if method=='EMT_DLFR':return model((x['text'],x['text_m']),(x['audio'],x['audio_m'],x['audio_lengths']),(x['vision'],x['vision_m'],x['vision_lengths']))['pred_m']
            return model((x['text'],x['text_m'],x['text_missing_mask']),(x['audio'],x['audio_m'],x['audio_mask'],x['audio_missing_mask']),(x['vision'],x['vision_m'],x['vision_mask'],x['vision_missing_mask']))[0]
    return collect(DataLoader(dsobj,batch_size=batch,shuffle=False,num_workers=0),forward,device)

def infer_old(src,method,ds,condition,split,device):
    import torch
    from torch.utils.data import DataLoader
    from metrics_common import replay_selection
    from corruption import wrap_loader_dataset
    lock=read(src/'validation_lock.json')
    if method in ('API','TLRA'):
        source=src/'source';sys.path.insert(0,str(source));os.chdir(source)
        args=Config(lock['config']);args.device=device;args.pretrained=str(ROOT/'models/bert-base-uncased');args.featurePath=str(ROOT/f'data/api_aligned/{ds}/aligned_50.pkl')
        from data_loader import MMDataset
        dsobj=MMDataset(args,mode=split)
        if method=='API':
            from trains.singleTask.model.april import APRIL
            model=APRIL(args)
            forward=lambda b:model(b['text'].to(device),b['audio'].to(device),b['vision'].to(device),num_modal=3,labels=None)['output_logit']
        else:
            from trains.singleTask.model.TLRA import TLRA
            from trains.singleTask.TLRA import TLRATrainer
            _,phase=replay_selection(read(src/'validation_history.json'),lock['selection'],args.early_stop,True)
            model=TLRA(args);model.phase2=(phase==2);trainer=TLRATrainer(args)
            def forward(b):
                t,a,v=trainer.loss_fn.apply_keep_mask(b['text'].to(device),b['audio'].to(device),b['vision'].to(device),condition)
                return trainer._safe_forward(model,t,a,v,path='missing',missing_keep=condition,freeze_bank=True)['output_logit']
        batch=int(args.batch_size)
    else:
        source=ROOT/'baselines/adapted'/src.name;sys.path.insert(0,str(source));os.chdir(source)
        cfg=read(src/'effective_config.json')
        if method=='LNLN':
            cfg['dataset']['dataPath']=str(ROOT/f'data/mmsa/{ds}/unaligned_50.pkl');cfg['model']['feature_extractor']['bert_pretrained']=str(ROOT/'models/bert-base-uncased')
            from core.dataset import MMDataset
            from models.lnln import build_model
            dsobj=MMDataset(cfg,mode=split);model=build_model(cfg);batch=int(cfg['base']['batch_size'])
            forward=lambda b:model((None,None,None),(b['vision_m'].to(device),b['audio_m'].to(device),b['text_m'].to(device)))['sentiment_preds']
        else:
            from data.load_data import MMDataset
            from models.AMIO import AMIO
            args=Config(cfg);args.device=device;args.dataPath=str(ROOT/f'data/mmsa/{ds}/unaligned_50.pkl')
            dsobj=MMDataset(args,mode=split);model=AMIO(args);batch=int(args.batch_size)
            def forward(b):
                al=b['audio_lengths'].to(device) if not args.need_data_aligned else 0;vl=b['vision_lengths'].to(device) if not args.need_data_aligned else 0
                return model(b['text'].to(device),(b['audio'].to(device),al),(b['vision'].to(device),vl))['M']
    model=load_state(model,src/'best.pt',device)
    dsobj=wrap_loader_dataset(dsobj,ROOT,ds,split,'whole',condition,0.,lnln=method=='LNLN')
    # Some native datasets call the ID field differently; retain their fixed dataset order.
    ps=[];ys=[]
    with torch.inference_mode():
        for b in DataLoader(dsobj,batch_size=batch,shuffle=False,num_workers=0):
            ps.append(forward(b).detach().cpu().numpy().reshape(-1));ys.append(b['labels']['M'].numpy().reshape(-1))
    return np.asarray(list(map(str,dsobj.ids))),np.concatenate(ys),np.concatenate(ps)

def infer_ebmc(src,ds,condition,split,device):
    import torch
    from torch.utils.data import DataLoader,Subset
    working=ROOT/'baselines/adapted/EBMC';sys.path[:0]=[str(working/'EBMC'),str(working)]
    # Existing feature extraction and teacher files only. No unpacking or training.
    feature=ROOT/'data/ebmc_author'/('CMU'+ds.upper())
    os.environ['EBMC_DATASET_ROOT']=str(feature.parent);os.environ['EBMC_SAVED_ROOT']=str(src/'author_saved')
    from dataloader_cmumosi import CMUMOSIDataset
    from ebmc import build_model
    from utils import generate_inputs
    tag='CMU'+ds.upper();keep=''.join(c for c in 'atv' if c.upper() in condition)
    args=argparse.Namespace(dataset=tag,hidden=256,n_classes=1,n_speakers=2,depth=4,num_heads=2,drop_rate=.55 if ds=='mosi' else .6,attn_drop_rate=0.,no_cuda=device.type!='cuda',device=device,test_condition=keep,lambda_msd=.5,lambda_cce=.1,lambda_emc=.1,lambda_imtd=.1)
    obj=CMUMOSIDataset(str(feature/(tag+'_features_raw_2way.pkl')),str(feature/'features/wav2vec-large-c-UTT'),str(feature/'features/deberta-large-4-UTT'),str(feature/'features/manet_UTT'))
    groups={k:set(getattr(obj,v)) for k,v in [('train','trainVids'),('valid','valVids'),('test','testVids')]}
    assert not any(groups[x]&groups[y] for x,y in [('train','valid'),('train','test'),('valid','test')])
    indices=[i for i,vid in enumerate(obj.vids) if vid in groups[split]]
    model=build_model(args,*obj.get_featDim())
    model.teacher_model=build_model(args,*obj.get_featDim())
    model=load_state(model,src/'stage2_best.pt',device)
    pred=[];truth=[];ids=[]
    with torch.inference_mode():
        for data in DataLoader(Subset(obj,indices),batch_size=64 if ds=='mosi' else 32,shuffle=False,collate_fn=obj.collate_fn):
            ah,th,vh,ag,tg,vg,qmask,umask,label=[x.to(device) for x in data[:9]]
            length,size=ah.shape[:2];am,tm,vm=[torch.full((length,size,1),float(c in keep),device=device) for c in 'atv']
            x=generate_inputs(ah*am,th*tm,vh*vm,ag*am,tg*tm,vg*vm,qmask)[0]
            mask=generate_inputs(am,tm,vm,am,tm,vm,qmask)[0]
            _,pc,*_=model(x,mask,umask,False,None,do_cf=False)
            good=umask.bool();truth.append(label[good].cpu().numpy());pred.append(pc.squeeze(-1)[good].cpu().numpy())
            for vid in data[9]:ids.extend(obj.videoIDs[vid])
    return np.asarray(ids).astype(str),np.concatenate(truth),np.concatenate(pred)

def check_replay(record,ids,y,pred,src):
    with np.load(src/'predictions_full_valid.npz',allow_pickle=False) as z:
        old_ids=list(map(str,z['ids']));new_ids=list(map(str,ids))
        if len(set(new_ids))!=len(new_ids) or set(old_ids)!=set(new_ids):raise RuntimeError('Validation sample IDs changed')
        index={s:i for i,s in enumerate(new_ids)};order=[index[s] for s in old_ids]
        if not np.allclose(y[order],z['y'].reshape(-1),rtol=0,atol=1e-6):raise RuntimeError('Validation labels changed')
        delta=float(np.max(np.abs(pred[order]-z['pred'].reshape(-1))))
        if delta>0.002:raise RuntimeError(f'Validation prediction parity failed: max delta={delta}')
    expected=read(src/'metrics_full_valid.json')['metrics'];checks={}
    for k,v in expected.items():
        a=record['metrics'][k];tol=.002 if k in ('MAE','MSE','RMSE') else .005
        ok=(v is None and a is None) or (v is not None and a is not None and abs(a-v)<=tol)
        checks[k]=dict(reference=v,actual=a,passed=ok,tolerance=tol)
    if not all(c['passed'] for c in checks.values()):raise RuntimeError('Validation metric parity failed: '+str(checks))
    return dict(max_prediction_delta=delta,metrics=checks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--split',choices=['valid','test'],required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    frozen=read(HERE/'frozen_plan.json');spec=next(s for s in frozen['jobs'] if s['job']==a.job)
    src=Path(spec['source']);out=result_dir(a.job);out.mkdir(parents=True,exist_ok=True)
    file=out/('validation_replay.json' if a.split=='valid' else 'metrics_full_test.json')
    if file.exists():raise RuntimeError('Existing result preserved; use queue to resume verified records')
    for name,h in spec['frozen_files'].items():
        if digest(src/name)!=h:raise RuntimeError('Frozen input changed: '+str(src/name))
    if a.split=='test':
        replay=read(out/'validation_replay.json')
        if replay['status']!='complete' or replay['plan_sha256']!=digest(HERE/'frozen_plan.json'):raise RuntimeError('Validation replay not approved by frozen plan')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(WHOLE),str(HAND)]
    import torch
    torch.set_num_threads(2);device=torch.device(a.device)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    m,d,c=spec['method'],spec['dataset'],spec['condition'];start=time.time()
    if m=='OURS_reliable':ids,y,pred=infer_ours(src,d,c,a.split,device)
    elif m=='EBMC':ids,y,pred=infer_ebmc(src,d,c,a.split,device)
    elif m in METHODS[5:9]:ids,y,pred=infer_four(src,m,d,c,a.split,device)
    else:ids,y,pred=infer_old(src,m,d,c,a.split,device)
    ids=np.asarray(ids).astype(str);y=np.asarray(y).reshape(-1);pred=np.asarray(pred).reshape(-1)
    expected={'valid':{'mosi':229,'mosei':1871},'test':{'mosi':686,'mosei':4659}}[a.split][d]
    if len(ids)!=expected or len(set(ids))!=expected or len(y)!=expected:raise RuntimeError(f'Sample count/unique ID mismatch; expected {expected}, got {len(ids)}')
    record=scores(y,pred)
    parity=check_replay(record,ids,y,pred,src) if a.split=='valid' else None
    if a.split=='test':
        with np.load(src/'predictions_full_valid.npz',allow_pickle=False) as z:
            if set(ids)&set(z['ids'].astype(str)):raise RuntimeError('Validation/test sample IDs overlap')
    for name,h in spec['frozen_files'].items():
        if digest(src/name)!=h:raise RuntimeError('Frozen file changed during inference')
    dest=out/('predictions_replay_valid.npz' if a.split=='valid' else 'predictions_full_test.npz')
    with dest.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,ids=ids,y=y,pred=pred)
    dest.with_suffix('.tmp').replace(dest)
    record.update(status='complete',job=a.job,method=m,dataset=d,condition=c,protocol='whole',split=a.split,selected_epoch=spec['selected_epoch'],checkpoint_sha256=spec['checkpoint_sha256'],plan_sha256=digest(HERE/'frozen_plan.json'),predictions_sha256=digest(dest),parity=parity,elapsed_seconds=time.time()-start,selection='Frozen validation-selected checkpoint; no test-driven selection or training')
    atomic_json(file,record);print(json.dumps({'status':'complete','job':a.job,'split':a.split,'n':len(y)}),flush=True)

if __name__=='__main__':main()
