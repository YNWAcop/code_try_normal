"""Evaluate a completed validation-selected checkpoint, never optimize or read a test loader."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from metrics_common import VERSION, atomic_json, digest, scores, replay_selection, expected_jobs

class Config(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
    def __setattr__(self,key,value): self[key]=value

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def wrap_existing(dataset,root,dataset_name,protocol,condition,rate,lnln):
    """Require exactly the validation masks used at training; never create replacements."""
    mask_path=None
    if protocol=='frame':
        import hashlib
        ids=np.asarray([str(x) for x in dataset.ids])
        shapes={m:np.asarray(getattr(dataset,k)).shape for m,k in [('T','text'),('A','audio'),('V','vision')]}
        signature=hashlib.sha256((repr(shapes)+'\n'+'\n'.join(ids)).encode()).hexdigest()[:16]
        mask_path=root/'baseline_protocol/masks'/f'{dataset_name}_valid_frame{rate:.1f}_{signature}.npz'
        if not mask_path.is_file():
            raise FileNotFoundError(f'Original validation mask missing; refusing to generate a new one: {mask_path}')
    from corruption import wrap_loader_dataset
    wrapped=wrap_loader_dataset(dataset,root,dataset_name,'valid',protocol,condition,rate,lnln=lnln)
    return wrapped,mask_path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    root=a.root.resolve();folder=root/'baseline_results'/a.job
    method,dataset,tag=a.job.rsplit('_',2)
    assert method in ('API','TLRA','EBMC','LNLN','Self_MM') and dataset in ('mosi','mosei') and tag=='TAV'
    spec=(a.job,method,dataset,'whole',tag)
    if spec is None: raise ValueError('Only the 130 formal jobs are supported; smoke jobs are excluded')
    _,method,dataset,protocol,tag=spec
    condition=tag if protocol=='whole' else 'TAV'
    rate=float(tag[5:]) if protocol=='frame' else 0.
    lock_path=folder/'validation_lock.json';lock=read(lock_path)
    if lock.get('smoke') or lock.get('status')!='validation_locked_no_test_evaluated':
        raise ValueError('Job is not formally validation-locked')
    weight=folder/('stage2_best.pt' if method=='EBMC' else 'best.pt')
    weight_hash=digest(weight)
    if lock.get('checkpoint_sha256') and lock['checkpoint_sha256']!=weight_hash:
        raise ValueError('Checkpoint hash differs from validation lock')
    initial_lock_hash=digest(lock_path)
    started=time.time()
    progress_path=root/'baseline_reports/full_metrics_progress.json'
    progress=lambda **kw:atomic_json(progress_path,dict(job=a.job,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    progress(stage='loading',device=a.device)
    expected={};mask_path=None;extra={};ids=None
    if method=='EBMC':
        saved=folder/'stage2_validation_predictions.npz'
        with np.load(saved,allow_pickle=False) as z:
            y,pred=z['y'].reshape(-1),z['pred'].reshape(-1)
            ids=z['ids'].astype(str).reshape(-1)
        expected={'MAE':float(lock['valid_score'])}
        extra=dict(prediction_source='Saved selected-stage2 validation predictions',prediction_source_sha256=digest(saved),selected_epoch=lock['selected_epoch'])
        if protocol=='frame':
            mask_path=folder/'valid_frame_masks.npz'
            if not mask_path.is_file(): raise FileNotFoundError(mask_path)
    else:
        if a.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']=''
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import torch
        from torch.utils.data import DataLoader
        torch.set_num_threads(a.threads)
        device=torch.device(a.device)
        sys.path.insert(0,str(a.handoff.resolve()))
        if method in ('API','TLRA'):
            source=folder/'source'
            sys.path.insert(0,str(source));os.chdir(source)
            args=Config(lock['config']);args.device=device
            args.pretrained=str(root/'models/bert-base-uncased')
            args.featurePath=str(root/'data/api_aligned'/dataset/'aligned_50.pkl')
            from data_loader import MMDataset
            ds=MMDataset(args,mode='valid')
            history=read(folder/'validation_history.json')
            epoch,phase=replay_selection(history,lock['selection'],args.early_stop,method=='TLRA')
            selected=history[epoch-1]
            expected={k:float(selected[s]) for k,s in [('MAE','MAE'),('Corr','Corr'),('Acc2_nonzero','Acc_2' if method=='API' else 'acc_2'),('F1_weighted_nonzero','F1_score')] if s in selected}
            extra.update(selected_epoch=epoch,selected_phase=phase,selection_reconstructed_from_history=True)
            if method=='API':
                from trains.singleTask.model.april import APRIL
                model=APRIL(args)
                forward=lambda b:model(b['text'].to(device),b['audio'].to(device),b['vision'].to(device),num_modal=3,labels=None)['output_logit']
            else:
                from trains.singleTask.model.TLRA import TLRA
                from trains.singleTask.TLRA import TLRATrainer
                model=TLRA(args);model.phase2=(phase==2)
                trainer=TLRATrainer(args)
                def forward(b):
                    t,au,v=trainer.loss_fn.apply_keep_mask(b['text'].to(device),b['audio'].to(device),b['vision'].to(device),condition)
                    return trainer._safe_forward(model,t,au,v,path='missing',missing_keep=condition,freeze_bank=True)['output_logit']
            batch_size=int(args.batch_size)
        else:
            source=root/'baselines/adapted'/a.job
            sys.path.insert(0,str(source));os.chdir(source)
            cfg=read(folder/'effective_config.json')
            extra['selected_epoch']=lock['selected_epoch']
            history=read(folder/'history.json')
            selected=next(h['valid'] for h in history if h['epoch']==lock['selected_epoch'])
            expected={k:float(selected[s]) for k,s in [('MAE','MAE'),('Corr','Corr'),('Acc2_nonzero','Non0_acc_2'),('F1_weighted_nonzero','Non0_F1_score')] if s in selected}
            if method=='LNLN':
                cfg['dataset']['dataPath']=str(root/f'data/mmsa/{dataset}/unaligned_50.pkl')
                cfg['model']['feature_extractor']['bert_pretrained']=str(root/'models/bert-base-uncased')
                from core.dataset import MMDataset
                from models.lnln import build_model
                ds=MMDataset(cfg,mode='valid');model=build_model(cfg)
                batch_size=int(cfg['base']['batch_size'])
                forward=lambda b:model((None,None,None),(b['vision_m'].to(device),b['audio_m'].to(device),b['text_m'].to(device)))['sentiment_preds']
            else:
                from data.load_data import MMDataset
                from models.AMIO import AMIO
                args=Config(cfg);args.device=device
                args.dataPath=str(root/f'data/mmsa/{dataset}/unaligned_50.pkl')
                ds=MMDataset(args,mode='valid');model=AMIO(args)
                batch_size=int(args.batch_size)
                def forward(b):
                    al=b['audio_lengths'].to(device) if not args.need_data_aligned else 0
                    vl=b['vision_lengths'].to(device) if not args.need_data_aligned else 0
                    return model(b['text'].to(device),(b['audio'].to(device),al),(b['vision'].to(device),vl))['M']
        if not source.is_dir(): raise FileNotFoundError('Training source snapshot missing')
        # Strictly load the saved model. Never train, repair weights, or use a different checkpoint.
        state_dict=torch.load(weight,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(state_dict,strict=True,assign=True)
        del state_dict
        model=model.to(device).eval()
        ds,mask_path=wrap_existing(ds,root,dataset,protocol,condition,rate,method=='LNLN')
        loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
        ids=np.asarray([str(x) for x in ds.ids])
        ps,ys=[],[];seen=0
        with torch.inference_mode():
            for batch in loader:
                prediction=forward(batch).detach().cpu().numpy().reshape(-1)
                truth=batch['labels']['M'].detach().cpu().numpy().reshape(-1)
                if len(prediction)!=len(truth): raise ValueError('Batch prediction/label mismatch')
                ps.append(prediction);ys.append(truth);seen+=len(truth)
                progress(stage='predicting',seen=seen,total=len(ds),device=a.device)
        y,pred=np.concatenate(ys),np.concatenate(ps)
        if len(y)!=len(ds): raise ValueError('Not every validation sample was evaluated')
        extra.update(prediction_source='Strict checkpoint reload and validation-only inference',batch_size=batch_size,
                     data_file=str(args.featurePath if method in ('API','TLRA') else root/f'data/mmsa/{dataset}/unaligned_50.pkl'),
                     source_python_hashes={str(f.relative_to(source)):digest(f) for f in source.rglob('*.py')},torch_version=torch.__version__)
    if ids is None or len(ids)!=len(y): raise ValueError('Prediction ID count mismatch')
    report=scores(y,pred)
    tolerances={'MAE':.002,'Corr':.005,'Acc2_nonzero':.005,'F1_weighted_nonzero':.005}
    parity={k:dict(reference=v,reevaluated=report['metrics'][k],tolerance=tolerances[k],
                   passed=report['metrics'][k] is not None and abs(report['metrics'][k]-v)<=tolerances[k]) for k,v in expected.items()}
    if digest(weight)!=weight_hash or digest(lock_path)!=initial_lock_hash:
        raise RuntimeError('Checkpoint/validation lock changed during evaluation; no completion record written')
    pred_path=folder/'predictions_full_valid.npz'
    with pred_path.with_suffix('.npz.tmp').open('wb') as f:
        np.savez_compressed(f,ids=ids,y=y,pred=pred)
    os.replace(pred_path.with_suffix('.npz.tmp'),pred_path)
    success=all(v['passed'] for v in parity.values())
    report.update(schema_version=VERSION,status='complete' if success else 'needs_review',job=a.job,method=method,dataset=dataset,
                  split='valid',protocol=protocol,condition=condition,rate=rate if protocol=='frame' else None,
                  checkpoint_sha256=weight_hash,validation_lock_sha256=initial_lock_hash,predictions_sha256=digest(pred_path),
                  checkpoint_size=weight.stat().st_size,checkpoint_mtime_ns=weight.stat().st_mtime_ns,
                  mask_path=str(mask_path) if mask_path else None,mask_sha256=digest(mask_path) if mask_path else None,
                  parity_checks=parity,elapsed_seconds=time.time()-started,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  checkpoint_selection='Existing validation-selected checkpoint; no reselection, training, or test inference',**extra)
    atomic_json(folder/'metrics_full_valid.json',report)
    progress(stage=report['status'],metrics=report['metrics'])
    print(json.dumps({'job':a.job,'status':report['status'],'metrics':report['metrics']}),flush=True)
    if not success: raise SystemExit('Metrics differ from saved history; inspect parity_checks before using these results.')

if __name__=='__main__': main()
