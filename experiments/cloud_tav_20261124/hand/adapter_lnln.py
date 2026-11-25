"""LNLN single run adapted to fixed whole-modality or frame-rate corruption.
Validation MAE selects checkpoints. No test inference occurs here.
"""
import argparse, ast, hashlib, json, os, shutil, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=['mosi','mosei'],required=True);p.add_argument('--condition',default='TAV');p.add_argument('--protocol',choices=['whole','frame'],default='whole');p.add_argument('--rate',type=float,default=.3);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.condition=''.join(m for m in 'TAV' if m in a.condition.upper())
    if not a.condition or not 0<=a.rate<1:raise ValueError('Invalid condition or rate')
    root=a.root.resolve();src=root/'baselines/upstream/LNLN'
    tag=a.condition if a.protocol=='whole' else f'frame{a.rate:.1f}'
    out=root/'baseline_results'/('LNLN_'+a.dataset+'_'+tag+('_smoke' if a.smoke else ''))
    out.mkdir(parents=True,exist_ok=False)
    dst=root/'baselines/adapted'/out.name
    shutil.copytree(src,dst)
    datafile=dst/'core/dataset.py';s=datafile.read_text(encoding='utf-8')
    assert 'np.random.seed(missing_seed)' in s
    s=s.replace('np.random.seed(missing_seed)','pass  # no manual seed; fresh random missingness')
    signature="    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):"
    assert signature in s
    injected='''
        # Whole-modality adaptation: fixed condition, no hidden lengths at inference.
        visible = {'text':'T', 'audio':'A', 'vision':'V'}[mode] in CONDITION
        missing_rate = np.full((len(modality),1), 0. if visible else 1.)
        if not visible:
            mask=np.zeros_like(input_mask) if mode=='text' else np.zeros(modality.shape[:2])
            if mode=='text':
                out=np.zeros_like(modality);out[:,0]=101;out[:,1]=102;mask[:,:2]=1
                return out,np.full(len(modality),2),mask,mask.copy()
            mask[:,0]=1
            return np.zeros_like(modality),np.ones(len(modality),dtype=int),mask,np.zeros_like(mask)
'''.replace('CONDITION',repr(a.condition))
    s=s.replace(signature,signature+injected)
    s=s.replace('        return sample',"        for key, modality in [('missing_rate_l','T'),('missing_rate_a','A'),('missing_rate_v','V')]:\n            sample['labels'][key] = torch.tensor([0. if modality in "+repr(a.condition)+" else 1.])\n        if 'T' not in "+repr(a.condition)+": sample['text_m'][2,:] = 0\n        return sample")
    # No test loader is constructed. Raw pickle still contains all split arrays.
    s=s.replace("        'valid': MMDataset(args, mode='valid'),\n        'test': MMDataset(args, mode='test')", "        'valid': MMDataset(args, mode='valid')")
    assert "'test': MMDataset" not in s
    datafile.write_text(s,encoding='utf-8');ast.parse(s)
    os.chdir(dst);sys.path.insert(0,str(dst));sys.argv=[str(dst/'train.py')]
    import torch, numpy as np, yaml
    import train as upstream
    from torch.utils.data import DataLoader, Subset
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    cfg=yaml.safe_load((dst/f'configs/train_{a.dataset}.yaml').read_text())
    cfg['dataset']['dataPath']=str(root/f'data/mmsa/{a.dataset}/unaligned_50.pkl')
    cfg['model']['feature_extractor']['bert_pretrained']=str(root/'models/bert-base-uncased')
    cfg['base']['num_workers']=0;cfg['base']['seed']=None
    cfg['base']['missing_rate_eval_test']=a.rate if a.protocol=='frame' else 0.
    if a.smoke:cfg['base']['n_epochs']=1
    (out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    loaders=upstream.MMDataLoader(cfg)
    ds=loaders['train'].dataset
    observed=[ds.text.shape[2],ds.vision.shape[1],ds.audio.shape[1]]
    if observed!=cfg['model']['feature_extractor']['input_length']:
        raise RuntimeError(f'Sequence shape mismatch: {observed} vs author config. Do not silently resample.')
    expected=cfg['model']['feature_extractor']['input_dims']
    if [768,ds.vision.shape[2],ds.audio.shape[2]]!=expected:raise RuntimeError('Feature dimensions mismatch')
    from corruption import wrap_loader_dataset
    loaders={k:DataLoader(wrap_loader_dataset(v.dataset,root,a.dataset,k,a.protocol,a.condition,a.rate,lnln=True),batch_size=cfg['base']['batch_size'],shuffle=(k=='train'),num_workers=0) for k,v in loaders.items()}
    (out/'mask_protocol.json').write_text(json.dumps({'protocol':a.protocol,'condition':a.condition,'rate':a.rate,'mask_paths':{k:getattr(v.dataset,'mask_path',None) for k,v in loaders.items()}},indent=2))
    if a.smoke:
        loaders={k:DataLoader(Subset(v.dataset,range(min(16,len(v.dataset)))),batch_size=8,shuffle=(k=='train')) for k,v in loaders.items()}
    model=upstream.build_model(cfg).to(upstream.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['base']['lr'],weight_decay=cfg['base']['weight_decay'])
    scheduler=upstream.get_scheduler(optimizer,cfg);loss_fn=upstream.MultimodalLoss(cfg)
    metrics=upstream.MetricsTop('regression').getMetics(a.dataset)
    best=float('inf');history=[];selected=None
    for epoch in range(1,cfg['base']['n_epochs']+1):
        upstream.train(model,loaders['train'],optimizer,loss_fn,epoch,metrics)
        result=upstream.evaluate(model,loaders['valid'],loss_fn,epoch,metrics)
        score=float(result['MAE']);history.append({'epoch':epoch,'valid':result})
        (out/'history.json').write_text(json.dumps(history,indent=2,default=float))
        if not np.isfinite(score):raise RuntimeError('Nonfinite validation MAE')
        if score<best:
            best=score;selected=epoch;torch.save(model.state_dict(),out/'best.pt')
        print('VALID',epoch,result,flush=True);scheduler.step()
    lock={'model':'LNLN','dataset':a.dataset,'condition':a.condition,'protocol':a.protocol,'rate':a.rate,'smoke':a.smoke,'status':'validation_locked_no_test_evaluated','selected_epoch':selected,'valid_MAE':best,'selection':'validation MAE under the same corruption protocol','adaptation':'fixed whole-modality or fixed-rate frame-missing adaptation; original complete-input privileged branch remains TRAIN-ONLY; not native stochastic-rate LNLN reproduction','seed_policy':'no manual seeds; one run','upstream_train_sha256':hashlib.sha256((src/'train.py').read_bytes()).hexdigest()}
    (out/'validation_lock.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(lock,indent=2))

if __name__=='__main__':main()
