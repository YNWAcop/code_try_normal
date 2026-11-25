"""Validation-only inference at the existing selected checkpoint, uniform metrics."""
import json
import numpy as np
import torch
from metrics_common import scores,atomic_json,digest

def evaluate_selected(model,loader,cli,args):
    out=cli.out.resolve();lock=json.loads((out/'validation_lock.json').read_text(encoding='utf-8'))
    checkpoint=out/'best.pt';sha=digest(checkpoint)
    if lock.get('checkpoint_sha256') and sha!=lock['checkpoint_sha256']:raise RuntimeError('Checkpoint hash changed')
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True),strict=True)
    model.to(args.device).eval();ys=[];ps=[];ids=[]
    with torch.inference_mode():
        for b in loader:
            x={k:(v.to(args.device) if torch.is_tensor(v) else v) for k,v in b.items() if k!='labels'}
            if cli.method=='EMT_DLFR':
                pred=model((x['text'],x['text_m']),(x['audio'],x['audio_m'],x['audio_lengths']),(x['vision'],x['vision_m'],x['vision_lengths']))['pred_m']
            elif cli.method=='TFR_NET':
                pred,_=model((x['text'],x['text_m'],x['text_missing_mask']),(x['audio'],x['audio_m'],x['audio_mask'],x['audio_missing_mask']),(x['vision'],x['vision_m'],x['vision_mask'],x['vision_missing_mask']))
            elif cli.method=='MISA':
                model.Model.visible_audio_lengths=b['audio_lengths'].cpu().long().clamp(min=1)
                model.Model.visible_vision_lengths=b['vision_lengths'].cpu().long().clamp(min=1)
                pred=model(x['text'],x['audio'],x['vision'])['M']
            else:pred=model(x['text'],x['audio'],x['vision'])
            ps.append(pred.detach().cpu().numpy().reshape(-1));ys.append(b['labels']['M'].numpy().reshape(-1));ids.extend(map(str,b['id']))
    y,pred=np.concatenate(ys),np.concatenate(ps)
    assert len(y)==len(loader.dataset)==len(ids)
    if (out/'history.json').exists():
        history=json.loads((out/'history.json').read_text());reference=next(r['validation'] for r in history if r['epoch']==lock['selected_epoch'])
    else:
        history=json.loads((out/'validation_history.json').read_text());reference=next(r['metrics'] for r in history if r['epoch']==lock['selected_epoch'])
    report=scores(y,pred);checks={}
    for source,key in [('MAE','MAE'),('Corr','Corr'),('Non0_acc_2','Acc2_nonzero'),('Non0_F1_score','F1_weighted_nonzero')]:
        if source in reference:
            value=report['metrics'][key];tol=.002 if key=='MAE' else .005
            checks[key]={'reference':reference[source],'actual':value,'tolerance':tol,'passed':value is not None and abs(value-reference[source])<=tol}
    success=bool(checks) and all(c['passed'] for c in checks.values())
    np.savez_compressed(out/'predictions_full_valid.npz',ids=np.asarray(ids),y=y,pred=pred)
    report.update(schema_version='priority-whole-v1',status='complete' if success else 'needs_review',job=out.name,method=cli.method,dataset=cli.dataset,protocol=cli.protocol,condition=cli.condition,split='valid',selected_epoch=lock['selected_epoch'],checkpoint_sha256=sha,validation_lock_sha256=digest(out/'validation_lock.json'),smoke=cli.smoke,parity_checks=checks)
    atomic_json(out/'metrics_full_valid.json',report)
    if not success:raise RuntimeError('Selected-checkpoint metrics differ from original validation; inspect parity_checks')
    print('FULL VALIDATION METRICS COMPLETE '+out.name,flush=True)
