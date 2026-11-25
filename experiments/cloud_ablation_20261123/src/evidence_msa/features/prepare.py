"""Export frozen observed-input features; exclude all historic supervised predictions."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
from ..settings import DATA_ROOT
ROOT=DATA_ROOT
SPLITS=('train','valid','test')
def ordered(z,ids,key):
    ix={str(v):i for i,v in enumerate(z['ids'])}
    if len(ix)!=len(z['ids']): raise ValueError('Duplicate source IDs')
    return np.asarray(z[key][[ix[str(v)] for v in ids]],dtype=np.float32)
def prepare(ds):
    dest=ROOT/'ready'/ds; dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/'generated'/ds
    audit={'dataset':ds,'splits':{},'supervised_historical_predictions_used':False}
    with np.load(ROOT/f'author/{ds}_features.npz') as author, np.load(source/'whisper_embeddings.npz') as asr, np.load(source/'vsr_features.npz') as vsr, np.load(source/'vsr_bert_words.npz') as words, np.load(source/'expression_features.npz') as face:
        id_sets={s:set(author[s+'_ids'].astype(str)) for s in SPLITS}
        group_sets={s:{i.rsplit('$_$',1)[0] for i in id_sets[s]} for s in SPLITS}
        for i,s in enumerate(SPLITS):
            assert len(id_sets[s])==len(author[s+'_ids'])
            for t in SPLITS[i+1:]:
                assert not id_sets[s]&id_sets[t],(s,t,'sample overlap')
                assert not group_sets[s]&group_sets[t],(s,t,'video overlap')
        for s in SPLITS:
            ids=author[s+'_ids'].astype(str)
            logits=ordered(face,ids,'logits');valid=ordered(face,ids,'valid');q=ordered(face,ids,'quality')[:,:,0]*valid
            p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            w=q/q.sum(1,keepdims=True).clip(1e-8);mean=(p*w[:,:,None]).sum(1)
            expression=np.column_stack((mean,np.sqrt(((p-mean[:,None])**2*w[:,:,None]).sum(1)),(p*valid[:,:,None]).max(1)))
            values=dict(ids=ids,groups=np.array([i.rsplit('$_$',1)[0] for i in ids]),y=author[s+'_y'],
                T=author[s+'_text'],A_native=author[s+'_audio'],A_words=ordered(asr,ids,'asr'),
                V_native=author[s+'_temporal'],V_lip=ordered(vsr,ids,'pooled'),V_words=ordered(words,ids,'features'),
                V_expression=expression,V_quality=ordered(vsr,ids,'quality'),V_timeline=ordered(vsr,ids,'temporal'))
            for k,v in values.items():
                if k not in ('ids','groups'):
                    if not np.isfinite(v).all():raise ValueError((ds,s,k,'nonfinite'))
                    values[k]=v.astype(np.float32)
            np.savez_compressed(dest/(s+'.npz'),**values)
            # No model performance is calculated here.
            audit['splits'][s]={'n':len(ids),'videos':len(group_sets[s]),'id_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'shapes':{k:list(v.shape) for k,v in values.items()}}
    audit['split_video_overlap']=False
    audit['limitation']='ID/video partition audit only; external-pretraining membership is not established by this check.'
    (dest/'input_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(ds,'prepared and ID/video partitions checked',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mosi','mosei','both'],default='both');a=p.parse_args()
    for ds in (['mosi','mosei'] if a.dataset=='both' else [a.dataset]):prepare(ds)
