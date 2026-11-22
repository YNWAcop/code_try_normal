"""MOSEI VSR I/O shared by matched kernel and neural ablations.

Does not modify or execute MOSI experiments. Both MOSEI paths use author A512.
"""
from pathlib import Path
import hashlib,json,os,zipfile
import numpy as np
import torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parent
REPO=Path(r'E:\UserData\Desktop\idea_final')
SPLITS=('train','valid','test')
SEEDS=(2051,2052,2053)
PROBE_FAMILIES=('manet_control','lip_latent','lip_words','visual_semantics','reliable_semantics')
NEURAL_FAMILIES=('uniform','gated','reliable','capacity_control')
MASKS=('V','AV','TV','TAV')
FEATURE=ROOT/'vsr_feasibility/mosei_vsr/features.npz'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def official_ids():
    with np.load(REPO/'data/processed/ebmc_features_20261111/all_author/mosei_features.npz') as z:
        return np.concatenate([z[s+'_ids'] for s in SPLITS])


def complete_features(path=FEATURE):
    """Check atomic aggregate IDs plus array headers without loading 1.1GB frames."""
    if not path.exists():return False
    try:
        ids=official_ids()
        with np.load(path,allow_pickle=False) as z:
            if len(ids)!=22856 or not np.array_equal(ids,z['ids']):return False
            extension=path.parent/'crop_boundary_extension.json'
            if extension.exists():
                if 'crop_boundary_extension_sha256' not in z or str(z['crop_boundary_extension_sha256'])!=sha(extension):return False
        expected={'pooled':(22856,1536),'temporal':(22856,16,768),'quality':(22856,5),'transcript':(22856,)}
        with zipfile.ZipFile(path) as z:
            for key,shape in expected.items():
                with z.open(key+'.npy') as f:
                    version=np.lib.format.read_magic(f)
                    actual,_,_=np.lib.format._read_array_header(f,version)
                    if actual!=shape:return False
        return True
    except (OSError,ValueError,EOFError,zipfile.BadZipFile,KeyError):return False


def encode_words(ids,texts):
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    cache=FEATURE.parent/'bert_words.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            np.testing.assert_array_equal(ids,z['ids'])
            np.testing.assert_array_equal(texts,z['transcripts'])
            x=z['features']
            if x.shape!=(len(ids),768) or not np.isfinite(x).all():raise ValueError('Invalid cached BERT features')
            return x
    from transformers import AutoTokenizer,AutoModel
    model_path=Path(r'E:\UserData\Desktop\LNLN\pretrained\bert-base-uncased')
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModel.from_pretrained(model_path,local_files_only=True).cuda().eval()
    features=[]
    with torch.inference_mode():
        for start in range(0,len(texts),32):
            batch=tok(texts[start:start+32].tolist(),return_tensors='pt',padding=True,truncation=True,max_length=128,return_special_tokens_mask=True).to('cuda')
            special=batch.pop('special_tokens_mask');mask=batch.attention_mask*(1-special)
            h=model(**batch).last_hidden_state
            features.append(((h*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]).cpu().numpy())
    x=np.concatenate(features)
    if not np.isfinite(x).all():raise ValueError('Non-finite new BERT features')
    temp=cache.with_name('bert_words.tmp.npz')
    np.savez_compressed(temp,ids=ids,transcripts=texts,features=x);os.replace(temp,cache)
    save(cache.with_suffix('.protocol.json'),dict(model=str(model_path),weights_sha256=sha(model_path/'model.safetensors'),
        input='CTC visual-only generated text; no original transcript',pooling='non-special, non-padding token mean',max_length=128))
    del model;torch.cuda.empty_cache()
    return x


def load_data(with_frames=False):
    if not complete_features():raise ValueError('MOSEI VSR aggregate incomplete or IDs/shapes differ')
    with np.load(REPO/'data/processed/ebmc_features_20261111/all_author/mosei_features.npz') as z:
        data={s:{k:z[f'{s}_{k}'].astype(np.float32) if k not in ('ids',) else z[f'{s}_{k}']
                 for k in ('ids','y','temporal','text','audio')} for s in SPLITS}
    ids=np.concatenate([d['ids'] for d in data.values()]);sizes=[len(d['ids']) for d in data.values()]
    groups={s:set(str(i).split('$_$')[0] for i in d['ids']) for s,d in data.items()}
    for j,s in enumerate(SPLITS):
        for other in SPLITS[:j]:
            if groups[s]&groups[other]:raise ValueError('Video crosses official splits')
    anchor_dir=REPO/'experiments/ebmc_visual_20261111/mosei_author_anchor'
    with np.load(anchor_dir/'training_audit.npz') as z:np.testing.assert_array_equal(data['train']['ids'],z['ids'])
    with np.load(anchor_dir/'anchors.npz') as z:
        for s,d in data.items():
            d['anchor']=z[s]
            if d['anchor'].shape!=(len(d['ids']),7):raise ValueError('Anchor shape mismatch')
    with np.load(FEATURE) as z:
        np.testing.assert_array_equal(ids,z['ids'])
        pooled=z['pooled'];quality=z['quality'].copy();texts=z['transcript']
        frames=z['temporal'] if with_frames else None
    frame_valid=(np.arange(16)[None,:]<np.minimum(np.rint(quality[:,-1]*25).astype(int),16)[:,None]) & (quality[:,1:2]>0)
    quality[:,-1]=np.log1p(quality[:,-1])
    words=encode_words(ids,texts)
    with np.load(ROOT/'rgb_mosei/features.npz') as z:
        np.testing.assert_array_equal(ids,z['ids']);logits=z['logits'];valid=z['valid'];q=z['quality'][:,:,0]*valid
    p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    weight=q/q.sum(1,keepdims=True).clip(1e-8);mean=(p*weight[:,:,None]).sum(1)
    expression=np.column_stack((mean,np.sqrt(((p-mean[:,None])**2*weight[:,:,None]).sum(1)),(p*valid[:,:,None]).max(1)))
    offset=0
    for s,n in zip(SPLITS,sizes):
        d=data[s];sl=slice(offset,offset+n)
        for key,value in [('lip',pooled),('words',words),('quality',quality),('expression',expression)]:d[key]=value[sl]
        if with_frames:d['frames']=frames[sl];d['valid']=frame_valid[sl]
        offset+=n
    return data


def protocol(kind):
    extension=FEATURE.parent/'crop_boundary_extension.json'
    extension_info=None
    if extension.exists():
        payload=json.loads(extension.read_text(encoding='utf-8'))
        for index,record in payload['records'].items():
            if sha(FEATURE.parent/'parts'/f'{int(index):05d}.npz')!=record['feature_sha256']:
                raise ValueError('Recovered crop cache differs from recorded extension')
        extension_info=dict(path=str(extension),sha256=sha(extension),records=payload['records'],policy=payload['policy'])
    return dict(dataset='mosei',method=kind,seeds=list(SEEDS),source_sha256=sha(Path(__file__)),crop_boundary_extension=extension_info,
        author_cache=str(REPO/'data/processed/ebmc_features_20261111/all_author/mosei_features.npz'),
        visual_speech_cache=str(FEATURE),ctc_bert_cache=str(FEATURE.parent/'bert_words.npz'),
        expression_cache=str(ROOT/'rgb_mosei/features.npz'),
        text_anchor=str(REPO/'experiments/ebmc_visual_20261111/mosei_author_anchor/anchors.npz'),
        input='author T1024/A512/MA-Net V1024 plus utterance-local visual speech, CTC-BERT and facial expression',
        audio='Both MOSEI probe and neural models use author A512; no Whisper. MOSI probe used different A779 resources.',
        extra_resources='official 3291h Auto-AVSR; BERT of visual-only CTC words; EmotiEffLib',
        selection='fixed validation five-metric objective; selected models saved before test predictions; exploratory repeated benchmark',
        preprocessing='fit only official training; text-to-V nuisance uses GroupKFold4 by original video',
        limits='YuNet approximate alignment, independent 12s chunks, noisy CTC text; no new lipreading model claimed')


def complete_run(out,families,neural=False):
    """An intermediate results.json never means an entire stage completed."""
    try:
        flag=json.loads((out/'completed.json').read_text(encoding='utf-8'))
        if flag.get('families')!=list(families) or flag.get('seeds')!=list(SEEDS):return False
        results=json.loads((out/'results.json').read_text(encoding='utf-8'))
        if not set(families)<=set(results):return False
        with np.load(REPO/'data/processed/ebmc_features_20261111/all_author/mosei_features.npz') as z:
            expected_ids={s:z[s+'_ids'] for s in ('valid','test')}
        masks=('T','A','V','TA','TV','AV','TAV') if neural else MASKS
        def predictions_ok(path,split):
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['ids'],expected_ids[split]):return False
                return all(z[m].shape==(len(expected_ids[split]),) and np.isfinite(z[m]).all() for m in masks)
        for family in families:
            for seed in SEEDS:
                if neural and (out/f'{family}_{seed}.pt').stat().st_size==0:return False
                if not neural:
                    if not all((out/f'{family}_{mask}_{seed}.joblib').stat().st_size>0 for mask in MASKS):return False
                for split in ('valid','test'):
                    if not predictions_ok(out/f'{family}_{seed}_{split}.npz',split):return False
            for split in ('valid','test'):
                if not predictions_ok(out/f'{family}_ensemble_{split}.npz',split):return False
        return True
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):return False


def prepare_neural(dataset,out):
    if dataset!='mosei':raise ValueError('MOSEI wrapper only')
    data=load_data(with_frames=True);scalers={}
    # Match MOSI neural scaling without loading unused native-frame caches.
    for key in ('text','audio','temporal','lip','words','expression','quality'):
        sc=StandardScaler().fit(data['train'][key]);scalers[key]=sc
        for d in data.values():d[key]=np.clip(sc.transform(d[key]),-8,8).astype(np.float32)
    visual={s:np.column_stack([d[k] for k in ('temporal','lip','words','expression','quality')]) for s,d in data.items()}
    text_pca=PCA(64,random_state=2051).fit(data['train']['text'])
    tx={s:text_pca.transform(d['text']) for s,d in data.items()}
    groups=np.array([str(i).split('$_$')[0] for i in data['train']['ids']]);folds=np.empty(len(groups),int)
    shared=np.zeros_like(visual['train'])
    for fold,(tr,va) in enumerate(GroupKFold(4).split(tx['train'],groups=groups)):
        nuisance=Ridge(alpha=10).fit(tx['train'][tr],visual['train'][tr]);shared[va]=nuisance.predict(tx['train'][va]);folds[va]=fold
    nuisance=Ridge(alpha=10).fit(tx['train'],visual['train'])
    rho=np.clip(1-((visual['train']-shared)**2).mean(0)/(visual['train'].var(0)+1e-8),0,1);rho[-5:]=0
    mean_v=visual['train'].mean(0)
    frame_pca=PCA(32,random_state=2051,whiten=True).fit(data['train']['frames'][data['train']['valid']])
    # base.run tensors requires semantic, but these four families do not use its loss.
    semantic_pca=PCA(32,random_state=2051,whiten=True).fit(data['train']['text'])
    for s,d in data.items():
        predicted=shared if s=='train' else nuisance.predict(tx[s])
        residual=visual[s]-rho*(predicted-mean_v)
        d['temporal']=np.column_stack((visual[s],residual))
        f=d['frames'];flat=frame_pca.transform(f.reshape(-1,768)).reshape(len(f),16,32)
        d['frames']=np.clip(flat,-7,7).astype(np.float32)*d['valid'][:,:,None]
        d['semantic']=np.clip(semantic_pca.transform(d['text']),-5,5).astype(np.float32)
        d['anchor']=d['anchor'][:,0].astype(np.float32)
        for k in ('lip','words','expression','quality'):del d[k]
    joblib.dump(dict(scalers=scalers,text_pca=text_pca,nuisance=nuisance,rho=rho,mean_v=mean_v,
                     frame_pca=frame_pca,semantic_pca=semantic_pca),out/'vsr_preprocessing.joblib',compress=3)
    np.savez_compressed(out/'training_audit.npz',ids=data['train']['ids'],fold=folds,rho=rho)
    return data
