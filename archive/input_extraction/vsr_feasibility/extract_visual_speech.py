"""Frozen visual-only speech encoder; all utterances, including long videos.

No audio stream, dataset transcript, sentiment label, or neighboring clip enters
the extractor. YuNet four-point alignment approximates the upstream MediaPipe
pipeline. Chunking and CTC greedy decoding are disclosed inference differences.
"""
from pathlib import Path
import sys,os,json,time,hashlib,argparse
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent; RUN=ROOT.parent; CODE=ROOT/'auto_avsr-main'
sys.path.insert(0,str(RUN/'vendor'));sys.path.insert(0,str(CODE))
import numpy as np,torch,cv2
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from preparation.detectors.mediapipe.video_process import VideoProcess

def atomic_json(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def mouth_chunks(path,chunk_frames=300):
    cap=cv2.VideoCapture(str(path.relative_to(RUN)))
    fps=cap.get(cv2.CAP_PROP_FPS);assert fps>0,(path,fps)
    detector=cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx','',(320,320),.75,.3,5000)
    frames=[];landmarks=[];chunks=[];detected=0;total=0;source_i=0;target_i=0
    def flush():
        if any(p is not None for p in landmarks):
            gray=VideoProcess(convert_gray=True)(np.asarray(frames),landmarks)
            chunks.append((gray[:,4:92,4:92].copy(),True))
        else:
            chunks.append((np.zeros((len(frames),88,88),dtype=np.uint8),False))
        frames.clear();landmarks.clear()
    while True:
        ok,frame=cap.read()
        if not ok:break
        wanted=int(np.rint(target_i*fps/25))
        while source_i==wanted:
            h,w=frame.shape[:2];scale=320/max(h,w)
            small=cv2.resize(frame,(round(w*scale),round(h*scale)))
            detector.setInputSize((small.shape[1],small.shape[0]));_,faces=detector.detect(small)
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));total+=1
            if faces is None:landmarks.append(None)
            else:
                face=max(faces,key=lambda f:f[2]*f[3]);pts=face[4:14].reshape(5,2)/scale
                eyes=pts[:2][np.argsort(pts[:2,0])]
                landmarks.append(np.vstack((eyes,pts[2],pts[3:].mean(0))));detected+=1
            target_i+=1;wanted=int(np.rint(target_i*fps/25))
            if len(frames)==chunk_frames:flush()
        source_i+=1
    cap.release()
    if frames:flush()
    assert total>0
    return chunks,detected/total,total

def text_from_ids(ids,tokens):
    selected=[];last=None
    for v in ids:
        v=int(v)
        if v!=last and v!=0:selected.append(tokens[v])
        last=v
    return ''.join(selected).replace('<eos>','').replace('<space>',' ').replace('\u2581',' ').strip()

def cached_ok(path,ident):
    if not path.exists():return False
    try:
        with np.load(path) as z:
            return str(z['id'])==str(ident) and z['pooled'].shape==(1536,) and z['temporal'].shape==(16,768) and np.isfinite(z['pooled']).all()
    except Exception:return False

def main(limit=None,workers=4):
    os.chdir(RUN);cv2.setNumThreads(1);torch.set_num_threads(2)
    out=ROOT/'mosi_vsr';out.mkdir(exist_ok=True);parts=out/'parts';parts.mkdir(exist_ok=True)
    weights=ROOT/'vsr_trlrs2lrs3vox2avsp_base.pth'
    units=(CODE/'spm/unigram/unigram5000_units.txt').read_text(encoding='utf-8').splitlines()
    tokens=['<blank>']+[s.split()[0] for s in units]+['<eos>']
    model=E2E(len(tokens),'video',ctc_weight=.1)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.cuda().eval()
    with np.load(r'E:\UserData\Desktop\idea_final\data\processed\ebmc_features_20261111\all_author\mosi_features.npz') as z:
        ids=np.concatenate([z[f'{s}_ids'] for s in ('train','valid','test')])
    atomic_json(out/'protocol.json',dict(input='current-utterance RGB only; no audio, original transcript, labels or neighboring clips',
        source='official Auto-AVSR 3291-hour visual speech model',weights=json.loads((ROOT/'weights_verified.json').read_text()),
        preprocessing='25 fps; YuNet largest face; four-point mouth alignment; gray88 normalized .421/.165',
        chunk_frames=300,aggregation='all frame embeddings mean/std; 16 equal-time summaries; CTC greedy text and entropy',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),extra_pretraining=True))
    indices=list(range(len(ids)))[:limit];todo=[i for i in indices if not cached_ok(parts/f'{i:05d}.npz',ids[i])]
    started=time.time();done=len(indices)-len(todo)
    depth=2*workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={i:pool.submit(mouth_chunks,RUN/'rgb_mosi/media'/f'{i:05d}.mp4') for i in todo[:depth]}
        for slot,i in enumerate(todo):
            chunks,detection,nframes=pending.pop(i).result()
            if slot+depth<len(todo):
                j=todo[slot+depth];pending[j]=pool.submit(mouth_chunks,RUN/'rgb_mosi/media'/f'{j:05d}.mp4')
            embeddings=[];entropy=[];blank=[];texts=[];present=[]
            for gray,ok in chunks:
                if not ok:
                    embeddings.append(np.zeros((len(gray),768),np.float32));present.extend([False]*len(gray));continue
                x=torch.as_tensor(gray.copy(),device='cuda',dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                x=(x/255-.421)/.165
                with torch.inference_mode():
                    h=model.proj_encoder(model.frontend(x));enc,_=model.encoder(h,None)
                    lp=model.ctc.log_softmax(enc)[0];p=lp.exp()
                    entropy.append((-(p*lp).sum(-1)/np.log(len(tokens))).cpu().numpy());blank.append(p[:,0].cpu().numpy())
                    texts.append(text_from_ids(lp.argmax(-1).cpu().numpy(),tokens));embeddings.append(enc[0].cpu().numpy())
                present.extend([True]*len(gray))
            h=np.concatenate(embeddings);present=np.asarray(present)
            assert len(h)==nframes
            usable=h[present]
            pooled=np.concatenate((usable.mean(0),usable.std(0))) if len(usable) else np.zeros(1536,np.float32)
            temporal=np.stack([a.mean(0) if len(a) else np.zeros(768) for a in np.array_split(h,16)]).astype(np.float32)
            quality=np.asarray([detection,present.mean(),np.concatenate(entropy).mean() if entropy else 1,
                np.concatenate(blank).mean() if blank else 1,nframes/25],np.float32)
            path=parts/f'{i:05d}.npz';temp=parts/f'{i:05d}.tmp.npz'
            np.savez_compressed(temp,id=ids[i],pooled=pooled.astype(np.float32),temporal=temporal,quality=quality,
                transcript=' '.join(texts),frames=nframes);temp.replace(path)
            done+=1
            if done%10==0 or done==len(indices):
                status=dict(stage='extracting',done=done,total=len(ids),elapsed=time.time()-started)
                atomic_json(out/'status.json',status);print(status,flush=True)
    if limit is not None and limit<len(ids):return
    records=[]
    for i,ident in enumerate(ids):
        assert cached_ok(parts/f'{i:05d}.npz',ident)
        with np.load(parts/f'{i:05d}.npz') as z:records.append({k:z[k] for k in ('pooled','temporal','quality','transcript')})
    temp=out/'features.tmp.npz'
    np.savez_compressed(temp,ids=ids,**{k:np.stack([r[k] for r in records]) for k in records[0]});temp.replace(out/'features.npz')
    atomic_json(out/'status.json',dict(stage='completed',done=len(ids),total=len(ids),elapsed=time.time()-started))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();main(args.limit,args.workers)
