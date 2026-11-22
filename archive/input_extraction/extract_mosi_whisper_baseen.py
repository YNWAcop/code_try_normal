"""All official MOSI clips, observed-audio-only Whisper base.en + BERT.

Eight-second independent windows match historical MOSI protocol. No native
transcripts or sentiment labels are opened. Interrupted clips are recomputed.
"""
from pathlib import Path
import argparse,json,time,os,wave,hashlib
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np,torch
from transformers import WhisperForConditionalGeneration,WhisperProcessor,AutoTokenizer,AutoModel
from prepare_mosi_audio_cache import get_ids
from mosei_vsr_common import save,sha
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'ugsr_strong_asr_probe_20261028/model'
BERT=Path('E:/UserData/Desktop/LNLN/pretrained/bert-base-uncased')

def main(args):
    args.out.mkdir(parents=True,exist_ok=True);parts=args.out/'parts';parts.mkdir(exist_ok=True)
    ids=get_ids();selected=min(args.limit,len(ids)) if args.limit else len(ids)
    protocol=dict(model=str(MODEL),model_sha256=sha(MODEL/'pytorch_model.bin'),source_sha256=sha(Path(__file__)),
        audio_protocol_sha256=sha(args.audio/'protocol.json'),sample_rate=16000,window_seconds=8,
        beam=1,max_new_tokens=96,device=args.device,dtype='float32',
        ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        input='current clip waveform only; no native transcript, label, neighbor or contextual prompt',
        torch_version=torch.__version__,bert=str(BERT),bert_pooling='non-special token mean, max96 tokens')
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (args.out/'protocol.json').exists():
        if json.loads((args.out/'protocol.json').read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Changed extraction protocol: fresh output required')
    save(args.out/'protocol.json',protocol)
    torch.set_num_threads(2)
    model=WhisperForConditionalGeneration.from_pretrained(MODEL,local_files_only=True).to(args.device).eval()
    processor=WhisperProcessor.from_pretrained(MODEL,local_files_only=True)
    pending=[];active={};done=0;started=time.time();state=ROOT/'mosi_whisper_status.json'

    def flush():
        nonlocal done
        if not pending:return
        encoded=processor([item[2] for item in pending],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():
            generated=model.generate(input_features=encoded.input_features.to(args.device),attention_mask=encoded.attention_mask.to(args.device),
                max_new_tokens=96,num_beams=1,do_sample=False,return_dict_in_generate=True,output_scores=True)
        texts=processor.batch_decode(generated.sequences,skip_special_tokens=True)
        count=len(generated.scores);prefix=generated.sequences.shape[1]-count
        sums=torch.zeros(len(pending),device=args.device);counts=torch.zeros_like(sums)
        for j,score in enumerate(generated.scores):
            tok=generated.sequences[:,prefix+j];good=tok<processor.tokenizer.eos_token_id
            lp=score.gather(1,tok[:,None]).squeeze(1)-torch.logsumexp(score,dim=1)
            lp=torch.where(good,lp,torch.zeros_like(lp));sums+=lp;counts+=good
        sums=sums.cpu().numpy();counts=counts.cpu().numpy()
        for k,(index,window,_) in enumerate(pending):
            entry=active[index];entry['windows'][window]=texts[k].strip()
            entry['log_probability_sum']+=float(sums[k]);entry['lexical_tokens']+=int(counts[k])
            if all(t is not None for t in entry['windows']):
                entry['text']=' '.join(t for t in entry['windows'] if t).strip()
                entry['mean_log_probability']=entry['log_probability_sum']/max(1,entry['lexical_tokens'])
                save(parts/f'{index:05d}.json',entry);done+=1;del active[index]
        pending.clear();del generated,encoded
        if done%64<len(texts) or done==selected:
            status=dict(stage='running',pid=os.getpid(),out=str(args.out),completed=done,total=selected,elapsed=time.time()-started,heartbeat=time.time())
            save(state,status);print(status,flush=True)

    for i in range(selected):
        target=parts/f'{i:05d}.json'
        if target.exists():
            existing=json.loads(target.read_text(encoding='utf-8'))
            if existing['id']!=str(ids[i]) or existing['fingerprint']!=fingerprint:raise RuntimeError('Incompatible per-clip transcript')
            done+=1;continue
        wav=args.audio/'wav'/f'{i:05d}.wav';meta=args.audio/'metadata'/f'{i:05d}.json'
        while not (wav.exists() and meta.exists()):
            flush()
            upstream=ROOT/'mosi_audio_cache_status.json'
            if upstream.exists() and json.loads(upstream.read_text(encoding='utf-8')).get('stage')=='failed_preserved':
                raise RuntimeError(f'Audio input {i} missing after failed prefetch; resume prefetch first')
            save(state,dict(stage='waiting_for_audio',pid=os.getpid(),out=str(args.out),index=i,completed=done,heartbeat=time.time()))
            time.sleep(10)
        info=json.loads(meta.read_text(encoding='utf-8'));assert info['id']==str(ids[i])
        if sha(wav)!=info['wav_sha256']:raise RuntimeError('Waveform hash mismatch')
        with wave.open(str(wav),'rb') as w:
            assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2
            waveform=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
        nwindow=(len(waveform)+127999)//128000
        active[i]=dict(id=str(ids[i]),index=i,fingerprint=fingerprint,waveform_sha256=info['wav_sha256'],
            duration=len(waveform)/16000,windows=[None]*nwindow,log_probability_sum=0.,lexical_tokens=0)
        for j,start in enumerate(range(0,len(waveform),128000)):
            pending.append((i,j,waveform[start:start+128000]))
            if len(pending)>=args.batch:flush()
    flush();assert not active
    del model,processor
    if args.device=='cuda':torch.cuda.empty_cache()
    if selected<len(ids):
        save(args.out/'pilot_completed.json',dict(total=selected,seconds=time.time()-started));return
    records=[json.loads((parts/f'{i:05d}.json').read_text(encoding='utf-8')) for i in range(len(ids))]
    texts=[r['text'] for r in records]
    tokenizer=AutoTokenizer.from_pretrained(BERT,local_files_only=True)
    bert=AutoModel.from_pretrained(BERT,local_files_only=True).to(args.device).eval();features=[]
    with torch.inference_mode():
        for start in range(0,len(ids),32):
            batch=tokenizer(texts[start:start+32],return_tensors='pt',padding=True,truncation=True,max_length=96,return_special_tokens_mask=True).to(args.device)
            special=batch.pop('special_tokens_mask');mask=batch.attention_mask*(1-special)
            h=bert(**batch).last_hidden_state
            features.append(((h*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]).cpu().numpy())
    quality=np.array([[r['duration'],len(r['windows']),len(r['text'].split()),r['mean_log_probability'],np.mean([not w for w in r['windows']])] for r in records],np.float32)
    array=np.concatenate(features);assert array.shape==(len(ids),768) and np.isfinite(array).all() and np.isfinite(quality).all()
    temp=args.out/'whisper_embeddings.tmp.npz'
    np.savez_compressed(temp,ids=ids,asr=array,quality=quality,text=np.array(texts),fingerprint=np.array(fingerprint));os.replace(temp,args.out/'whisper_embeddings.npz')
    save(args.out/'completed.json',dict(total=len(ids),fingerprint=fingerprint,feature_sha256=sha(args.out/'whisper_embeddings.npz'),seconds=time.time()-started))
    save(state,dict(stage='completed',out=str(args.out),total=len(ids),time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audio',type=Path,default=Path('D:/idea_final_overflow/data/mosi_audio_baseen_20261112'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch',type=int,default=16);p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
