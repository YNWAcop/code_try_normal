"""Current-utterance audio from CRC-checked MOSEI ZIP members; no labels/text."""
from pathlib import Path
import argparse,json,re,subprocess,time,os,wave,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from mosei_archive_io import decode_member,URL,ZIP_BYTES
from mosei_vsr_common import save,sha
ROOT=Path(__file__).resolve().parent
IDS=Path('E:/UserData/Desktop/idea_final/data/processed/ebmc_features_20261111/all_author/mosei_features.npz')
FFMPEG=ROOT.parent/'ugsr_data_20261028/ffmpeg.exe'
INDEX=ROOT/'rgb_mosei/zip_index.json'

def get_ids():
    with np.load(IDS) as z:return np.concatenate([z[s+'_ids'] for s in ('train','valid','test')]).astype(str)

def one(index,identifier,info,out):
    wav=out/'wav'/f'{index:05d}.wav';meta=out/'metadata'/f'{index:05d}.json'
    if meta.exists() and wav.exists():
        old=json.loads(meta.read_text())
        if old['id']==identifier and old['source_crc32']==info['crc'] and old['wav_sha256']==sha(wav):return 'cached'
    video,clip=identifier.split('$_$')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',video) or not clip.isdecimal():raise ValueError('Invalid clip identifier')
    member=f'Raw/{video}/{clip}.mp4';media=out/'tmp'/f'{index:05d}.mp4'
    try:
        video_bytes=decode_member(member,info);media.write_bytes(video_bytes)
        run=subprocess.run([str(FFMPEG),'-nostdin','-loglevel','error','-threads','1','-i',str(media),
                            '-vn','-ac','1','-ar','16000','-f','s16le','pipe:1'],capture_output=True,timeout=120)
        if run.returncode or len(run.stdout)<2 or len(run.stdout)%2:raise ValueError('Audio decode failed or empty PCM')
        pcm=run.stdout;temporary=wav.with_suffix('.tmp.wav')
        with wave.open(str(temporary),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm)
        os.replace(temporary,wav)
        save(meta,dict(id=identifier,index=index,member=member,source_crc32=info['crc'],source_uncompressed_bytes=info['size'],
                       source_video_sha256=__import__('hashlib').sha256(video_bytes).hexdigest(),wav_sha256=sha(wav),
                       samples=len(pcm)//2,sample_rate=16000,channels=1,sample_width_bytes=2,created=time.time()))
        return 'decoded'
    finally:
        # This exact task-created temporary file only; no directory deletion.
        if media.exists():media.unlink()

def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    for name in ('wav','metadata','tmp'):(args.out/name).mkdir(exist_ok=True)
    ids=get_ids();index=json.loads(INDEX.read_text());selected=range(min(args.limit,len(ids))) if args.limit else range(len(ids))
    protocol=dict(url=URL,zip_bytes=ZIP_BYTES,index_sha256=sha(INDEX),ids_sha256=__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(),
                  ffmpeg=str(FFMPEG),ffmpeg_sha256=sha(FFMPEG),source_sha256=sha(Path(__file__)),scope='official IDs only, current clip audio; no transcript or sentiment labels read')
    save(args.out/'protocol.json',protocol);failed=[];counts={'cached':0,'decoded':0};start=time.time()
    save(ROOT/'mosei_audio_prefetch_status.json',dict(stage='running',pid=os.getpid(),out=str(args.out),selected=len(selected),started=start))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={}
        for i in selected:
            v,c=ids[i].split('$_$');futures[pool.submit(one,i,str(ids[i]),index[f'Raw/{v}/{c}.mp4'],args.out)]=i
        for f in as_completed(futures):
            i=futures[f]
            try:counts[f.result()]+=1
            except Exception as exc:
                failed.append(dict(index=i,id=str(ids[i]),error=type(exc).__name__+': '+str(exc)[:200]))
                save(args.out/'failures.json',failed)
            done=sum(counts.values())+len(failed)
            if done%32==0 or done==len(selected):
                status=dict(stage='running',pid=os.getpid(),out=str(args.out),done=done,selected=len(selected),counts=counts,failed=len(failed),elapsed=time.time()-start,heartbeat=time.time())
                save(ROOT/'mosei_audio_prefetch_status.json',status);print(status,flush=True)
    save(args.out/'download_summary.json',dict(selected=len(selected),counts=counts,failed=failed,seconds=time.time()-start))
    if failed:
        save(ROOT/'mosei_audio_prefetch_status.json',dict(stage='failed_preserved',out=str(args.out),failed=len(failed),done=len(selected)))
        raise RuntimeError('Some audio clips failed; rerun resumes only missing/invalid entries')
    if len(selected)==len(ids):save(args.out/'completed.json',dict(total=len(ids),time=time.time()))
    save(ROOT/'mosei_audio_prefetch_status.json',dict(stage='completed' if len(selected)==len(ids) else 'pilot_completed',out=str(args.out),total=len(selected)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('D:/idea_final_overflow/data/mosei_audio_baseen_20261112'))
    p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
