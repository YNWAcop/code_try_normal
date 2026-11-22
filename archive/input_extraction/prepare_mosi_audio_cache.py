"""Preserve current-clip MOSI waveform provenance for repeatable ASR text caching."""
from pathlib import Path
import zipfile,json,os,time,wave,subprocess,re,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from mosei_vsr_common import save,sha
ROOT=Path(__file__).resolve().parent
OUT=Path('D:/idea_final_overflow/data/mosi_audio_baseen_20261112')
RAW=ROOT.parent/'ugsr_data_20261028';ARCHIVE=RAW/'MOSI_Raw.zip';FFMPEG=RAW/'ffmpeg.exe'
IDS=Path('E:/UserData/Desktop/idea_final/data/processed/ebmc_features_20261111/all_author/mosi_features.npz')

def get_ids():
    with np.load(IDS) as z:return np.concatenate([z[s+'_ids'] for s in ('train','valid','test')]).astype(str)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('wav','metadata','tmp'):(OUT/name).mkdir(exist_ok=True)
    ids=get_ids();save(OUT/'protocol.json',dict(source_sha256=sha(Path(__file__)),archive=str(ARCHIVE),
        ffmpeg_sha256=sha(FFMPEG),ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        input='current-clip waveform only; ZIP CRC verified; no text or labels opened'))
    start=time.time();errors=[];done=0
    with zipfile.ZipFile(ARCHIVE) as archive:
        def one(i):
            uid=str(ids[i]);v,c=uid.split('$_$')
            assert re.fullmatch(r'[A-Za-z0-9_-]+',v) and c.isdecimal()
            member=f'Raw/{v}/{c}.mp4';info=archive.getinfo(member)
            target=OUT/'wav'/f'{i:05d}.wav';meta=OUT/'metadata'/f'{i:05d}.json';tmp=OUT/'tmp'/f'{i:05d}.mp4'
            if meta.exists() and target.exists():
                m=json.loads(meta.read_text())
                if m['id']==uid and m['source_crc32']==info.CRC and m['wav_sha256']==sha(target):return
            media=archive.read(member)
            try:
                tmp.write_bytes(media)
                run=subprocess.run([str(FFMPEG),'-nostdin','-loglevel','error','-threads','1','-i',str(tmp),'-vn','-ac','1','-ar','16000','-f','s16le','pipe:1'],capture_output=True,timeout=120)
                if run.returncode or not run.stdout or len(run.stdout)%2:raise RuntimeError('Invalid decoded PCM')
                wav=target.with_suffix('.tmp.wav')
                with wave.open(str(wav),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(run.stdout)
                os.replace(wav,target)
                save(meta,dict(id=uid,index=i,member=member,source_crc32=info.CRC,source_video_sha256=hashlib.sha256(media).hexdigest(),wav_sha256=sha(target),samples=len(run.stdout)//2))
            finally:
                if tmp.exists():tmp.unlink()
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks={pool.submit(one,i):i for i in range(len(ids))}
            for future in as_completed(tasks):
                try:future.result()
                except Exception as exc:errors.append(dict(index=tasks[future],error=type(exc).__name__+': '+str(exc)[:200]))
                done+=1
                if done%100==0 or done==len(ids):
                    state=dict(stage='running',pid=os.getpid(),out=str(OUT),done=done,total=len(ids),failed=len(errors),heartbeat=time.time())
                    save(ROOT/'mosi_audio_cache_status.json',state);print(state,flush=True)
    save(OUT/'download_summary.json',dict(errors=errors,seconds=time.time()-start))
    if errors:
        save(ROOT/'mosi_audio_cache_status.json',dict(stage='failed_preserved',out=str(OUT),errors=errors));raise RuntimeError('Preserve audio failures')
    save(OUT/'completed.json',dict(total=len(ids),time=time.time()))
    save(ROOT/'mosi_audio_cache_status.json',dict(stage='completed',out=str(OUT),total=len(ids)))

if __name__=='__main__':main()
