"""Explicit crop-boundary extension for clips rejected by upstream bias guard.

Original successful caches and all original source files remain unchanged.
Each recovered cache records the extension hash in addition to the base hash.
No dataset text, audio or sentiment labels are read.
"""
import argparse,json,os,sys,time
from pathlib import Path
import extract_mosei_visual_speech_hybrid as base
from preparation.detectors.mediapipe import video_process as vp
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mosei_vsr'
ORIGINAL=vp.cut_patch
COUNT=0
CLAMPED=0
def bounded_cut(img,landmarks,height,width,threshold=5):
    global COUNT,CLAMPED
    try:return ORIGINAL(img,landmarks,height,width,threshold)
    except OverflowError:
        center=np.mean(landmarks,axis=0)
        if not np.isfinite(center).all():raise ValueError('Nonfinite mouth center')
        h,w=img.shape[:2]
        if h<2*height or w<2*width:raise ValueError('Warped frame too small')
        x=int(round(np.clip(center[0],width,w-width)))
        y=int(round(np.clip(center[1],height,h-height)))
        CLAMPED+=int(center[0]<width or center[0]>w-width or center[1]<height or center[1]>h-height)
        crop=img[y-height:y+height,x-width:x+width].copy()
        if crop.shape[:2]!=(2*height,2*width):raise ValueError('Crop shape')
        COUNT+=1
        return crop

def cases():
    found={}
    for path in sorted(OUT.glob('events_*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if event.get('stage')=='mouth_preprocess' and 'OverflowError: too much bias' in event.get('error',''):
                media=Path(event['retained_tmp'])
                if media.exists():found[event['i']]=event
    return found

def main(verify):
    global COUNT,CLAMPED
    os.chdir(ROOT.parent);base.cv2.setNumThreads(1);torch.set_num_threads(2)
    source_sha=base.sha256(Path(__file__))
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    fingerprint=protocol['fingerprint']
    report_path=OUT/'crop_boundary_extension.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else dict(
        base_fingerprint=fingerprint,extension_sha256=source_sha,
        policy='Only on upstream OverflowError, clamp mouth center to full-crop bounds; normal calls remain exact upstream output',
        input='current clip RGB only',records={})
    if report['extension_sha256']!=source_sha or report['base_fingerprint']!=fingerprint:raise ValueError('Extension changed')
    model=None
    tokens=['<blank>']+[s.split()[0] for s in (base.CODE/'spm/unigram/unigram5000_units.txt').read_text(encoding='utf-8').splitlines()]+['<eos>']
    for i,event in cases().items():
        path=OUT/'parts'/f'{i:05d}.npz'
        if not verify and base.cached_ok(path,event['id'],fingerprint):continue
        media=Path(event['retained_tmp'])
        if media.resolve().parent!=(OUT/'tmp').resolve() or base.sha256(media)!=event['media_sha256']:raise ValueError('Retained media identity mismatch')
        COUNT=0;CLAMPED=0;vp.cut_patch=bounded_cut
        chunks,detection,frames=base.mouth_chunks(media)
        vp.cut_patch=ORIGINAL
        info=dict(id=event['id'],frames=frames,bias_guard_fallback_frames=COUNT,actual_center_clamp_frames=CLAMPED,media_sha256=event['media_sha256'],extension_sha256=source_sha)
        if not all(x.shape[1:]==(88,88) for x,ok in chunks):raise ValueError('Invalid crop shape')
        if verify:
            print(json.dumps(info),flush=True);continue
        if model is None:
            weights=ROOT/'vsr_trlrs2lrs3vox2avsp_base.pth'
            if base.sha256(weights)!=protocol['weights_sha256']:raise ValueError('Model hash changed')
            model=base.E2E(len(tokens),'video',ctc_weight=.1)
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
            model.cuda().eval()
        record=base.infer(chunks,detection,frames,model,tokens)
        tmp=path.with_name(f'{i:05d}.boundary.tmp.npz')
        np.savez_compressed(tmp,id=event['id'],fingerprint=fingerprint,preprocess_variant='bounded_crop_extension',extension_sha256=source_sha,media_sha256=event['media_sha256'],**record)
        if not base.cached_ok(tmp,event['id'],fingerprint):raise ValueError('Recovered cache invalid')
        if path.exists():raise FileExistsError('Preserve existing invalid cache for diagnosis')
        os.replace(tmp,path)
        info['feature_sha256']=base.sha256(path)
        report['records'][str(i)]=info;base.atomic_json(report_path,report)
        print(json.dumps(info),flush=True)
    if verify:
        rng=np.random.default_rng(1);img=rng.integers(0,256,(256,256),dtype=np.uint8)
        for x in range(80,177,8):
            for y in range(80,177,8):
                a=ORIGINAL(img,np.array([[x,y]]),48,48)
                np.testing.assert_array_equal(a,bounded_cut(img,np.array([[x,y]]),48,48))
        print('169 ordinary crops byte-identical; failed clips now produce complete sequences',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();main(a.verify)
