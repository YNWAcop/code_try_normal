"""Frozen external facial-expression features, using RGB video only (no audio).

Eight uniformly spaced frames per utterance. Largest detected face, square
crop with 15% margin. Failed detections are explicitly masked, never filled
with text/audio or ground-truth emotion. Uses official EmotiEffLib ONNX model.
"""
from pathlib import Path
import sys,json,zipfile,hashlib,time,os
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import cv2,numpy as np,onnx,onnxruntime as ort
from onnx import helper,TensorProto,numpy_helper

def make_session(threads=3):
    model=onnx.load(str(ROOT/'enet_b0_8_best_afew.onnx'))
    node=model.graph.node[-1];assert node.op_type=='Gemm'
    weights={t.name:numpy_helper.to_array(t) for t in model.graph.initializer}
    w,b=weights[node.input[1]],weights[node.input[2]]
    feat=node.input[0];model.graph.node.remove(node)
    model.graph.output.append(helper.make_tensor_value_info(feat,TensorProto.FLOAT,[None,w.shape[1]]))
    del model.graph.output[0]
    opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1;opts.log_severity_level=3
    return ort.InferenceSession(model.SerializeToString(),opts,providers=['CPUExecutionProvider']),w,b

def frames_from_clip(path):
    cap=cv2.VideoCapture(str(path.relative_to(ROOT)));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indexes=np.rint(np.linspace(0,max(0,count-1),8)).astype(int)
    detector=cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx','',(320,320),.75,.3,5000)
    crops=[];quality=[];boxes=[]
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,frame=cap.read()
        if not ok:
            crops.append(np.zeros((3,224,224),np.float32));quality.append([0.,0.,0.]);boxes.append([-1]*4);continue
        h,w=frame.shape[:2];ratio=320/max(h,w);small=cv2.resize(frame,(max(1,round(w*ratio)),max(1,round(h*ratio))))
        detector.setInputSize((small.shape[1],small.shape[0]));_,faces=detector.detect(small)
        if faces is None:
            crops.append(np.zeros((3,224,224),np.float32));quality.append([0.,0.,0.]);boxes.append([-1]*4);continue
        face=max(faces,key=lambda f:f[2]*f[3]);x,y,fw,fh=face[:4]/ratio
        cx,cy=x+fw/2,y+fh/2;side=max(fw,fh)*1.15
        x0,y0=max(0,int(cx-side/2)),max(0,int(cy-side/2));x1,y1=min(w,int(cx+side/2)),min(h,int(cy+side/2))
        crop=frame[y0:y1,x0:x1];sharp=cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
        rgb=cv2.cvtColor(cv2.resize(crop,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        inp=((rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)
        crops.append(inp);quality.append([float(face[-1]),float(sharp),float(fw*fh/(w*h))]);boxes.append([x0,y0,x1,y1])
    cap.release();return np.stack(crops),np.asarray(quality,np.float32),np.asarray(boxes),indexes

def main():
    os.chdir(ROOT)
    cv2.setNumThreads(1)
    raw=Path(r'E:\UserData\Documents\ChatGPT\科研经历\ugsr_data_20261028\MOSI_Raw.zip')
    cache=Path(r'E:\UserData\Desktop\idea_final\data\processed\ebmc_features_20261111\all_author\mosi_features.npz')
    out=ROOT/'rgb_mosi';out.mkdir(exist_ok=True);parts=out/'parts';parts.mkdir(exist_ok=True)
    media=out/'media';media.mkdir(exist_ok=True)
    session,w,b=make_session()
    with np.load(cache) as z:
        ids=np.concatenate([z[f'{s}_ids'] for s in ('train','valid','test')])
    report=dict(model='EmotiEffLib enet_b0_8_best_afew',upstream='https://github.com/sb-ai-lab/EmotiEffLib',
        weights_sha256=hashlib.sha256((ROOT/'enet_b0_8_best_afew.onnx').read_bytes()).hexdigest(),
        detector_sha256=hashlib.sha256((ROOT/'face_detection_yunet_2023mar.onnx').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frames=8,classes=['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'],
        input='RGB frames only; no audio, ASR, dataset text or labels used for extraction',
        external_supervision='upstream pretrained expression model; must disclose extra pretraining relative to fixed features',
        extraction='largest detected face; 15% crop margin; failed detection zero feature + mask; no manual test selection')
    (out/'protocol.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    start=time.time();features=[];qualities=[];logits=[];valid=[]
    with zipfile.ZipFile(raw) as archive:
        names={n.replace('\\','/'):n for n in archive.namelist()}
        for i,ident in enumerate(ids):
            video,clip=ident.rsplit('$_$',1);part=parts/f'{i:05d}.npz'
            if not part.exists():
                member=names[f'Raw/{video}/{clip}.mp4'];path=media/f'{i:05d}.mp4'
                path.write_bytes(archive.read(member))
                crops,q,boxes,indexes=frames_from_clip(path)
                good=q[:,0]>0;feat=np.zeros((8,w.shape[1]),np.float32)
                if good.any():feat[good]=session.run(None,{session.get_inputs()[0].name:crops[good]})[0]
                rawlogit=(feat@w.T+b)*good[:,None]
                np.savez_compressed(part,id=ident,features=feat,quality=q,logits=rawlogit,valid=good,boxes=boxes,frame_indices=indexes)
                # The decoded source is reproducible from immutable archive; keep files for auditing.
            with np.load(part) as p:
                assert str(p['id'])==ident
                features.append(p['features']);qualities.append(p['quality']);logits.append(p['logits']);valid.append(p['valid'])
            if (i+1)%50==0:print(i+1,len(ids),'seconds',round(time.time()-start),flush=True)
    np.savez_compressed(out/'features.npz',ids=ids,features=np.stack(features),quality=np.stack(qualities),logits=np.stack(logits),valid=np.stack(valid))
    report.update(clips=len(ids),valid_faces=int(np.sum(valid)),total_frames=len(ids)*8,clips_without_face=int((~np.any(valid,axis=1)).sum()),elapsed=time.time()-start)
    (out/'completion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETED',report,flush=True)

if __name__=='__main__':main()
