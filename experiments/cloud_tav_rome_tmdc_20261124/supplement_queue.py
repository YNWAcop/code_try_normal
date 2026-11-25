"""Four fixed TAV jobs, validated cached features, serial GPU, frozen test export."""
import argparse,csv,json,os,pickle,shutil,subprocess,sys,time,zipfile
from pathlib import Path
from metrics_common import atomic_json as save,digest as sha,scores
P=Path(__file__).resolve().parent
JOBS=[(m,d) for m in ('RoME','TMDC') for d in ('mosi','mosei')]
MET=('Acc7','Acc5','Acc2_nonzero','F1_weighted_nonzero','MAE','Corr')
FEATURES=('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def norm(v):return [str(x).replace('$_$','_') for x in v]
def source_check():
    for n,h in read(P/'manifest.json').items():assert sha(P/n)==h,('Source changed',n)
    return sha(P/'manifest.json')
def state(a,stage,**kw):save(a.out/'status.json',dict(stage=stage,time=time.time(),**kw))
def unit(a,m,d,smoke=False):return a.out/m/('smoke' if smoke else 'main')/(d+'_TAV')
def locked(a,m,d,smoke=False):
    u=unit(a,m,d,smoke);s=u/'selection_lock.json'
    if not s.exists():return False
    x=read(s);assert x['checkpoint_sha256']==sha(u/'best.pt') and x['protocol_sha256']==sha(u/'protocol.json')
    return True
def check(a):
    import numpy as np,torch,pandas,einops,tqdm,sklearn
    assert torch.cuda.is_available(),'Selected Python has no CUDA'
    assert shutil.disk_usage(a.out).free>12*1024**3,'Need >12 GiB free; nothing deleted'
    assert read(a.previous/'status.json')['stage']=='complete','Previous 18-job queue not complete'
    old=read(a.previous/'contract.json')['inputs'];inputs={};counts={}
    for ds in ('mosi','mosei'):
        name='CMU'+ds.upper();root=a.data_root/name;lab=root/(name+'_features_raw_2way.pkl')
        inputs[str(lab)]=sha(lab)
        assert old[str(lab)]==inputs[str(lab)],'Label archive differs from previous queue'
        with lab.open('rb') as f:obj=pickle.load(f,encoding='latin1')
        seen=set();counts[ds]={}
        for split,vs,n in zip(('train','valid','test'),obj[4:7],(1284,229,686) if ds=='mosi' else (16326,1871,4659)):
            ids=norm([u for v in vs for u in obj[0][v]])
            assert len(ids)==len(set(ids))==n and not seen.intersection(ids);seen.update(ids)
            ref=a.ready/ds/(split+'.npz');inputs[str(ref)]=sha(ref);assert old[str(ref)]==inputs[str(ref)]
            with np.load(ref,allow_pickle=False) as z:assert set(ids)==set(norm(z['ids']))
            counts[ds][split]=n
        for feat in FEATURES:
            for vs in obj[0].values():
                for uid in vs:
                    f=root/'features'/feat/(str(uid)+'.npy');folder=root/'features'/feat/str(uid)
                    candidates=[f] if f.is_file() else sorted(folder.glob('*.npy'))
                    assert candidates,('Missing feature',f)
                    for p in candidates:inputs[str(p)]=sha(p)
    ident=dict(source_manifest=source_check(),inputs=inputs,jobs=[list(j) for j in JOBS],gpu_cap_gib=a.gpu_cap_gib)
    cp=a.out/'contract.json'
    if cp.exists():assert read(cp)==ident,'Source/data/config changed; inspect before resuming'
    else:save(cp,ident)
    report=dict(passed=True,jobs=4,counts=counts,torch=torch.__version__,numpy=np.__version__,test_usage='File hashes and split IDs; no test predictions or test-driven selection')
    save(a.out/'preflight.json',report);print(json.dumps(report),flush=True)
def command(a,m,d,stage):
    script=P/('rome/rome_adapter.py' if m=='RoME' else 'tmdc/tmdc_worker.py')
    cmd=[sys.executable,'-u',str(script),stage,'--dataset',d,'--condition','TAV','--data-root',str(a.data_root),'--output-root',str(a.out/m),'--device','cuda']
    if m=='TMDC':cmd+=['--gpu-cap-gib',str(a.gpu_cap_gib)]
    return cmd
def run(a,m,d,stage):
    source_check();assert shutil.disk_usage(a.out).free>6*1024**3,'Disk reserve <6 GiB'
    log=a.out/'logs'/f'{m}_{d}_{stage}.log';log.parent.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with log.open('a',encoding='utf-8') as f:
        child=subprocess.Popen(command(a,m,d,stage),cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT);due=0
        while child.poll() is None:
            state(a,'running',task=f'{m}_{d}_{stage}',worker_pid=child.pid,log=str(log),completed=sum(locked(a,x,y) for x,y in JOBS),expected=4)
            if time.time()>=due:
                with (a.out/'monitor_15min.jsonl').open('a') as h:h.write(json.dumps(read(a.out/'status.json'))+'\n')
                due=time.time()+900
            time.sleep(10)
    assert child.returncode==0,f'{m}/{d}/{stage} failed; inspect {log}; preserved checkpoints'
def export(a):
    import numpy as np
    assert all(locked(a,m,d) for m,d in JOBS)
    rows=[]
    for m,d in JOBS:
        u=unit(a,m,d);done=read(u/'test_done.json');p=u/'predictions_test.npz'
        assert done['predictions_sha256']==sha(p) and done['selection_sha256']==sha(u/'selection_lock.json')
        with np.load(p,allow_pickle=False) as z,np.load(a.ready/d/'test.npz',allow_pickle=False) as ref:
            ids=norm(z['ids']);expected=norm(ref['ids']);assert len(ids)==len(set(ids))==len(expected) and set(ids)==set(expected)
            ix={x:i for i,x in enumerate(ids)};np.testing.assert_allclose(z['y'].reshape(-1)[[ix[x] for x in expected]],ref['y'].reshape(-1),atol=1e-6,rtol=0)
            metrics=scores(z['y'],z['pred'])['metrics']
        rows.append(dict(method=m,dataset=d,condition='TAV',**{k:metrics[k] for k in MET},n=len(ids),prediction_sha256=sha(p)))
    with (a.out/'TAV_RoME_TMDC_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    state(a,'complete',completed=4,expected=4,test_opened=True)
    target=a.out/'TAV_RoME_TMDC_results.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in a.out.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.csv','.npz') and 'smoke' not in p.relative_to(a.out).parts:z.write(p,p.relative_to(a.out))
    target.with_suffix('.zip.sha256').write_bytes((sha(target)+'  '+target.name+'\n').encode('ascii'));print(target)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','run','status','export']);p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/MSA_Cloud_20261115/data/ebmc_author'));p.add_argument('--ready',type=Path,default=Path('/root/autodl-tmp/MSA_Cloud_20261115/data/ready'));p.add_argument('--previous',type=Path,default=Path('/root/autodl-tmp/cloud_tav_20261124/outputs'));p.add_argument('--out',type=Path,default=P/'outputs');p.add_argument('--gpu-cap-gib',type=float,default=20);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='status':print(json.dumps(read(a.out/'status.json'),indent=2));return
    if a.action=='check':check(a);return
    import fcntl
    with (a.out/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.action=='export':export(a);return
            state(a,'checking');check(a)
            for m,d in JOBS:
                if locked(a,m,d):continue
                if not locked(a,m,d,True):run(a,m,d,'smoke')
                if m=='TMDC':run(a,m,d,'profile')
                run(a,m,d,'train');assert locked(a,m,d)
            for m,d in JOBS:run(a,m,d,'test')
            export(a)
        except Exception as e:state(a,'failed',error=repr(e));raise
if __name__=='__main__':main()
