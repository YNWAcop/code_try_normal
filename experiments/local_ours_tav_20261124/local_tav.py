"""Local coordinator for the isolated full C5+ERU TAV extension."""
import argparse,csv,hashlib,json,os,platform,shutil,subprocess,sys,time,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent
METRICS=('Acc7','Acc5','Acc2_nonzero','F1_weighted_nonzero','MAE','Corr')
COUNTS={'mosi':(1284,229,686),'mosei':(16326,1871,4659)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for attempt in range(30):
        try:os.replace(tmp,p);break
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def code_check():
    for n,h in read(P/'release/runtime_manifest.json')['files'].items():
        assert sha(P/n)==h,('Source changed',n)
    return sha(P/'release/runtime_manifest.json')
def state(a,stage,**kw):save(a.run_root/'queue_status.json',dict(stage=stage,time=time.time(),queue_pid=os.getpid(),smoke=a.smoke,**kw))
def check(a):
    import numpy as np,torch,sklearn,joblib
    source=code_check();a.run_root.mkdir(parents=True,exist_ok=True)
    expected=read(P/'expected_data.json')['files'];sources={}
    for ds in COUNTS:
        ids={};groups={}
        for i,split in enumerate(('train','valid','test')):
            p=a.data_root/'ready'/ds/(split+'.npz');h=sha(p)
            if not a.smoke:assert h==expected[ds+'/'+split+'.npz']['sha256'],('Cache mismatch',p)
            with np.load(p,allow_pickle=False) as z:
                assert set(('T','A_native','A_words','V_native','V_lip','V_words','V_expression','V_quality','V_timeline','ids','groups','y')).issubset(z.files)
                ids[split]=[str(x).replace('$_$','_') for x in z['ids']];groups[split]=set(str(x).replace('$_$','_') for x in z['groups'])
                assert len(ids[split])==len(set(ids[split]))
                if not a.smoke:assert len(ids[split])==COUNTS[ds][i]
            sources[ds+'/'+split]=dict(sha256=h,n=len(ids[split]))
        for s,t in [('train','valid'),('train','test'),('valid','test')]:assert not set(ids[s])&set(ids[t]) and not groups[s]&groups[t]
        assert len(groups['train'])>=8
    runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,device='cpu' if a.smoke else 'cuda')
    identity=dict(source_sha=source,sources=sources,runtime=runtime,smoke=a.smoke,conditions=['T','TA','TV','TAV'],primary='TAV',input_mode='enhanced',backbone='C5+full ERU')
    cp=a.run_root/'input_contract.json'
    if cp.exists():assert read(cp)==identity,'Source/data/runtime changed; keep old artifacts and inspect'
    else:save(cp,identity)
    # A fresh full run retains main and crossfit feature caches. Never clear old experiments.
    minimum=.1 if a.smoke else (3. if list((a.run_root/'runs').glob('*/fold_0/main/C5/contract.json')) else 25.)
    free=shutil.disk_usage(a.run_root).free/1024**3
    report=dict(data_source_passed=True,cuda_available=torch.cuda.is_available(),free_gib=free,required_free_gib=minimum,
                ready=(a.smoke or torch.cuda.is_available()) and free>=minimum,run_root=str(a.run_root),runtime=runtime,
                test_access='file hashes and IDs/groups only; no test labels read')
    save(a.run_root/'preflight.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not report['ready']:raise RuntimeError('Preflight not ready: CUDA or free disk. Need %.1f GiB free; available %.2f GiB'%(minimum,free))
    return sha(cp)
def environment(a):
    env=os.environ.copy();env.update(MSA_DATA_ROOT=str(a.data_root),MSA_RUN_ROOT=str(a.run_root),MSA_INPUT_MODE='enhanced',MSA_BACKBONE='c5',
        MSA_SMOKE='1' if a.smoke else '0',MSA_DEVICE='cpu' if a.smoke else 'cuda',PYTHONPATH=str(P/'src'),PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',PYTHONUTF8='1',
        OMP_NUM_THREADS='2' if a.smoke else '4',OPENBLAS_NUM_THREADS='2' if a.smoke else '4',MKL_NUM_THREADS='2' if a.smoke else '4')
    return env
def execute(a,name,args):
    code_check()
    if shutil.disk_usage(a.run_root).free < (.1 if a.smoke else 3)*1024**3:raise RuntimeError('Insufficient disk reserve')
    logs=a.run_root/'logs';logs.mkdir(exist_ok=True);log=logs/(name+'.log')
    with log.open('a',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,'-u','-m']+args,cwd=P,env=environment(a),stdout=f,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        due=0
        while True:
            code=proc.poll();files=list((a.run_root/'runs').rglob('status.json'))
            latest=max(files,key=lambda p:p.stat().st_mtime) if files else None
            try:child=read(latest) if latest else {}
            except (OSError,ValueError):child={}
            info=dict(job=name,worker_pid=proc.pid,worker_alive=code is None,exit_code=code,log=str(log),child=child,
                      completed=sum((a.run_root/'runs'/ds/'fold_0/ours_done.json').exists() for ds in COUNTS),expected=2)
            state(a,'evaluating' if name=='evaluate' else 'training',**info)
            if time.time()>=due:
                with (a.run_root/'monitor_15min.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(dict(time=time.time(),**info))+'\n')
                due=time.time()+900
            if code is not None:break
            time.sleep(1 if a.smoke else 10)
    if code:raise RuntimeError(f'{name} failed ({code}); see {log}; preserve checkpoints')
def locks(a):
    for ds in COUNTS:
        u=a.run_root/'runs'/ds/'fold_0';done=read(u/'ours_done.json')
        assert sha(u/'selection_lock.json')==done['files']['selection_lock.json']
        lk=read(u/'selection_lock.json');assert lk['policy_sha']==sha(a.run_root/'protocol.json')
        assert lk['manifest_sha']==sha(a.run_root/'manifests'/ds/'fold_0.json')
        for n,h in lk['artifacts'].items():assert sha(u/n)==h,(u,n)
def train(a):
    state(a,'checking');check(a)
    execute(a,'prepare',['evidence_msa.text_present.runner','prepare'])
    for ds in COUNTS:execute(a,ds,['evidence_msa.text_present.train','--dataset',ds])
    locks(a);summarize(a,'valid')
    save(a.run_root/'training_done.json',dict(input_contract_sha=sha(a.run_root/'input_contract.json'),datasets=2))
    evaluated=(a.run_root/'evaluation_done.json').exists()
    state(a,'complete' if evaluated else 'training_complete',completed=2,expected=2,test_opened=evaluated)
def evaluate(a):
    check(a);assert read(a.run_root/'training_done.json')['input_contract_sha']==sha(a.run_root/'input_contract.json');locks(a)
    execute(a,'evaluate',['evidence_msa.text_present.runner','evaluate']);summarize(a,'test')
    save(a.run_root/'evaluation_done.json',dict(input_contract_sha=sha(a.run_root/'input_contract.json'),test_opened=True))
    export(a);state(a,'complete',completed=2,expected=2,test_opened=True)
def summarize(a,split):
    import numpy as np
    sys.path.insert(0,str(P/'src'));from evidence_msa.common.metrics import scores
    locks(a);rows=[];audit=[]
    for ds in COUNTS:
        u=a.run_root/'runs'/ds/'fold_0';preds={}
        if split=='valid':
            p=u/'validation_predictions.npz'
            with np.load(p,allow_pickle=False) as z:
                ids,y=z['ids'],z['y'];preds={m:{c:z[prefix+c] for c in ('T','TA','TV','TAV')} for m,prefix in [('C5','c5_'),('Ours','final_')]}
            paths=[p]
        else:
            dest=a.run_root/'evaluation'/ds;done=read(dest/'done.json')
            assert done['selection_lock_sha']==sha(u/'selection_lock.json') and done['test_sha']==sha(a.data_root/'ready'/ds/'test.npz')
            paths=[]
            for m,name,key in [('C5','predictions_c5_test.npz','c5_predictions_sha'),('Ours','predictions_test.npz','predictions_sha')]:
                p=dest/name;assert sha(p)==done[key];paths.append(p)
                with np.load(p,allow_pickle=False) as z:
                    if m=='C5':ids,y=z['ids'],z['y']
                    else:assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['y'])
                    preds[m]={c:z['pred_'+c] for c in ('T','TA','TV','TAV')}
        with np.load(a.data_root/'ready'/ds/(split+'.npz'),allow_pickle=False) as z:
            assert np.array_equal(ids,np.array([str(x).replace('$_$','_') for x in z['ids']])) and np.array_equal(y,z['y'])
        for m,pred in preds.items():
            for c,v in pred.items():
                sc=scores(y,v);rows.append(dict(dataset=ds.upper(),method=m+'_TAV_extension',condition=c,n=sc['n_all'],**{k:sc['metrics'][k] for k in METRICS}))
        audit.extend(dict(path=str(p.relative_to(a.run_root)),sha256=sha(p)) for p in paths)
    dest=a.run_root/'tables';dest.mkdir(exist_ok=True)
    def write(name,values):
        with (dest/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    write(split+'_all_diagnostics.csv',rows)
    write(split+'_TAV_results.csv',[r for r in rows if r['condition']=='TAV'])
    write('Ours_TAV_'+split+'_results.csv',[r for r in rows if r['condition']=='TAV' and r['method'].startswith('Ours')])
    save(dest/(split+'_audit.json'),dict(predictions=audit,primary_rows=2,comparator_rows=2,smoke=a.smoke,historical_test_exposure=True,independent_confirmation=False))
def export(a):
    assert read(a.run_root/'evaluation_done.json')['input_contract_sha']==sha(a.run_root/'input_contract.json')
    summarize(a,'test');p=a.run_root/('SMOKE_Ours_TAV_results.zip' if a.smoke else 'Ours_TAV_results.zip')
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for f in a.run_root.rglob('*'):
            if f.is_file() and (f.suffix in ('.json','.csv') or f.name in ('validation_predictions.npz','predictions_test.npz','predictions_c5_test.npz')):z.write(f,f.relative_to(a.run_root))
    p.with_suffix('.zip.sha256').write_bytes((sha(p)+'  '+p.name+'\n').encode('ascii'));print(p,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','train','evaluate','export','status'])
    p.add_argument('--data-root',type=Path,default=Path('D:/MSA_Cloud_20261115/data'))
    p.add_argument('--run-root',type=Path,default=P/'outputs/formal');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.data_root=a.data_root.resolve();a.run_root=a.run_root.resolve()
    if a.smoke:assert 'smoke' in a.run_root.name,'Smoke output name must contain smoke'
    if a.action=='status':
        d=read(a.run_root/'queue_status.json');d['seconds_since_update']=time.time()-d['time'];print(json.dumps(d,indent=2,ensure_ascii=False));return
    a.run_root.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(P/'src'))
    from evidence_msa.common.checkpoint import lock
    with lock(a.run_root/'coordinator.lock'):
        try:
            if a.action=='check':check(a)
            elif a.action=='train':train(a)
            elif a.action=='evaluate':evaluate(a)
            else:check(a);export(a)
        except Exception as e:
            state(a,'failed',action=a.action,error=repr(e));raise
if __name__=='__main__':main()
