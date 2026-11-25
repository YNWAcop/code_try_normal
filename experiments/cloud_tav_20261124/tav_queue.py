"""18 full-TAV jobs, isolated outputs, wait for ablation, finite single-run queue."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent
METHODS=('EBMC','TLRA','API','LNLN','Self_MM','EMT_DLFR','TFR_NET','ALMT','MISA')
OLD=METHODS[:5];FOUR=METHODS[5:9];DS=('mosi','mosei');METRICS=('Acc7','Acc5','Acc2_nonzero','F1_weighted_nonzero','MAE','Corr')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def source_check():
    m=read(P/'manifest.json')
    for n,h in m.items():assert sha(P/n)==h,('Package source changed',n)
    return sha(P/'manifest.json')
def state(a,stage,**kw):save(a.out/'status.json',dict(stage=stage,time=time.time(),**kw))
def env():
    e=os.environ.copy();e.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1');return e
def workspace(a,smoke=False):return a.out/('smoke_workspace' if smoke else 'workspace')
def setup(a,smoke=False):
    w=workspace(a,smoke);w.mkdir(parents=True,exist_ok=True)
    for name,target in [('data',a.root/'data'),('models',a.root/'models'),('baselines/upstream',a.root/'baselines/upstream')]:
        p=w/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.is_symlink():assert p.resolve()==target.resolve()
        elif p.exists():raise RuntimeError('Expected shared-data link, found real path '+str(p))
        else:p.symlink_to(target.resolve(),target_is_directory=True)
    for name in ('baseline_results','baseline_reports','baselines/adapted'): (w/name).mkdir(parents=True,exist_ok=True)
    return w
def check(a):
    signature=source_check()
    import torch,numpy,sklearn,transformers,pandas,einops,yaml,scipy,safetensors,tqdm,easydict,joblib
    assert torch.cuda.is_available(),'CUDA unavailable in selected Python'
    assert shutil.disk_usage(a.out).free>15*1024**3,'Need at least 15 GiB free for new baseline jobs'
    sys.path.insert(0,str(P/'four'));from mmsa_mapped_pickle import load_fields
    sources={};counts={};norm=lambda v:[str(x).replace('$_$','_') for x in v]
    for ds in DS:
        refs={}
        for split,n in zip(('train','valid','test'),(1284,229,686) if ds=='mosi' else (16326,1871,4659)):
            f=a.root/'data/ready'/ds/(split+'.npz')
            with numpy.load(f,allow_pickle=False) as z:refs[split]=set(norm(z['ids']));assert len(refs[split])==len(z['ids'])==n
            sources[str(f)]=sha(f)
        assert not any(refs[x]&refs[y] for x,y in [('train','valid'),('train','test'),('valid','test')])
        for rel in (f'data/mmsa/{ds}/unaligned_50.pkl',f'data/api_aligned/{ds}/aligned_50.pkl'):
            f=a.root/rel;assert f.is_file(),f;sources[str(f)]=sha(f)
            data=load_fields(f,('id',),splits=('train','valid','test'))
            for split in refs:
                ids=norm(data[split]['id']);assert len(ids)==len(refs[split]) and set(ids)==refs[split],(rel,split,'ID mismatch')
        import pickle
        name='CMU'+ds.upper();folder=a.root/'data/ebmc_author'/name
        label=folder/(name+'_features_raw_2way.pkl');sources[str(label)]=sha(label)
        with label.open('rb') as f:o=pickle.load(f,encoding='latin1')
        assert len(o)==7
        for split,vs in zip(('train','valid','test'),o[4:7]):
            ids=[u for v in vs for u in o[0][v]];assert len(ids)==len(refs[split]) and set(norm(ids))==refs[split],(name,split)
        for feat in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT'):
            froot=folder/'features'/feat;assert froot.is_dir(),froot
            for v in o[0].values():
                for uid in v:assert (froot/(str(uid)+'.npy')).is_file() or (froot/str(uid)).is_dir(),(feat,uid)
        counts[ds]={s:len(v) for s,v in refs.items()}
    bert=a.root/'models/bert-base-uncased'
    for n in ('config.json','vocab.txt'):assert (bert/n).is_file()
    assert any((bert/n).is_file() for n in ('model.safetensors','pytorch_model.bin'))
    for method in OLD:
        directory=a.root/'baselines/upstream'/method;assert directory.is_dir(),directory
        for f in directory.rglob('*'):
            if f.is_file() and f.suffix in ('.py','.json','.yaml','.yml'):sources[str(f)]=sha(f)
    identity={'source_manifest':signature,'inputs':sources,'jobs':[[m,d,'TAV'] for m in METHODS for d in DS]}
    contract=a.out/'contract.json'
    if contract.exists():assert read(contract)==identity,'Source/data identity changed; preserve old output and inspect'
    else:save(contract,identity)
    report={'passed':True,'counts':counts,'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'sklearn':sklearn.__version__,'test_usage':'IDs and file hashes checked; no test predictions or selection','jobs':18}
    save(a.out/'preflight.json',report);print(json.dumps(report),flush=True)
def run(a,cmd,label):
    source_check();log=a.out/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        p=subprocess.Popen([sys.executable,'-u']+list(map(str,cmd)),cwd=P,env=env(),stdout=f,stderr=subprocess.STDOUT);last=0
        while p.poll() is None:
            state(a,'running',task=label,worker_pid=p.pid,log=str(log))
            if time.time()-last>=900:
                with (a.out/'monitor_15min.jsonl').open('a') as h:h.write(json.dumps(read(a.out/'status.json'))+'\n')
                last=time.time()
            time.sleep(10)
    if p.returncode:raise RuntimeError(f'{label} failed ({p.returncode}); inspect {log}. No automatic restart from epoch 1.')
def ablation_stage(a):
    f=a.ablation/'outputs/formal/status.json'
    if not f.exists():raise FileNotFoundError('Ablation status missing: '+str(f))
    value=read(f)
    if value.get('smoke'):raise RuntimeError('Cannot use synthetic ablation as prerequisite')
    if value['stage']=='failed':raise RuntimeError('Ablation failed; TAV will not start')
    return value['stage']
def wait_ablation(a):
    while True:
        stage=ablation_stage(a)
        if stage=='complete':return
        if stage=='training_complete':
            # Complete the authorized ablation task before taking over its GPU.
            run(a,[a.ablation/'ablation.py','evaluate','--data-root',a.root/'data','--run-root',a.ablation/'outputs/formal'],'finish_ablation_test')
            run(a,[a.ablation/'ablation.py','export','--run-root',a.ablation/'outputs/formal'],'export_ablation');continue
        state(a,'waiting_ablation',ablation_stage=stage);time.sleep(30)
def folder(a,m,d,smoke=False):
    if m in OLD:return workspace(a,smoke)/'baseline_results'/(m+'_'+d+'_TAV'+('_smoke' if smoke else ''))
    if m in FOUR:return a.out/('smoke_four' if smoke else 'four')/(m+'_'+d+'_TAV')
    raise ValueError(m)
def commands(a,m,d,smoke=False):
    w=workspace(a,smoke);f=folder(a,m,d,smoke);extra=['--smoke'] if smoke else []
    if m in OLD:
        script={'API':'adapter_api_tlra.py','TLRA':'adapter_api_tlra.py','EBMC':'adapter_ebmc.py','LNLN':'adapter_lnln.py','Self_MM':'adapter_selfmm.py'}[m]
        cmd=[P/'hand'/script,'--root',w,'--dataset',d,'--condition','TAV']
        if m!='EBMC':cmd+=['--protocol','whole','--rate','0']
        if m in ('API','TLRA'):cmd+=['--method',m]
        return cmd+extra
    if m in FOUR:
        script='adapter_emt_tfr.py' if m in ('EMT_DLFR','TFR_NET') else 'adapter_misa_almt.py'
        cmd=[P/'four'/script,'--method',m,'--dataset',d,'--protocol','whole','--condition','TAV','--rate','0','--out',f,'--data-root',w]
        if m=='EMT_DLFR':cmd+=['--batch-size',str(2 if smoke else (32 if d=='mosi' else 16)),'--activation-checkpointing']
        elif m=='TFR_NET':cmd+=['--batch-size',str(2 if smoke else 16)]
        elif m=='MISA':cmd+=['--gradient-checkpointing']
        return cmd+extra
    raise ValueError(m)

def locked(a,m,d,smoke=False):
    f=folder(a,m,d,smoke);p=f/'validation_lock.json'
    if not p.exists():return False
    v=read(p)
    assert set(v['condition'].upper())==set('TAV') and v['smoke']==smoke and v['dataset']==d,(f,'Wrong validation lock')
    assert v['status']=='validation_locked_no_test_evaluated',(f,'Unlocked checkpoint')
    if (f/'tav_checkpoint_lock.json').exists():
        for n,h in read(f/'tav_checkpoint_lock.json').items():assert sha(f/n)==h,(f,n)
    else:
        weight='stage2_best.pt' if m=='EBMC' else 'best.pt';assert (f/weight).is_file()
        if v.get('checkpoint_sha256'):assert sha(f/weight)==v['checkpoint_sha256']
        save(f/'tav_checkpoint_lock.json',{p.name:sha(p),weight:sha(f/weight)})
    return True
def train(a):
    wait_ablation(a);check(a);setup(a);setup(a,True)
    for m in METHODS:
        for d in DS:
            if shutil.disk_usage(a.out).free<10*1024**3:raise RuntimeError('Disk reserve <10 GiB; no files deleted')
            for smoke in (True,False):
                if locked(a,m,d,smoke):continue
                f=folder(a,m,d,smoke)
                if f.exists():raise RuntimeError('Incomplete non-resumable job preserved: '+str(f))
                run(a,commands(a,m,d,smoke),m+'_'+d+('_smoke' if smoke else '_train'))
                assert locked(a,m,d,smoke)
            f=folder(a,m,d)
            if m in OLD and not (f/'metrics_full_valid.json').exists():
                run(a,[P/'hand/evaluate_valid.py','--root',workspace(a),'--handoff',P/'hand','--job',f.name,'--device','cuda'],m+'_'+d+'_valid_metrics')
            save(a.out/'progress.json',{'locked':sum(locked(a,x,y) for x in METHODS for y in DS),'expected':18,'last':m+'_'+d})
    state(a,'training_complete',completed=18,expected=18,test_opened=False)
def evaluate(a):
    assert all(locked(a,m,d) for m in METHODS for d in DS),'Train all 18 jobs first'
    for m in METHODS:
        for d in DS:
            for split in ('valid','test'):
                run(a,[P/'evaluate.py','--workspace',workspace(a),'--run',folder(a,m,d),'--method',m,'--dataset',d,'--split',split],m+'_'+d+'_'+split)
    summarize(a);state(a,'complete',completed=18,expected=18,test_opened=True)
def summarize(a):
    import numpy as np
    from metrics_common import scores
    rows=[]
    for m in METHODS:
        for d in DS:
            f=folder(a,m,d);p=f/'evaluation/test_predictions.npz'
            record=read(f/'evaluation/test_report.json')
            assert record['predictions_sha256']==sha(p)
            with np.load(p,allow_pickle=False) as z, np.load(a.root/'data/ready'/d/'test.npz',allow_pickle=False) as ref:
                norm=lambda v:[str(x).replace('$_$','_') for x in v]
                ids=norm(z['ids']);expected=norm(ref['ids']);assert len(set(ids))==len(ids)==len(expected) and set(ids)==set(expected)
                index={v:i for i,v in enumerate(ids)};np.testing.assert_allclose(z['y'].reshape(-1)[[index[v] for v in expected]],ref['y'].reshape(-1),rtol=0,atol=1e-6)
                metrics=scores(z['y'],z['pred'])['metrics']
                rows.append(dict(method=m,dataset=d,condition='TAV',**{k:metrics[k] for k in METRICS},n=len(ids),prediction_sha256=sha(p)))
    with (a.out/'TAV_test_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def export(a):
    assert read(a.out/'status.json')['stage']=='complete'
    target=a.out/'TAV_results.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for root,dirs,files in os.walk(a.out,followlinks=False):
            dirs[:]=[n for n in dirs if n not in ('source','adapted','upstream','models','data','__pycache__','smoke_workspace','smoke_four','smoke') and not Path(root,n).is_symlink()]
            for n in files:
                p=Path(root,n)
                if p.suffix in ('.json','.csv','.npz') and p!=target:z.write(p,p.relative_to(a.out))
    target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='ascii');print(target)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','run','status','export']);p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/MSA_Cloud_20261115'));p.add_argument('--out',type=Path,default=P/'outputs');p.add_argument('--ablation',type=Path,default=Path('/root/autodl-tmp/cloud_ablation_20261123'));a=p.parse_args()
    a.root=a.root.resolve();a.out=a.out.resolve();a.ablation=a.ablation.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.action=='status':print(json.dumps(read(a.out/'status.json'),ensure_ascii=False,indent=2));return
    if a.action=='check':check(a);return
    if a.action=='export':export(a);return
    import fcntl
    with (a.out/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (a.out/'status.json').exists() and read(a.out/'status.json')['stage']=='complete':
                assert all(locked(a,m,d) for m in METHODS for d in DS)
                summarize(a);export(a);return
            train(a);evaluate(a);export(a)
        except Exception as e:
            save(a.out/'error.json',{'error':repr(e),'time':time.time()});state(a,'failed',error=repr(e));raise
if __name__=='__main__':main()
