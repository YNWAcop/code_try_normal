"""Five predeclared ablations; three pipelines share their C5 comparators."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
FAMILIES={'native_full':('native','c5'),'enhanced_full':('enhanced','c5'),'enhanced_plain':('enhanced','simple')}
VARIANTS={'A0':('native_full','c5'),'A1':('enhanced_full','c5'),'A2':('native_full','final'),'A3':('enhanced_full','final'),'A4':('enhanced_plain','c5')}
METRICS=('Acc7','Acc5','Acc2_nonzero','F1_weighted_nonzero','MAE','Corr')
COUNTS={'mosi':{'train':1284,'valid':229,'test':686},'mosei':{'train':16326,'valid':1871,'test':4659}}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def code_check():
    m=read(HERE/'release/runtime_manifest.json')
    for n,h in m['files'].items():
        if sha(HERE/n)!=h:raise RuntimeError('Package source changed: '+n)
    return sha(HERE/'release/runtime_manifest.json')
def check(a):
    import numpy as np,torch,sklearn,joblib
    from threadpoolctl import threadpool_limits
    digest=code_check();result={'source_sha':digest,'smoke':a.smoke,'torch':torch.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'cuda_available':torch.cuda.is_available(),'sources':{}}
    if not a.smoke and not torch.cuda.is_available():raise RuntimeError('Formal training requires CUDA; use the existing cloud CUDA Python environment')
    expected={} if a.smoke else read(HERE/'expected_data.json')['files']
    for ds in COUNTS:
        ids={};groups={}
        for split in ('train','valid','test'):
            p=a.data_root/'ready'/ds/(split+'.npz')
            if not p.exists():raise FileNotFoundError('Required ready feature cache: '+str(p))
            h=sha(p);key=ds+'/'+split+'.npz'
            if not a.smoke and h!=expected[key]['sha256']:raise RuntimeError('Input differs from verified local cache: '+str(p))
            with np.load(p,allow_pickle=False) as z:
                required=('T','A_native','A_words','V_native','V_lip','V_words','V_expression','V_quality','V_timeline','ids','groups','y')
                assert all(k in z.files for k in required),(key,z.files)
                ids[split]=[str(v).replace('$_$','_') for v in z['ids']];groups[split]=set(str(v).replace('$_$','_') for v in z['groups'])
                assert len(ids[split])==len(set(ids[split]))
                if not a.smoke:assert len(ids[split])==COUNTS[ds][split]
            result['sources'][key]={'sha256':h,'samples':len(ids[split])}
        for x,y in [('train','valid'),('train','test'),('valid','test')]:
            assert not set(ids[x])&set(ids[y]);assert not groups[x]&groups[y]
        assert len(groups['train'])>=8,'At least eight training video groups required'
    result.update(passed=True,test_usage='File hash and IDs/groups only; test labels not loaded by check')
    a.run_root.mkdir(parents=True,exist_ok=True)
    frozen=a.run_root/'input_contract.json'
    identity={'source_sha':digest,'smoke':a.smoke,'sources':result['sources']}
    if frozen.exists():assert read(frozen)==identity,'Source/data/smoke mode changed; use a new output directory'
    else:save(frozen,identity)
    save(a.run_root/'preflight.json',result)
    print(json.dumps({'passed':True,'smoke':a.smoke,'cuda':result['cuda_available'],'run_root':str(a.run_root)},ensure_ascii=False),flush=True)
def env_for(a,family):
    mode,backbone=FAMILIES[family];env=os.environ.copy()
    env.update(MSA_DATA_ROOT=str(a.data_root),MSA_RUN_ROOT=str(a.run_root/family),MSA_INPUT_MODE=mode,MSA_BACKBONE=backbone,MSA_SMOKE='1' if a.smoke else '0',MSA_DEVICE='cpu' if a.smoke else 'cuda',PYTHONPATH=str(HERE/'src'),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2' if a.smoke else '4',OPENBLAS_NUM_THREADS='2' if a.smoke else '4',MKL_NUM_THREADS='2' if a.smoke else '4')
    return env
def execute(a,jobs,stage):
    for num,(family,name,args) in enumerate(jobs,1):
        code_check()
        if shutil.disk_usage(a.run_root).free < (0.1 if a.smoke else 3)*1024**3:raise RuntimeError('Insufficient disk reserve; no historical artifacts deleted')
        log=a.run_root/'logs'/(family+'_'+name+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('a',encoding='utf-8') as f:
            proc=subprocess.Popen([sys.executable,'-u','-m']+args,cwd=HERE,env=env_for(a,family),stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            due=0
            while True:
                code=proc.poll();now=time.time();files=list((a.run_root/family/'runs').rglob('status.json'));latest=max(files,key=lambda p:p.stat().st_mtime) if files else None
                child=read(latest) if latest else {}
                value={'stage':stage,'family':family,'job':name,'job_number':num,'job_count':len(jobs),'worker_pid':proc.pid,'alive':code is None,'exit_code':code,'updated':now,'child':child,'log':str(log),'smoke':a.smoke}
                save(a.run_root/'status.json',value)
                if now>=due:
                    with (a.run_root/'monitor_15min.jsonl').open('a',encoding='utf-8') as m:m.write(json.dumps(value,ensure_ascii=False)+'\n')
                    due=now+900
                if code is not None:break
                time.sleep(2 if a.smoke else 10)
        if code:raise RuntimeError(f'{family}/{name} exit {code}: {log}. Fix the reported cause; rerun train to resume saved state, not to draw a new run.')
def locks(a):
    for family in FAMILIES:
        for ds in COUNTS:
            p=a.run_root/family/'runs'/ds/'fold_0';done=read(p/'ours_done.json');lk=p/'selection_lock.json'
            assert sha(lk)==done['files']['selection_lock.json']
            for n,h in read(lk)['artifacts'].items():assert sha(p/n)==h,(p,n)
def train(a):
    check(a)
    if (a.run_root/'training_done.json').exists():
        assert read(a.run_root/'training_done.json')['source_sha']==code_check();locks(a)
        print('Training already locked; no model retrained. Existing evaluation status is preserved.',flush=True)
        return
    jobs=[]
    for family in FAMILIES:
        jobs.append((family,'prepare',['evidence_msa.text_present.runner','prepare']))
        for ds in COUNTS:
            mod=['evidence_msa.text_present.plain_worker','train'] if family=='enhanced_plain' else ['evidence_msa.text_present.train']
            jobs.append((family,ds,mod+['--dataset',ds]))
    execute(a,jobs,'training');locks(a);summarize(a,'valid')
    save(a.run_root/'training_done.json',{'source_sha':code_check(),'smoke':a.smoke,'test_opened':False,'families':list(FAMILIES)})
    save(a.run_root/'status.json',{'stage':'training_complete','variants':5,'datasets':2,'test_opened':False,'updated':time.time(),'smoke':a.smoke})
def evaluate(a):
    check(a);assert read(a.run_root/'training_done.json')['source_sha']==code_check();locks(a)
    jobs=[(f,'evaluate',['evidence_msa.text_present.plain_worker' if f=='enhanced_plain' else 'evidence_msa.text_present.runner','evaluate']) for f in FAMILIES]
    execute(a,jobs,'evaluating');summarize(a,'test')
    save(a.run_root/'status.json',{'stage':'complete','variants':5,'datasets':2,'test_opened':True,'updated':time.time(),'smoke':a.smoke})
def summarize(a,split):
    import numpy as np
    sys.path.insert(0,str(HERE/'src'))
    from evidence_msa.common.metrics import scores
    dest=a.run_root/'tables';dest.mkdir(exist_ok=True);rows=[];audit=[];refs={}
    for ds in COUNTS:
        for variant,(family,kind) in VARIANTS.items():
            root=a.run_root/family;unit=root/'runs'/ds/'fold_0'
            if split=='valid':p=unit/'validation_predictions.npz';prefix='final_' if kind=='final' else 'c5_'
            else:
                p=root/'evaluation'/ds/('predictions_c5_test.npz' if kind=='c5' and family!='enhanced_plain' else 'predictions_test.npz');prefix='pred_'
                done=read(p.parent/'done.json');assert sha(unit/'selection_lock.json')==done['selection_lock_sha']
                assert sha(p)==done['c5_predictions_sha' if p.name=='predictions_c5_test.npz' else 'predictions_sha']
            with np.load(p,allow_pickle=False) as z:
                ids=[str(x).replace('$_$','_') for x in z['ids']];y=z['y'].reshape(-1)
                assert len(ids)==len(set(ids))==len(y)
                if ds in refs:assert ids==refs[ds][0] and np.array_equal(y,refs[ds][1])
                else:refs[ds]=(ids,y.copy())
                for c in ('T','TA','TV'):
                    computed=scores(y,z[prefix+c]);rows.append(dict(dataset=ds.upper(),variant=variant,condition=c,**{k:computed['metrics'][k] for k in METRICS},n=computed['n_all'],split=split,smoke=a.smoke))
            audit.append({'variant':variant,'dataset':ds,'source':str(p),'sha256':sha(p),'selection_lock_sha':sha(unit/'selection_lock.json')})
    def write(name,items):
        with (dest/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    write(split+'_results.csv',rows);means=[];deltas=[]
    lookup={(r['dataset'],r['variant'],r['condition']):r for r in rows}
    for ds in ('MOSI','MOSEI'):
        for v in VARIANTS:
            for label,conds in [('LA_LV',('TA','TV')),('L_LA_LV',('T','TA','TV'))]:
                means.append(dict(dataset=ds,variant=v,conditions=label,**{k:None if any(lookup[(ds,v,c)][k] is None for c in conds) else float(np.mean([lookup[(ds,v,c)][k] for c in conds])) for k in METRICS}))
        for newer,older in [('A1','A0'),('A2','A0'),('A3','A1'),('A3','A2'),('A1','A4')]:
            for c in ('T','TA','TV'):
                for k in METRICS:
                    x=lookup[(ds,newer,c)][k];y=lookup[(ds,older,c)][k]
                    deltas.append(dict(dataset=ds,contrast=newer+'-'+older,condition=c,metric=k,improvement=None if x is None or y is None else (y-x if k=='MAE' else x-y)))
    write(split+'_means.csv',means);write(split+'_contrasts.csv',deltas)
    save(dest/(split+'_audit.json'),{'rows':len(rows),'metrics':len(rows)*6,'sources':audit,'smoke':a.smoke,'historical_test_exposure':True,'independent_confirmation':False,'paired_core':'A0/A2 and A1/A3 use identical selected C5 checkpoint; no-ERU rows contain no ERU predictions','note':'A/V/AV not evaluated here; do not label three-condition mean as six-condition mean'})
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','train','evaluate','status','export']);p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/MSA_Cloud_20261115/data'));p.add_argument('--run-root',type=Path,default=HERE/'outputs/formal');p.add_argument('--smoke',action='store_true');a=p.parse_args();a.data_root=a.data_root.resolve();a.run_root=a.run_root.resolve()
    if a.smoke and 'smoke' not in a.run_root.name.lower():p.error('Synthetic smoke output directory must contain smoke')
    if a.action=='status':
        print(json.dumps(read(a.run_root/'status.json'),ensure_ascii=False,indent=2));return
    if a.action=='check':check(a);return
    if a.action=='export':
        import zipfile
        status=read(a.run_root/'status.json');assert status['stage'] in ('training_complete','complete')
        target=a.run_root/'ablation_results.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for file in a.run_root.rglob('*'):
                if not file.is_file() or file==target:continue
                if file.suffix in ('.json','.csv') or file.name in ('validation_predictions.npz','predictions_test.npz','predictions_c5_test.npz'):z.write(file,file.relative_to(a.run_root))
        (target.with_suffix('.zip.sha256')).write_text(sha(target)+'  '+target.name+'\n');print(target);return
    a.run_root.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(HERE/'src'))
    from evidence_msa.common.checkpoint import lock
    with lock(a.run_root/'ablation_queue.lock'):
        try:train(a) if a.action=='train' else evaluate(a)
        except Exception as e:
            save(a.run_root/'error.json',{'error':repr(e),'time':time.time(),'action':a.action})
            save(a.run_root/'status.json',{'stage':'failed','error':repr(e),'updated':time.time(),'action':a.action,'smoke':a.smoke});raise
if __name__=='__main__':main()
