"""Publish a new comparison only after all predeclared repairs pass. Never overwrite old tables."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from metrics_common import atomic_json,digest,scores
OLD=ROOT.parent/'outputs/final_paper_tables_20261122';RUNS=Path('D:/baseline_repair_runs_20261123')
OUT=ROOT.parent/'outputs/baseline_repaired_tables_20261123'
KEYS=['Acc7','Acc5','Acc2_nonzero','F1_weighted_nonzero','MAE','Corr']
LABEL={'T':'L','A':'A','V':'V','TA':'LA','TV':'LV','AV':'AV'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def writecsv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def close(a,b):return (a is None and b is None) or (a is not None and b is not None and np.isclose(float(a),float(b),rtol=1e-7,atol=1e-8))
def verify_prediction(source,key,sha,reference=None):
    assert digest(source)==sha,source
    with np.load(source,allow_pickle=False) as z:
        ids=[str(v).replace('$_$','_') for v in z['ids']];y=z['y'].reshape(-1);pred=z[key].reshape(-1)
    assert len(ids)==len(set(ids))==len(y)==len(pred)
    current=dict(zip(ids,y))
    if reference is not None:
        assert set(current)==set(reference)
        ref=np.array([reference[i] for i in ids]);assert np.allclose(y,ref,rtol=0,atol=1e-6)
        assert np.array_equal(y==0,ref==0) and np.array_equal(y>0,ref>0)
        for bound in (2,3):assert np.array_equal(np.rint(np.clip(y,-bound,bound)),np.rint(np.clip(ref,-bound,bound)))
    return scores(y,pred),current
def assemble(publish):
    plan=read(ROOT/'repair_retrain_plan.json')
    with (OLD/'all_methods_by_condition.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    original={ (r['method'],r['dataset'],r['condition']):r.copy() for r in rows }
    sources=read(OLD/'qa/verification.json')['sources'];refs={};lookup={ (r['method'],r['dataset'],r['condition']):r for r in rows }
    for s in sources:
        report,ref=verify_prediction(Path(s['source']),s['prediction_key'],s['sha256'],refs.get(s['dataset']))
        refs.setdefault(s['dataset'],ref);row=lookup[(s['method'],s['dataset'].upper(),LABEL[s['condition']])]
        for k in KEYS:assert close(report['metrics'][k],row[k]),(s,k)
    ready=[];missing=[];deltas=[]
    for job in plan['jobs']:
        tag=f"{job['method']}_{job['dataset']}_{job['condition']}";run=RUNS/tag
        p=run/'evaluation/test_report.json'
        if not p.exists():missing.append(tag);continue
        test=read(p);valid=read(run/'evaluation/valid_report.json');lock=read(run/'validation_lock.json')
        assert test['passed'] and valid['passed'] and not lock['smoke']
        assert digest(run/'best.pt')==test['checkpoint_sha256']==valid['checkpoint_sha256']==lock['checkpoint_sha256']
        computed,_=verify_prediction(run/'evaluation/test_predictions.npz','pred',test['predictions_sha256'],refs[job['dataset']])
        key=(job['method'],job['dataset'].upper(),LABEL[job['condition']]);row=lookup[key]
        for k in KEYS:
            assert close(computed['metrics'][k],test['metrics'][k]),(tag,k)
            old=float(row[k]);new=computed['metrics'][k]
            deltas.append(dict(method=key[0],dataset=key[1],condition=key[2],metric=k,previous=old,repaired=new,delta=None if new is None else new-old))
            row[k]=new
        row['n_all']=computed['n_all'];row['n_nonzero']=computed['n_nonzero'];row['provenance']='2026-11-23 protocol repair; original test IDs; validation-selected native-batch retrain'
        ready.append(tag)
    audit={'ready':len(ready),'expected':len(plan['jobs']),'pending':missing,'old_configurations_verified':len(rows),'old_metric_values_verified':len(rows)*6,'repaired_metric_values_verified':len(ready)*6,'all_prediction_hashes_ids_labels_checked':True,'historical_test_exposure':True,'independent_confirmation':False,'published':False,'updated':time.time()}
    atomic_json(ROOT/'table_gate_status.json',audit)
    if not publish or missing:return audit
    assert len(ready)==24
    for p,h in plan['sources'].items():assert digest(p)==h,('Frozen source changed',p)
    assert not (OUT/'verification.json').exists(),'Preserve published revision'
    OUT.mkdir(parents=True,exist_ok=True)
    writecsv(OUT/'all_methods_by_condition.csv',rows);writecsv(OUT/'repair_deltas.csv',deltas)
    means=[]
    for ds in ('MOSI','MOSEI'):
        for method in dict.fromkeys(r['method'] for r in rows):
            group=[r for r in rows if r['dataset']==ds and r['method']==method];assert len(group)==6
            means.append(dict(dataset=ds,method=method,**{k:None if any(r[k] is None for r in group) else float(np.mean([float(r[k]) for r in group])) for k in KEYS}))
    writecsv(OUT/'six_condition_means.csv',means)
    for row in rows:
        if row['method'] not in ('EMT_DLFR','MISA'):assert row==original[(row['method'],row['dataset'],row['condition'])]
    audit.update(published=True,output=str(OUT),changed_methods=['EMT_DLFR','MISA'],unchanged_ours=True,old_tables_preserved=True,completed_jobs=ready,plan_sha256=digest(ROOT/'repair_retrain_plan.json'))
    audit['output_hashes']={p.name:digest(p) for p in OUT.glob('*.csv')}
    atomic_json(OUT/'verification.json',audit);atomic_json(ROOT/'table_gate_status.json',audit)
    return audit
def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.watch:
        while True:
            if (ROOT/'queue_error.json').exists():raise RuntimeError('Queue failed: '+str(read(ROOT/'queue_error.json')))
            s=read(ROOT/'queue_status.json')
            atomic_json(ROOT/'finalizer_status.json',{'stage':'waiting_for_repairs','completed':s.get('completed'),'expected':s.get('expected'),'updated':time.time()})
            if s['stage']=='complete':break
            if time.time()-s['updated']>600:raise RuntimeError('Queue heartbeat stale; no final table published')
            time.sleep(30)
    result=assemble(not a.check_only);print(json.dumps(result,ensure_ascii=False))
    if a.watch:assert result['published']
if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(ROOT/'finalizer_error.json',{'error':repr(e),'time':time.time()});raise
