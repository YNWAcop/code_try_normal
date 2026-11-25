import argparse,csv,json,os,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();w=a.work.resolve();w.mkdir(parents=True,exist_ok=False)
    data=w/'data';run=w/'smoke';rng=np.random.default_rng()
    for ds in ('mosi','mosei'):
        for split,n in [('train',48),('valid',16),('test',16)]:
            d=dict(ids=np.array([f'{ds}_{split}_{i}' for i in range(n)]),groups=np.array([f'{ds}_{split}_g{i//4}' for i in range(n)]),y=np.resize(np.array([-2.3,1.5,-.2,.3],dtype='float32'),n))
            for k in ('T','A_native','A_words','V_native','V_lip','V_words','V_expression','V_quality'):
                d[k]=rng.normal(size=(n,8 if k=='T' else 4)).astype('float32')
            d['V_timeline']=rng.normal(size=(n,4,3)).astype('float32')
            dest=data/'ready'/ds;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/(split+'.npz'),**d)
    env=os.environ.copy();env.update(MSA_SMOKE='1',MSA_DEVICE='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    def call(args,name):
        with (w/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable]+args,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(name,(w/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-6000:])
        print(name,'passed',flush=True)
    call([str(P/'tests/test_tav_contract.py')],'contracts')
    def action(s):return [str(P/'local_tav.py'),s,'--smoke','--data-root',str(data),'--run-root',str(run)]
    call(action('train'),'train')
    assert not list((run/'evaluation').rglob('predictions_test.npz'))
    call(action('evaluate'),'evaluate')
    with (run/'tables/Ours_TAV_test_results.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and all(r['condition']=='TAV' for r in rows)
    # Independent metric recomputation from the exported final TAV predictions.
    from sklearn.metrics import f1_score
    checks=0
    for ds in ('mosi','mosei'):
        with np.load(run/'evaluation'/ds/'predictions_test.npz') as z:y=z['y'].astype(float);v=z['pred_TAV'].astype(float)
        nz=y!=0;metrics=dict(MAE=np.abs(y-v).mean(),Corr=np.corrcoef(y,v)[0,1],Acc7=np.mean(np.rint(y.clip(-3,3))==np.rint(v.clip(-3,3))),Acc5=np.mean(np.rint(y.clip(-2,2))==np.rint(v.clip(-2,2))),Acc2_nonzero=np.mean((y[nz]>0)==(v[nz]>0)),F1_weighted_nonzero=f1_score(y[nz]>0,v[nz]>0,average='weighted',zero_division=0))
        r=next(r for r in rows if r['dataset']==ds.upper())
        for k,v in metrics.items():assert np.isclose(float(r[k]),v,atol=1e-12,rtol=0),(ds,k);checks+=1
    import hashlib
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*.pt')}
    old=hashes();call(action('train'),'completed_resume');assert old==hashes()
    predictions={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'evaluation').rglob('predictions*.npz')}
    call(action('evaluate'),'evaluate_again');assert predictions=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'evaluation').rglob('predictions*.npz')}
    with zipfile.ZipFile(run/'SMOKE_Ours_TAV_results.zip') as z:assert z.testzip() is None and not any(n.endswith('.pt') for n in z.namelist())
    report=dict(passed=True,synthetic_only=True,formal_training_started=False,datasets=2,metric_recomputations=checks,
      checks=['T/TA/TV forward equality with frozen source','TAV five attention tokens; both A/V gradients nonzero','D5 zero identity and frozen C5','53-dimensional four-condition intensity features','TAV-only validation selection','complete C5+D5+anchors+4 crossfits+calibration+factor+R3 CPU workflow','training and testing separated','completed rerun does not retrain or rewrite predictions','result ZIP integrity'])
    (w/'LOCAL_TEST_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
