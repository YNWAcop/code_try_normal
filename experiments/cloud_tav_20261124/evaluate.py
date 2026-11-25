"""Nine baseline checkpoint replays and test inference; never optimize."""
import argparse,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--dataset',required=True);p.add_argument('--split',choices=['valid','test'],required=True);a=p.parse_args()
sys.path[:0]=[str(P/'four'),str(P/'hand')]
os.environ.update(MSA_TEST_ROOT=str(a.workspace),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
import frozen_test_worker as w
from metrics_common import scores,digest,atomic_json
w.ROOT=a.workspace.resolve();w.WHOLE=P/'four';w.HAND=P/'hand'
run=a.run.resolve();dest=run/'evaluation';dest.mkdir(exist_ok=True)
lk=run/'validation_lock.json';lock=w.read(lk);assert set(lock['condition'].upper())==set('TAV') and not lock['smoke']
weight=run/('stage2_best.pt' if a.method=='EBMC' else 'best.pt');h=digest(weight)
if lock.get('checkpoint_sha256'):assert lock['checkpoint_sha256']==h
reportfile=dest/(a.split+'_report.json');predfile=dest/(a.split+'_predictions.npz')
if reportfile.exists():
    old=w.read(reportfile);assert old['passed'] and old['checkpoint_sha256']==h and old['predictions_sha256']==digest(predfile) and old['lock_sha256']==digest(lk);sys.exit(0)
if a.split=='test':
    old=w.read(dest/'valid_report.json');assert old['passed'] and old['checkpoint_sha256']==h and old['lock_sha256']==digest(lk)
torch.set_num_threads(4);device=torch.device('cuda')
if a.method=='EBMC':ids,y,pred=w.infer_ebmc(run,a.dataset,'TAV',a.split,device)
elif a.method in ('EMT_DLFR','TFR_NET','ALMT','MISA'):ids,y,pred=w.infer_four(run,a.method,a.dataset,'TAV',a.split,device)
else:ids,y,pred=w.infer_old(run,a.method,a.dataset,'TAV',a.split,device)
ids=np.asarray(ids).astype(str);y=np.asarray(y).reshape(-1);pred=np.asarray(pred).reshape(-1)
with np.load(a.workspace/'data/ready'/a.dataset/(a.split+'.npz'),allow_pickle=False) as z:
    norm=lambda x:[str(v).replace('$_$','_') for v in x]
    actual=norm(ids);expected=norm(z['ids']);assert len(set(actual))==len(actual)==len(expected) and set(actual)==set(expected)
    index={v:i for i,v in enumerate(actual)};order=[index[v] for v in expected];np.testing.assert_allclose(y[order],z['y'].reshape(-1),atol=1e-6,rtol=0)
report=scores(y,pred)
if a.split=='valid':report['replay']=w.check_replay(report,ids,y,pred,run)
assert digest(weight)==h
np.savez_compressed(predfile,ids=ids,y=y,pred=pred)
report.update(passed=True,method=a.method,dataset=a.dataset,condition='TAV',split=a.split,checkpoint_sha256=h,lock_sha256=digest(lk),predictions_sha256=digest(predfile),historical_test_exposure=True,independent_confirmation=False)
atomic_json(reportfile,report)
print(json.dumps({'passed':True,'method':a.method,'dataset':a.dataset,'split':a.split,'n':len(y)}))
