"""Frozen checkpoint validation replay, then same original test IDs/metrics."""
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from metrics_common import digest, scores, atomic_json
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--split',choices=['valid','test'],required=True)
c=p.parse_args();run=c.run.resolve();lock=json.loads((run/'validation_lock.json').read_text(encoding='utf-8'))
assert not lock['smoke'] and lock['status']=='validation_locked_no_test_evaluated'
checkpoint_hash=digest(run/'best.pt')
assert checkpoint_hash==lock['checkpoint_sha256']
dest=run/'evaluation';dest.mkdir(exist_ok=True)
if (dest/f'{c.split}_report.json').exists():raise RuntimeError('Preserve existing evaluation')
if c.split=='test':
    replay=json.loads((dest/'valid_report.json').read_text(encoding='utf-8'))
    assert replay['passed'] and replay['checkpoint_sha256']==checkpoint_hash
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MSA_TEST_ROOT='D:/MSA_Cloud_20261115')
worker_path=ROOT/'frozen_test_worker.py'
sys.path.insert(0,str(worker_path.parent))
spec=importlib.util.spec_from_file_location('archived_test_worker',worker_path);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
worker.ROOT=Path('D:/MSA_Cloud_20261115');worker.WHOLE=ROOT
torch.set_num_threads(4)
ids,y,pred=worker.infer_four(run,lock['method'],lock['dataset'],lock['condition'],c.split,torch.device('cuda'))
ids=np.asarray(ids).astype(str);y=np.asarray(y).reshape(-1);pred=np.asarray(pred).reshape(-1)
ready=worker.ROOT/'data/ready'/lock['dataset']/(c.split+'.npz')
with np.load(ready,allow_pickle=False) as z:
    normal=lambda vals:[str(v).replace('$_$','_') for v in vals]
    expected=normal(z['ids']);actual=normal(ids)
    assert len(actual)==len(set(actual))==len(expected) and set(actual)==set(expected)
    idx={v:i for i,v in enumerate(actual)};order=[idx[v] for v in expected]
    assert np.allclose(y[order],z['y'].reshape(-1),rtol=0,atol=1e-6)
report=scores(y,pred)
if c.split=='valid':report['replay']=worker.check_replay(report,ids,y,pred,run)
# Independent implementation, including support-weighted binary F1.
from sklearn.metrics import f1_score,mean_absolute_error
nz=y!=0
independent={'MAE':float(mean_absolute_error(y.astype(float),pred.astype(float))),
    'Corr':float(np.corrcoef(y,pred)[0,1]) if np.std(y)>0 and np.std(pred)>0 else None,'Acc2_nonzero':float(np.mean((pred[nz]>0)==(y[nz]>0))),
    'F1_weighted_nonzero':float(f1_score(y[nz]>0,pred[nz]>0,average='weighted',zero_division=0)),
    'Acc5':float(np.mean(np.rint(np.clip(pred,-2,2))==np.rint(np.clip(y,-2,2)))),
    'Acc7':float(np.mean(np.rint(np.clip(pred,-3,3))==np.rint(np.clip(y,-3,3))))}
for key,value in independent.items():
    actual=report['metrics'][key]
    assert (value is None and actual is None) or (value is not None and actual is not None and abs(value-actual)<1e-7),(key,value,actual)
assert digest(run/'best.pt')==checkpoint_hash
np.savez_compressed(dest/f'{c.split}_predictions.npz',ids=ids,y=y,pred=pred)
report.update(passed=True,split=c.split,method=lock['method'],dataset=lock['dataset'],condition=lock['condition'],
    checkpoint_sha256=checkpoint_hash,selected_epoch=lock['selected_epoch'],
    predictions_sha256=digest(dest/f'{c.split}_predictions.npz'),independent_metrics=independent,
    historical_test_exposure=True,independent_confirmation=False)
atomic_json(dest/f'{c.split}_report.json',report)
print(json.dumps({'passed':True,'run':run.name,'split':c.split,'n':len(y)}))
