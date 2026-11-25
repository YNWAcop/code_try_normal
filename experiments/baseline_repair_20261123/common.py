import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('MSA_TEST_ROOT','/root/autodl-tmp/MSA_Cloud_20261115')).resolve()
WHOLE=ROOT.parent/'whole_priority_20261117';HAND=ROOT.parent/'baseline_handoff_20261115'
sys.path.insert(0,str(HERE))
from metrics_common import atomic_json,digest,scores,METRIC_KEYS
METHODS=('EBMC','TLRA','API','LNLN','Self_MM','EMT_DLFR','TFR_NET','ALMT','MISA','OURS_reliable')
CONDITIONS=('A','V','T','AV','TA','TV')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def source(method,job):return (ROOT/'baseline_results' if method in METHODS[:5] else WHOLE/'results')/job
def result_dir(job):return HERE/'results'/job
def plan_jobs():
    return [dict(job=f'{m}_{d}_{c}',method=m,dataset=d,condition=c,source=str(source(m,f'{m}_{d}_{c}'))) for d in ('mosi','mosei') for m in METHODS for c in CONDITIONS]
def completed(spec,split,plan_hash):
    out=result_dir(spec['job']);name='validation_replay.json' if split=='valid' else 'metrics_full_test.json'
    try:
        r=read(out/name);p=out/('predictions_replay_valid.npz' if split=='valid' else 'predictions_full_test.npz')
        return r['status']=='complete' and r['split']==split and r['plan_sha256']==plan_hash and r['checkpoint_sha256']==spec['checkpoint_sha256'] and r['predictions_sha256']==digest(p)
    except (OSError,ValueError,KeyError):return False

