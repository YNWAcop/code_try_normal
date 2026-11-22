"""Repository entrypoint. Run `python run.py --help` before training."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))


def main():
    p=argparse.ArgumentParser(description='Final V1 + C5/D24/R3 release; no algorithm search')
    p.add_argument('--data-root',type=Path,default=HERE/'data',help='Contains ready/mosi and ready/mosei')
    p.add_argument('--run-root',type=Path,default=HERE/'runs/text_present')
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('check',help='Check packages, CUDA, feature headers and code hashes; no training')
    sub.add_parser('train-text',help='Train both fixed text-present branches, no test evaluation')
    sub.add_parser('evaluate-text',help='Evaluate both datasets after both model locks exist')
    sub.add_parser('all-text',help='Fixed training followed by one locked test evaluation')
    sub.add_parser('status',help='Print current queue and worker state')
    q=sub.add_parser('prepare-inputs',help='Pack existing author and generated feature caches')
    q.add_argument('--dataset',choices=['mosi','mosei','both'],default='both')
    q=sub.add_parser('train-v1',help='Optional fresh V1 A/V/AV training; changes archived weights')
    q.add_argument('--dataset',choices=['mosi','mosei'],required=True)
    q.add_argument('--condition',choices=['A','V','AV'],required=True)
    q.add_argument('--out',type=Path,required=True)
    q.add_argument('--device',default='cuda')
    q=sub.add_parser('predict',help='Predict without reading labels; use selected branch checkpoint')
    q.add_argument('--input',type=Path,required=True)
    q.add_argument('--model-dir',type=Path,required=True)
    q.add_argument('--condition',required=True)
    q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('score',help='Score saved predictions; aligns IDs before reading metrics')
    q.add_argument('--predictions',type=Path,required=True)
    q.add_argument('--labels',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ['MSA_DATA_ROOT']=str(a.data_root.resolve())
    os.environ['MSA_RUN_ROOT']=str(a.run_root.resolve())
    if a.action=='check':
        from evidence_msa.preflight import check
        print(json.dumps(check(),ensure_ascii=False,indent=2))
    elif a.action in ('train-text','all-text'):
        from evidence_msa.text_present.supervisor import run
        run(evaluate=a.action=='all-text')
    elif a.action=='evaluate-text':
        from evidence_msa.text_present.runner import evaluate
        from evidence_msa.common.checkpoint import lock
        a.run_root.mkdir(parents=True,exist_ok=True)
        with lock(a.run_root/'queue.lock'):evaluate()
    elif a.action=='prepare-inputs':
        from evidence_msa.features.prepare import prepare
        for ds in ('mosi','mosei') if a.dataset=='both' else (a.dataset,):prepare(ds)
    elif a.action=='status':
        candidates=[a.run_root/'status.json',a.run_root/'monitor_latest.json']
        candidates+=sorted((a.run_root/'runs').glob('*/fold_0/status.json'))
        for f in candidates:
            if f.exists():print(f, f.read_text(encoding='utf-8'))
    elif a.action=='train-v1':
        if a.data_root.name!='data':
            raise ValueError('The preserved V1 entrypoint expects a bundle root containing data/ready')
        sys.argv=['v1','--root',str(a.data_root.resolve().parent),'--dataset',a.dataset,
                  '--condition',a.condition,'--out',str(a.out.resolve()),'--device',a.device]
        runpy.run_module('evidence_msa.text_absent.v1',run_name='__main__')
    elif a.action=='predict':
        import numpy as np
        from evidence_msa.inference import features,canonical,predict_text,predict_absent
        if a.output.exists():raise FileExistsError(a.output)
        d=features(a.input);c=canonical(a.condition)
        pred=predict_text(d,a.model_dir,(c,))[c] if 'T' in c else predict_absent(d,a.model_dir,c)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.output,ids=d['ids'],pred=pred,condition=c)
    elif a.action=='score':
        import numpy as np
        from evidence_msa.common.metrics import scores
        if a.output.exists():raise FileExistsError(a.output)
        with np.load(a.predictions,allow_pickle=False) as z:ids=z['ids'].astype(str);pred=z['pred']
        with np.load(a.labels,allow_pickle=False) as z:target=z['ids'].astype(str);y=z['y']
        norm=lambda v:np.array([s.replace('$_$','_') for s in v])
        ids,target=norm(ids),norm(target)
        if len(set(ids))!=len(ids) or len(set(target))!=len(target) or set(ids)!=set(target):
            raise ValueError('IDs must be unique and match exactly')
        order={v:i for i,v in enumerate(target)}
        report=scores(y[[order[v] for v in ids]],pred)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
