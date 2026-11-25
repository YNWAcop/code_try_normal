"""Finite EMT/MISA repair queue. Original EMT plan retained; no score-dependent retries."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from metrics_common import digest,atomic_json
OUT=Path('D:/baseline_repair_runs_20261123')
PLAN=ROOT/'repair_retrain_plan.json'

def freeze_plan():
    paths=[ROOT/name for name in ('adapter_emt_tfr.py','evaluate_repair.py','mmsa_corruption.py','full_four.py','metrics_common.py','mmsa_slim_pickle.py','mmsa_mapped_pickle.py','frozen_test_worker.py','common.py','run_emt_queue.py')]
    paths+=list((ROOT/'vendor/emt_dlfr').rglob('*.py'))
    paths+=[ROOT/'adapter_misa_almt.py',ROOT/'vendor/mmsa/config/config_regression.json']
    paths+=list((ROOT/'vendor/mmsa').rglob('*.py'))
    sources={str(p):digest(p) for p in paths}
    jobs=[{'method':'EMT_DLFR','dataset':d,'condition':c,'batch_size':32 if d=='mosi' else 16,
           'activation_checkpointing':d=='mosi'} for d in ('mosi','mosei') for c in ('A','V','T','AV','TA','TV')]
    jobs += [{'method':'MISA','dataset':d,'condition':c,'batch_size':16 if d=='mosi' else 64,
              'activation_checkpointing':True} for d in ('mosi','mosei') for c in ('A','V','T','AV','TA','TV')]
    plan={'jobs':jobs,'sources':sources,'selection':'Original validation incomplete-view MAE, original early-stop8, no fixed seeds; replace all affected EMT rows regardless of gain.',
          'reason':'EMT: full public reconstruction grid and native physical batch. MISA: native physical batch preserves CMD/DiffLoss statistics. Original loss weights retained.',
          'test':'Frozen validation checkpoint only, replay first, no test-driven revisions.', 'created':time.time()}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding='utf-8'))['sources']==sources
    else:atomic_json(PLAN,plan)
    return json.loads(PLAN.read_text(encoding='utf-8'))

def main():
    if not json.loads((ROOT/'emt_contract_checks.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('Contract checks missing')
    probe=json.loads((ROOT/'memory_probe_mosi_32_1.json').read_text(encoding='utf-8'));assert probe['passed'] and probe['peak_gib']<6
    smoke=json.loads((OUT/'smoke_native_mosi_A/metrics_full_valid.json').read_text(encoding='utf-8'));assert smoke['status']=='complete'
    for ds in ('mosi','mosei'):
        probe=json.loads((ROOT/f'misa_memory_{ds}.json').read_text(encoding='utf-8'));assert probe['passed'] and probe['peak_gib']<6
    smoke=json.loads((OUT/'smoke_native_MISA_mosei_A/metrics_full_valid.json').read_text(encoding='utf-8'));assert smoke['status']=='complete'
    plan=freeze_plan();done=[]
    for job in plan['jobs']:
        for name,h in plan['sources'].items():assert digest(name)==h,('Source changed',name)
        tag=f"{job['method']}_{job['dataset']}_{job['condition']}"
        run=OUT/tag
        if (run/'evaluation/test_report.json').exists():
            r=json.loads((run/'evaluation/test_report.json').read_text(encoding='utf-8'));assert r['passed'];done.append(tag);continue
        if shutil.disk_usage(OUT).free<2*1024**3:raise RuntimeError('Disk guard: preserve old artifacts, insufficient reserve')
        commands=[]
        if not (run/'validation_lock.json').exists():
            if (run/'best.pt').exists():raise RuntimeError('Interrupted run requires explicit recovery; never silently restart')
            adapter='adapter_misa_almt.py' if job['method']=='MISA' else 'adapter_emt_tfr.py'
            cmd=[sys.executable,str(ROOT/adapter),'--method',job['method'],'--dataset',job['dataset'],'--protocol','whole','--condition',job['condition'],'--batch-size',str(job['batch_size']),'--out',str(run)]
            if job['activation_checkpointing']:cmd+=['--gradient-checkpointing' if job['method']=='MISA' else '--activation-checkpointing']
            commands.append(('training',cmd))
        for split in ('valid','test'):
            if not (run/f'evaluation/{split}_report.json').exists():commands.append((split,[sys.executable,str(ROOT/'evaluate_repair.py'),'--run',str(run),'--split',split]))
        for stage,cmd in commands:
            OUT.mkdir(exist_ok=True);log=OUT/(tag+'_'+stage+'.log')
            with log.open('w',encoding='utf-8') as f:
                proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                last_check=0
                while True:
                    code=proc.poll();now=time.time();history=[]
                    try:history=json.loads((run/'history.json').read_text(encoding='utf-8'))
                    except (OSError,ValueError):pass
                    if not history:
                        try:history=json.loads((run/'validation_history.json').read_text(encoding='utf-8'))
                        except (OSError,ValueError):pass
                    status={'stage':stage,'job':tag,'pid':proc.pid,'queue_pid':os.getpid(),'updated':now,'completed':len(done),'expected':len(plan['jobs']),'epoch':len(history),'exit_code':code,'log':str(log),'run':str(run)}
                    atomic_json(ROOT/'queue_status.json',status)
                    if now-last_check>=900:
                        with (ROOT/'monitor_15min.jsonl').open('a',encoding='utf-8') as m:m.write(json.dumps(status)+'\n')
                        last_check=now
                    if code is not None:break
                    time.sleep(10)
            if code:raise RuntimeError(f'{tag} {stage} exit {code}; inspect {log}')
        done.append(tag)
        while (ROOT/'diagnostic_hold.flag').exists():
            atomic_json(ROOT/'queue_status.json',{'stage':'waiting_for_diagnostics','completed':len(done),'expected':len(plan['jobs']),'updated':time.time(),'queue_pid':os.getpid()})
            time.sleep(10)
    atomic_json(ROOT/'queue_status.json',{'stage':'complete','completed':len(done),'expected':len(plan['jobs']),'updated':time.time(),'jobs':done,'tables_updated':False})

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(ROOT/'queue_error.json',{'error':repr(e),'time':time.time()})
        raise
