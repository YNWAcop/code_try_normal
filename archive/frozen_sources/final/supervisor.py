import os,sys,time,subprocess,shutil,json
from protocol import ROOT,save,read,sha,status,lock

def check_code():
    for n,h in read(ROOT/'code_freeze.json')['files'].items():assert sha(n)==h,'Frozen code changed: '+n

def run():
    with lock(ROOT/'queue.lock'):
        check_code();logs=ROOT/'logs';logs.mkdir(exist_ok=True)
        for name,args in [('prepare',['runner.py','prepare']),('mosi',['train_ours.py','--dataset','mosi']),('mosei',['train_ours.py','--dataset','mosei']),('evaluate',['runner.py','evaluate']),('compare',['compare_results.py'])]:
            check_code();assert shutil.disk_usage(ROOT).free>3*1024**3
            with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
                p=subprocess.Popen([sys.executable,'-u']+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                status(ROOT,'running',job=name,worker_pid=p.pid,log=str(logs/(name+'.log')),test_opened=name in ('evaluate','compare'))
                due=time.monotonic()
                while p.poll() is None:
                    if time.monotonic()>=due:
                        files=list((ROOT/'runs').rglob('status.json'))
                        latest=max(files,key=lambda q:q.stat().st_mtime) if files else None
                        record=dict(time=time.time(),job=name,pid=p.pid,alive=True,free_gib=shutil.disk_usage(ROOT).free/1024**3,latest_status=str(latest),child=read(latest) if latest else {})
                        save(ROOT/'monitor_latest.json',record)
                        with (ROOT/'monitor.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                        due=time.monotonic()+900
                    time.sleep(5)
                if p.returncode:
                    status(ROOT,'failed',job=name,exit_code=p.returncode,log=str(logs/(name+'.log')));raise RuntimeError(name)
        status(ROOT,'complete',datasets=2,test_evaluation='single post-lock original benchmark',independent_confirmation=False,test_opened=True)

if __name__=='__main__':run()
