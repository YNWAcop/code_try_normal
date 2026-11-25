"""Run the fixed text-present protocol; monitor every 900 seconds."""
import json
import os
import shutil
import subprocess
import sys
import time
from ..settings import PROJECT
from .protocol import ROOT, save, read, sha, status, lock


def check_code():
    manifest = read(PROJECT / 'release/runtime_manifest.json')
    for name, expected in manifest['files'].items():
        if sha(PROJECT / name) != expected:
            raise RuntimeError('Release source changed: ' + name)
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / 'portable_code_freeze.json'
    if frozen.exists():
        if read(frozen) != manifest:
            raise RuntimeError('Run was created with a different release')
    else:
        save(frozen, manifest)


def run(evaluate=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    with lock(ROOT / 'queue.lock'):
        check_code()
        logs = ROOT / 'logs'
        logs.mkdir(exist_ok=True)
        jobs = [
            ('prepare', ['evidence_msa.text_present.runner', 'prepare']),
            ('mosi', ['evidence_msa.text_present.train', '--dataset', 'mosi']),
            ('mosei', ['evidence_msa.text_present.train', '--dataset', 'mosei']),
        ]
        if evaluate:
            jobs.append(('evaluate', ['evidence_msa.text_present.runner', 'evaluate']))
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
        for name, args in jobs:
            check_code()
            if shutil.disk_usage(ROOT).free < 3 * 1024**3:
                raise RuntimeError('Less than 3 GiB free; preserve artifacts and free space manually')
            with (logs / (name + '.log')).open('a', encoding='utf-8') as log:
                p = subprocess.Popen([sys.executable, '-u', '-m'] + args,
                                     cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                status(ROOT, 'running', job=name, worker_pid=p.pid, test_opened=name == 'evaluate')
                due = time.monotonic()
                while p.poll() is None:
                    if time.monotonic() >= due:
                        files = list((ROOT / 'runs').rglob('status.json'))
                        latest = max(files, key=lambda q: q.stat().st_mtime) if files else None
                        record = dict(time=time.time(), job=name, pid=p.pid, alive=True,
                                      free_gib=shutil.disk_usage(ROOT).free / 1024**3,
                                      child=read(latest) if latest else {})
                        save(ROOT / 'monitor_latest.json', record)
                        with (ROOT / 'monitor.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(record) + '\n')
                        due = time.monotonic() + 900
                    time.sleep(5)
                if p.returncode:
                    status(ROOT, 'failed', job=name, exit_code=p.returncode)
                    raise RuntimeError('Job failed: ' + name)
        status(ROOT, 'complete' if evaluate else 'training_complete', datasets=2,
               test_opened=evaluate, independent_confirmation=False)
