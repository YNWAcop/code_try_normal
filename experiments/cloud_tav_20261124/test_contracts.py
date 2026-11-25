"""CPU-only synthetic checks; no benchmark test predictions or formal training."""
import ast, importlib.util, json, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
P=Path(__file__).resolve().parent
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
q=module('tav_queue',P/'tav_queue.py')
checks=[]
files=list(P.rglob('*.py'))
for f in files:ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
checks.append({'check':'all_python_syntax','files':len(files)})
with tempfile.TemporaryDirectory() as tmp:
    a=SimpleNamespace(root=Path(tmp)/'root',out=Path(tmp)/'outputs',ablation=Path(tmp)/'ablation')
    a.out.mkdir()
    assert len(q.METHODS)==9 and len(q.DS)==2
    for m in q.METHODS:
        for d in q.DS:
            cmd=list(map(str,q.commands(a,m,d)))
            assert cmd[cmd.index('--condition')+1]=='TAV' and '--smoke' not in cmd
            if '--rate' in cmd:assert cmd[cmd.index('--rate')+1]=='0'
            assert '--smoke' in q.commands(a,m,d,True)
    checks.append({'check':'18_TAV_commands_and_smoke_separation','passed':True})
    calls=[]
    with patch.object(q,'ablation_stage',side_effect=['training','training_complete','complete']),patch.object(q.time,'sleep'),patch.object(q,'run',side_effect=lambda a,c,l:calls.append((c,l))):q.wait_ablation(a)
    assert [v[1] for v in calls]==['finish_ablation_test','export_ablation']
    q.save(a.ablation/'outputs/formal/status.json',{'stage':'failed'})
    try:q.wait_ablation(a);raise AssertionError('Failed ablation accepted')
    except RuntimeError:pass
    checks.append({'check':'ablation_wait_evaluate_export_and_fail_stop','passed':True})
    f=q.folder(a,'EBMC','mosi');f.mkdir(parents=True)
    (f/'stage2_best.pt').write_bytes(b'synthetic checkpoint')
    q.save(f/'validation_lock.json',dict(condition='atv',smoke=False,dataset='mosi',status='validation_locked_no_test_evaluated'))
    assert q.locked(a,'EBMC','mosi')
    (f/'stage2_best.pt').write_bytes(b'changed')
    try:q.locked(a,'EBMC','mosi');raise RuntimeError('Changed checkpoint accepted')
    except AssertionError:pass
    checks.append({'check':'EBMC_atv_alias_and_checkpoint_tamper_detection','passed':True})
    class Base:
        def __len__(self):return 1
        def __getitem__(self,i):return dict(text=torch.arange(24).reshape(3,8).float(),audio=torch.ones(9,3),vision=torch.ones(7,4)*2,audio_lengths=9,vision_lengths=7,labels={'M':torch.tensor([1.])})
    for file in ('hand/corruption.py','four/mmsa_corruption.py'):
        c=module('corruption_test',P/file)
        for lnln in (False,True):
            base=Base();value=c.CorruptedDataset(base,Path(tmp),'mosi','train','whole','TAV',0,lnln)[0]
            for key in ('text','audio','vision'):assert torch.equal(value[key+('_m' if lnln else '')],base[0][key])
            assert value['audio_lengths']==9 and value['vision_lengths']==7
            if lnln:assert all(value['labels']['missing_rate_'+k]==0 for k in 'lav')
    checks.append({'check':'TAV_preserves_all_features_lengths_LNLN_zero_missingness','passed':True})
report={'passed':True,'scope':'CPU synthetic contracts only; no cloud training or benchmark scores','checks':checks}
(P/'LOCAL_TEST_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
