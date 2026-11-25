"""Synthetic export/checkpoint integrity test; no benchmark results."""
import csv,json,tempfile,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import supplement_queue as q

with tempfile.TemporaryDirectory(prefix='tav-export-test-') as tmp:
    root=Path(tmp);a=SimpleNamespace(out=root/'outputs',ready=root/'ready');a.out.mkdir()
    for ds in ('mosi','mosei'):
        (a.ready/ds).mkdir(parents=True)
        np.savez(a.ready/ds/'test.npz',ids=np.array(['clip_1','clip_2','clip_3']),y=np.array([-1.,0.,1.]))
    for m,d in q.JOBS:
        u=q.unit(a,m,d);u.mkdir(parents=True)
        (u/'best.pt').write_bytes(b'synthetic checkpoint integrity fixture')
        q.save(u/'protocol.json',{'synthetic':True})
        q.save(u/'selection_lock.json',dict(checkpoint_sha256=q.sha(u/'best.pt'),protocol_sha256=q.sha(u/'protocol.json')))
        np.savez(u/'predictions_test.npz',ids=np.array(['clip_3','clip_1','clip_2']),y=np.array([1.,-1.,0.]),pred=np.array([.8,-.9,.1]))
        q.save(u/'test_done.json',dict(predictions_sha256=q.sha(u/'predictions_test.npz'),selection_sha256=q.sha(u/'selection_lock.json')))
    q.export(a)
    rows=list(csv.DictReader((a.out/'TAV_RoME_TMDC_results.csv').open(encoding='utf-8-sig')))
    assert len(rows)==4 and all(abs(float(r['MAE'])-4/30)<1e-7 for r in rows)
    with zipfile.ZipFile(a.out/'TAV_RoME_TMDC_results.zip') as z:assert len([x for x in z.namelist() if x.endswith('predictions_test.npz')])==4
    u=q.unit(a,*q.JOBS[0]);(u/'best.pt').write_bytes(b'tampered')
    try:q.locked(a,*q.JOBS[0])
    except AssertionError:pass
    else:raise AssertionError('Changed checkpoint accepted')
print('PASS: four-job export, ID reorder alignment, metric recomputation, ZIP contents, checkpoint tamper rejection.')
