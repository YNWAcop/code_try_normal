"""Synthetic integration check only: no benchmark scores produced."""
import pickle, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='rome-smoke-') as tmp:
    root=Path(tmp);folder=root/'data/CMUMOSI';folder.mkdir(parents=True)
    vids=['train0','train1','valid0','valid1','test0'];ids={v:[v+'_'+str(i) for i in range(2 if v.endswith('1') else 3)] for v in vids}
    labels={v:[-1.,0.,1.][:len(ids[v])] for v in vids};speakers={v:['']*len(ids[v]) for v in vids};text={v:['x']*len(ids[v]) for v in vids}
    with (folder/'CMUMOSI_features_raw_2way.pkl').open('wb') as f:pickle.dump((ids,labels,speakers,text,vids[:2],vids[2:4],vids[4:]),f)
    for mod in ['wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT']:
        d=folder/'features'/mod;d.mkdir(parents=True)
        for v in vids:
            for uid in ids[v]:np.save(d/(uid+'.npy'),np.random.normal(size=8).astype('float32'))
    cmd=[sys.executable,str(HERE/'rome_adapter.py'),'smoke','--dataset','mosi','--condition','TAV','--device','cpu','--data-root',str(root/'data'),'--output-root',str(root/'out')]
    subprocess.run(cmd,check=True)
    job=root/'out/smoke/mosi_TAV'
    assert (job/'selection_lock.json').exists()
    import torch,json
    saved=torch.load(job/'last.pt',map_location='cpu',weights_only=False)
    assert saved['epoch']==1 and saved['best_epoch']==2
    with np.load(job/'predictions_valid.npz') as z:
        assert len(z['ids'])==5 and z['y'].tolist()==[-1.,0.,1.,-1.,0.]
    # Re-enter an already frozen run; must not retrain.
    before=(job/'last.pt').read_bytes();subprocess.run(cmd,check=True)
    assert before==(job/'last.pt').read_bytes()
    # Finish again from last epoch checkpoint (simulates interruption before lock write).
    (job/'selection_lock.json').unlink();subprocess.run(cmd,check=True)
    assert (job/'selection_lock.json').exists()
    print('PASS: official model/loss stage1+stage2; clip alignment; completed-run skip; checkpoint finalization')

