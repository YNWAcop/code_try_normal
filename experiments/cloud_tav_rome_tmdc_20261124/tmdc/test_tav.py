"""Synthetic CPU integration/gradient checks. Produces no benchmark results."""
import contextlib,importlib.util,io,json,pickle,sys,tempfile
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE))
import tmdc_worker as w

def synthetic(root):
    folder=root/'CMUMOSI';folder.mkdir(parents=True)
    vids=['tr0','tr1','va0','va1','te0'];ids={v:[v+'_'+str(i) for i in range(3 if v.endswith('0') else 2)] for v in vids}
    ys={v:[-1.,0.,1.][:len(ids[v])] for v in vids};sp={v:['']*len(ids[v]) for v in vids};text={v:['x']*len(ids[v]) for v in vids}
    with (folder/'CMUMOSI_features_raw_2way.pkl').open('wb') as f:pickle.dump((ids,ys,sp,text,vids[:2],vids[2:4],vids[4:]),f)
    for feat in w.FEATURES:
        d=folder/'features'/feat;d.mkdir(parents=True)
        for names in ids.values():
            for uid in names:np.save(d/(uid+'.npy'),np.random.normal(size=8).astype('float32'))
def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='tmdc-test-') as tmp:
        root=Path(tmp);synthetic(root/'data')
        from types import SimpleNamespace
        a=SimpleNamespace(dataset='mosi',condition='TAV',stage='train',batch_size=2,data_root=root/'data',device='cpu',gpu_cap_gib=8)
        c=dict(epochs=2,stage1=1,batch=2,hidden=16,depth=1,dropout=.3,condition='atv')
        d,ls,folder,indices=w.dataset(a,c)
        args=SimpleNamespace(device=torch.device('cpu'),hidden=16,depth=1,num_heads=2,n_classes=1,drop_rate=.3,attn_drop_rate=0.,no_cuda=True,test_condition='atv',dataset='CMUMOSI')
        # Compare adapter gradients/updates to the official loss implementation.
        import train_TMDC as upstream
        upstream.device=torch.device('cpu')
        for first in (True,False):
            m1=w.build_model(args,8,8,8);m2=w.build_model(args,8,8,8);m2.load_state_dict(w.snapshot(m1))
            o1=torch.optim.Adam(m1.parameters(),lr=1e-4,weight_decay=1e-5);o2=torch.optim.Adam(m2.parameters(),lr=1e-4,weight_decay=1e-5)
            b=next(iter(ls['train']));r=w.rng()
            upstream.train_or_eval_model(args,m1,w.MaskedMSELoss(),None,[b],optimizer=o1,train=True,first_stage=first)
            w.restore(r);w.train_batch(m2,b,'cpu','TAV',first,o2)
            for name,x in m1.state_dict().items():torch.testing.assert_close(x,m2.state_dict()[name],atol=2e-6,rtol=2e-5)
        # Absent modalities must not affect the output; hold stochastic VIB draws fixed.
        m2.eval()
        for condition in w.CONDITIONS:
            m2.test_condition={'TAV':'atv','TA':'at'}.get(condition,condition.lower())
            altered=[x.clone() if isinstance(x,torch.Tensor) else x for x in b]
            for i,mod in enumerate(['A','T','V']*2):
                if mod not in condition:altered[i]=torch.randn_like(altered[i])*100
            r=w.rng()
            with torch.no_grad():y1=w.forward(m2,b,'cpu',condition,False)[0][1]
            w.restore(r)
            with torch.no_grad():y2=w.forward(m2,altered,'cpu',condition,False)[0][1]
            torch.testing.assert_close(y1,y2,atol=0,rtol=0)
        # End-to-end training, selection, validation replay and frozen test export.
        w.config=lambda unused:c
        def invoke(stage):
            sys.argv=['tmdc_worker.py',stage,'--dataset','mosi','--condition','TAV','--device','cpu','--data-root',str(root/'data'),'--output-root',str(root/'out')]
            with contextlib.redirect_stdout(io.StringIO()):w.main()
        invoke('train');job=root/'out/main/mosi_TAV'
        assert json.loads((job/'selection_lock.json').read_text())['best_epoch']==2
        with np.load(job/'predictions_valid.npz') as z:
            assert z['y'].tolist()==[-1,0,1,-1,0] and len(z['ids'])==5
        before=(job/'last.pt').read_bytes();invoke('train');assert before==(job/'last.pt').read_bytes()
        (job/'selection_lock.json').unlink();invoke('train');assert before==(job/'last.pt').read_bytes()
        invoke('test');assert (job/'test_done.json').exists()
        pred=(job/'predictions_test.npz').read_bytes();invoke('test');assert pred==(job/'predictions_test.npz').read_bytes()
        profile=w.profile(m2,o2,d,indices,c,a);assert profile['test_used'] is False
        # A changed protocol is rejected, instead of silently mixing outputs.
        changed=dict(c,epochs=3);w.config=lambda unused:changed
        try:invoke('train')
        except AssertionError:pass
        else:raise AssertionError('Changed protocol was accepted')
    print('PASS: official stage1/stage2 optimizer parity; six missing-channel invariance checks; padding/zero labels; validation selection/replay; no duplicate training/testing; interrupted finalization; protocol mismatch; profile train/valid only.')
if __name__=='__main__':main()

