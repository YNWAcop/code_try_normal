"""OOF-supervised monotone intensity calibration with exact class preservation."""
import sys,json,itertools
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));import local_train as L
from train_distribution import K,references,select_rank
from confidence_cell_gate import predict as gate_predict

def signature(p):
    return np.column_stack((p>0,p>=0,np.rint(p.clip(-2,2)),np.rint(p.clip(-3,3))))

def cells(p):
    p=np.asarray(p,dtype='float64');assert np.isfinite(p).all() and (abs(p)<=3).all()
    r=np.rint(p).astype(int)
    i=np.where(r<0,r+3,np.where(r>0,r+5,np.where(p<0,3,np.where(p>0,5,4))))
    lo=np.array([-3,-2.5,np.nextafter(-1.5,np.inf),-.5,0,np.nextafter(0.,np.inf),np.nextafter(.5,np.inf),1.5,np.nextafter(2.5,np.inf)])
    hi=np.array([np.nextafter(-2.5,-np.inf),-1.5,np.nextafter(-.5,-np.inf),np.nextafter(0.,-np.inf),0,.5,np.nextafter(1.5,-np.inf),2.5,3])
    assert ((p>=lo[i])&(p<=hi[i])).all()
    return i,lo[i],hi[i]

def warp(p,parameters,strength):
    p=np.asarray(p,dtype='float64')
    if strength==0:return p.copy()
    i,lo,hi=cells(p);width=hi-lo;u=np.clip((p-lo)/np.maximum(width,1e-30),1e-10,1-1e-10)
    a,b=np.asarray(parameters).T
    v=np.clip(np.exp(a[i])*np.log(u/(1-u))+b[i],-40,40)
    q=lo+width/(1+np.exp(-v));q=np.clip((1-strength)*p+strength*q,lo,hi);q[i==4]=0
    assert np.array_equal(signature(p),signature(q)), 'Classification changed'
    return q

def fit(p,y,penalty):
    i,lo,hi=cells(p);width=hi-lo;u=np.clip((p-lo)/np.maximum(width,1e-30),1e-10,1-1e-10)
    ix=torch.tensor(i);lower=torch.tensor(lo);w=torch.tensor(width);logit=torch.tensor(np.log(u/(1-u)));target=torch.tensor(y,dtype=torch.float64)
    par=torch.zeros((9,2),dtype=torch.float64,requires_grad=True);opt=torch.optim.Adam([par],lr=.02)
    weights=torch.tensor(np.bincount(i,minlength=9)/len(i),dtype=torch.float64)
    for epoch in range(200):
        opt.zero_grad();q=lower+w*torch.sigmoid(par[ix,0].exp()*logit+par[ix,1])
        loss=torch.nn.functional.smooth_l1_loss(q,target,beta=.1)+penalty*(weights*(par**2).sum(-1)).sum()
        assert torch.isfinite(loss);loss.backward();opt.step()
        with torch.no_grad():par[:,0].clamp_(-np.log(4),np.log(4));par[:,1].clamp_(-2,2)
    return par.detach().numpy().tolist()

def valid_data(ds):
    with np.load(HERE/'gate_round13'/(ds+'_valid.npz')) as f:z={k:f[k] for k in f.files}
    ps,_=gate_predict(z,.5,.4)
    return z,ps

def select():
    torch.set_num_threads(4);out=HERE/'warp_round14';out.mkdir(exist_ok=True)
    protocol=dict(penalties=[.001,.01],strengths=[0.,.1,.25,.5,1.],epochs=200,loss='smoothL1 beta0.1 + occupancy-weighted identity penalty',training='group-held-out full D16 composite predictions',classification_invariant=True,code_sha=L.sha(Path(__file__)),test_used=False)
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:L.save(out/'protocol.json',protocol)
    if (out/'selection_lock.json').exists():print((out/'selection_lock.json').read_text());return
    parameters={};sources={};data={};refs={}
    for ds in ('mosi','mosei'):
        folder=HERE/'composite_crossfit'/ds
        done=json.loads((folder/'done.json').read_text())
        for n,h in done['files'].items():assert L.sha(folder/n)==h
        p=folder/'predictions_oof.npz';sources[ds]=L.sha(p)
        with np.load(p) as f:tr={k:f[k] for k in f.files}
        assert (tr['count']==1).all()
        data[ds]=valid_data(ds);refs[ds]=references(ds,data[ds][0])[0]
        parameters[ds]={str(reg):{c:fit(tr['pred_'+c],tr['y'],reg) for c in L.CS} for reg in protocol['penalties']}
    L.save(out/'parameters.json',parameters);grid=[]
    for reg,strength in itertools.product(protocol['penalties'],protocol['strengths']):
        metrics={};ranks=[]
        for ds,(z,ps) in data.items():
            metrics[ds]={c:L.scores(z['y'],warp(ps[c],parameters[ds][str(reg)][c],strength))['metrics'] for c in L.CS}
            ranks.append(select_rank(metrics[ds],refs[ds]))
        rank=[sum(r[0] for r in ranks),sum(r[1] for r in ranks),float(np.mean([r[2] for r in ranks])),min(r[3] for r in ranks)]
        grid.append(dict(penalty=reg,strength=strength,rank=rank,metrics=metrics))
    grid.sort(key=lambda a:(a['rank'],-a['strength']),reverse=True);best=grid[0];leader=json.loads((HERE/'validation_front_runner.json').read_text())
    promote=best['rank'][:2]>leader['rank'][:2] or (best['rank'][:2]==leader['rank'][:2] and best['rank'][2]>leader['rank'][2]+1e-6)
    L.save(out/'validation_grid.json',grid);L.save(out/'selection_lock.json',dict(best=best,previous=leader,promote=promote,sources=sources,parameter_sha=L.sha(out/'parameters.json'),code_sha=L.sha(Path(__file__)),test_used=False))
    print(json.dumps(dict(best={k:v for k,v in best.items() if k!='metrics'},promote=promote)))

def evaluate():
    folder=HERE/'warp_round14';decision=json.loads((folder/'selection_lock.json').read_text());assert decision['code_sha']==L.sha(Path(__file__)) and decision['parameter_sha']==L.sha(folder/'parameters.json')
    if not decision['promote']:print('No validation improvement; test not run');return
    out=HERE/'test_round14';out.mkdir(exist_ok=True)
    if (out/'done.json').exists():
        for n,h in json.loads((out/'done.json').read_text())['files'].items():assert L.sha(out/n)==h
        return
    best=decision['best'];parameters=json.loads((folder/'parameters.json').read_text());results={}
    prior=json.loads((HERE.parents[1]/'V4_test_20261121/test_comparison_audit.json').read_text(encoding='utf-8'))
    for ds in ('mosi','mosei'):
        results[ds]={}
        for c in L.CS:
            with np.load(HERE/'test_round13'/(ds+'_'+c+'.npz')) as f:z={k:f[k] for k in f.files}
            ps=warp(z['pred'],parameters[ds][str(best['penalty'])][c],best['strength']);m=L.scores(z['y'],ps)['metrics']
            wins={k:all((m[k]<prior['metrics'][ds][v][c][k]-1e-8) if k=='MAE' else (m[k]>prior['metrics'][ds][v][c][k]+1e-8) for v in ('RoME','TMDC')) for k in K}
            results[ds][c]=dict(metrics=m,beats_both=wins,count=sum(wins.values()),changed_classifications=0)
            np.savez_compressed(out/(ds+'_'+c+'.npz'),ids=z['ids'],y=z['y'],pred=ps,local_C5=z['local_C5'])
    total=sum(v['count'] for ds in results.values() for v in ds.values())
    L.save(out/'summary.json',dict(variant='D17_cell_monotone_warp',total_wins=total,total=36,target=30,passed=total>=30,results=results,exploratory_test=True))
    L.save(out/'done.json',dict(files={p.name:L.sha(p) for p in out.iterdir() if p.is_file() and p.name!='done.json'}))
    L.save(HERE/'validation_front_runner.json',dict(round='round14',variant='D17_cell_monotone_warp',rank=best['rank']))
    print(json.dumps(dict(wins=total,passed=total>=30)))

def selftest():
    endpoints=np.array([-3,-2.5,-1.5,-.5,0,.5,1.5,2.5,3.]);p=np.unique(np.concatenate((np.linspace(-3,3,6001),endpoints,np.nextafter(endpoints,np.inf).clip(-3,3),np.nextafter(endpoints,-np.inf).clip(-3,3))))
    for a,b in itertools.product([-.9,0.,.9],[-2.,0.,2.]):
        q=warp(p,[[a,b]]*9,1.);assert (np.diff(q)>=0).all()
    print('PASS: endpoint ties, zero signs, Acc5/7 and monotonicity')
if __name__=='__main__':
    if len(sys.argv)==1:select()
    elif sys.argv[1]=='selftest':selftest()
    else:assert sys.argv[1]=='evaluate';evaluate()
