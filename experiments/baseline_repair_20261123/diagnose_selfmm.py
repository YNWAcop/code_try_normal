"""Real Self-MM audio-only gradient, native pseudo-label, and update checks."""
import argparse,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'baseline_handoff_20261115/architecture_checks/Self_MM'))
os.environ['HF_HUB_OFFLINE']='1';torch.set_num_threads(2)
from config.config_regression import ConfigRegression
from models.AMIO import AMIO
from trains.multiTask.SELF_MM import SELF_MM
a=ConfigRegression(argparse.Namespace(modelName='self_mm',datasetName='mosi',train_mode='regression',num_workers=0,cur_time=1,excludeZero=False)).get_config()
a.train_samples=4;a.device=torch.device('cpu')
model=AMIO(a).train();trainer=SELF_MM(a);idx=torch.arange(4);y=torch.tensor([-2.,-1.,1.,2.]);trainer.init_labels(idx,y)
t=torch.zeros(4,3,50);t[:,0,0]=101;t[:,0,1]=102;t[:,1,:2]=1
x=torch.randn(4,50,5,requires_grad=True);v=torch.zeros(4,50,20);al=torch.tensor([50,47,45,43]);vl=torch.ones(4,dtype=torch.long)
out=model(t,(x,al),(v,vl))
features=[out['Feature_f'].detach(),out['Feature_t'].detach(),out['Feature_a'].detach(),out['Feature_v'].detach()]
trainer.update_features(*features,idx);trainer.update_centers();trainer.update_labels(*features,2,idx,out)
assert all(torch.isfinite(z).all() for z in trainer.label_map.values())
parts={m:trainer.weighted_loss(out[m],trainer.label_map[trainer.name_map[m]][idx],idx,trainer.name_map[m]) for m in 'MTAV'}
loss=sum(parts.values());loss.backward();assert x.grad.norm()>0
grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if 'audio_model' in n and p.grad is not None}
assert max(grads.values())>0
opt=torch.optim.Adam(model.parameters(),lr=1e-4);param=next(p for n,p in model.named_parameters() if 'audio_model' in n and p.grad is not None and p.grad.norm()>0);before=param.detach().clone();opt.step();delta=float((before-param).abs().max());assert delta>0
result={'passed':True,'scope':'Synthetic audio-only input; real local BERT and original four-head model, pseudo-label and weighted losses; not a replay of unavailable historical weights','audio_gradient_norm':float(x.grad.norm()),'audio_parameter_update_max':delta,'losses':{k:float(v.detach()) for k,v in parts.items()},'pseudo_labels':{k:v.tolist() for k,v in trainer.label_map.items()}}
(ROOT/'selfmm_mechanism_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
