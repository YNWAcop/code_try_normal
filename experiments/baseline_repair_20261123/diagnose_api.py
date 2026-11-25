"""API fixed-condition core-mechanism checks on synthetic data; no score tuning."""
import importlib,importlib.util,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
UP=Path('D:/MSA_Cloud_20261115/baselines/upstream/API')
sys.path.insert(0,str(UP));os.environ['HF_HUB_OFFLINE']='1';torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('easydict','E:/anaconda/envs/mult-repro-mulT/Lib/site-packages/easydict/__init__.py');easy=importlib.util.module_from_spec(spec);sys.modules['easydict']=easy;spec.loader.exec_module(easy)
from config import get_config_regression
mod=importlib.import_module('trains.singleTask.model.april')
source=Path(mod.__file__).read_text(encoding='utf-8')
needle='        ava_modal_idx = sample(modal_idx, num_modal)'
assert source.count(needle)==1
exec(compile(source.replace(needle,'        ava_modal_idx = DIAGNOSTIC_AVAILABLE\n        num_modal = len(ava_modal_idx)'),mod.__file__,'exec'),mod.__dict__)
a=get_config_regression('april','mosi',UP/'config/config_april.json');a.pretrained='D:/MSA_Cloud_20261115/models/bert-base-uncased';a.device=torch.device('cpu');a.train_mode='regression';a.is_training=True
model=mod.APRIL(a);records=[]
t=torch.zeros(7,3,50);t[:,0]=103;t[:,0,0]=101;t[:,0,-1]=102;t[:,1]=1
x=torch.randn(7,50,5);v=torch.randn(7,50,20);y=torch.arange(-3,4,dtype=torch.float32).view(-1,1)
mod.DIAGNOSTIC_AVAILABLE=[2]
model.train()
with torch.no_grad():model(t,x,v,num_modal=1,labels=y)
model.update_prototypes_epoch_end()
assert model._all_prototypes_ready()
assert all(k in model.state_dict() for k in ('proto_text','proto_audio','proto_vision','proto_ready_audio'))
for cond,indices in [('A',[2]),('V',[1])]:
    mod.DIAGNOSTIC_AVAILABLE=indices;model.train();model.zero_grad()
    out=model(t,x,v,num_modal=1,labels=y)['output_logit']
    recon=model.personal_recon_loss;contrast=model.prototype_contrastive_loss
    loss=(out-y).square().mean()+recon+contrast
    assert torch.isfinite(loss);loss.backward()
    prefix='proj_a' if cond=='A' else 'proj_v'
    grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if prefix in n and p.grad is not None}
    assert grads and max(grads.values())>0
    model.eval()
    with torch.no_grad():
        pred=model(t,x,v,num_modal=1,labels=None)['output_logit']
        t2=t.clone();t2[:,0,1:-1]=104
        changed=model(t2,x if cond=='A' else x*7,v if cond=='V' else v*9,num_modal=1,labels=None)['output_logit']
    delta=float((pred-changed).abs().max());assert delta==0,(cond,delta)
    records.append({'condition':cond,'projection_gradient':grads,'prototype_contrastive_loss':float(contrast.detach()),'six_way_reconstruction_loss':float(recon.detach()),'hidden_input_delta':delta})
result={'passed':True,'scope':'Author network with same fixed TVA-index adaptation; synthetic complete training views, no labels at inference; does not establish convergence of historical run','all_seven_class_prototypes_ready':True,'prototype_buffers_checkpointed':True,'checks':records}
(ROOT/'api_mechanism_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
