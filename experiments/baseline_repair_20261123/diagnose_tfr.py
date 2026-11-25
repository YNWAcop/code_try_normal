"""TFR reconstruction coverage and hidden-input isolation, synthetic CPU check."""
import argparse,importlib,json,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'vendor/emt_dlfr'))
from config.config_regression import ConfigRegression
mod=importlib.import_module('models.missingTask.TFR_NET.model')
class FakeBert(nn.Module):
    def __init__(self,**kw):super().__init__();self.emb=nn.Embedding(110,768)
    def forward(self,t):return self.emb(t[:,0].long())
mod.BertTextEncoder=FakeBert;torch.set_num_threads(2)
a=ConfigRegression(argparse.Namespace(modelName='tfr_net',datasetName='mosei',train_mode='regression',fusion_method='emt',seed='unseeded',seeds=['unseeded'],num_seeds=1,cur_time=0,use_normalized_data=False)).get_config()
a.device=torch.device('cpu');a.protocol_condition='A';a.seq_lens=(50,50,50)
model=mod.TFR_NET(a).eval();b=2
t=torch.zeros(b,3,50);t[:,0]=103;t[:,1]=1;tm=torch.zeros_like(t);tm[:,0,0]=101;tm[:,0,1]=102;tm[:,1,:2]=1
x=torch.randn(b,50,a.feature_dims[1],requires_grad=True);v=torch.randn(b,50,a.feature_dims[2]);mask=torch.ones(b,50);zero=torch.zeros_like(mask)
captured={}
def hook(name):
    def fn(m,inp,out):out.retain_grad();captured[name]=out
    return fn
model.generator_v.register_forward_hook(hook('vision'))
pred,loss=model((t,tm,zero),(x,x,mask,mask),(v,torch.zeros_like(v),mask,zero))
assert torch.isfinite(loss).all();loss.sum().backward(retain_graph=True)
grad=captured['vision'].grad;assert grad[:,1:].abs().sum()>0
generator_norm=sum(float(p.grad.norm()) for p in model.generator_v.parameters() if p.grad is not None)
model.zero_grad();pred.square().mean().backward();audio_grad=float(x.grad.norm());assert audio_grad>0
t2=t.clone();t2[:,0]=104
altered,_=model((t2,tm,zero),(x,x,mask,mask),(v*13,torch.zeros_like(v),mask,zero))
delta=float((pred-altered).abs().max());assert delta==0
result={'passed':True,'scope':'Actual crossmodal alignment, generators, C_GATE; fake BERT; synthetic MOSEI A only, not historical convergence proof','hidden_input_delta':delta,'missing_vision_reconstruction_length':int(grad.shape[1]),'later_positions_gradient_norm':float(grad[:,1:].norm()),'generator_gradient_norm':generator_norm,'audio_prediction_gradient_norm':audio_grad,'use_gen_fusion':a.use_gen_fusion,'without_generator':a.without_generator}
(ROOT/'tfr_mechanism_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
