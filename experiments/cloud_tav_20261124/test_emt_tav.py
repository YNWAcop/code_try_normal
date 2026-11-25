"""Actual EMT layers with a lightweight BERT substitute, no test data."""
import argparse
import copy
import json
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'four/vendor/emt_dlfr'))
from config.config_regression import ConfigRegression
import importlib
module = importlib.import_module('models.missingTask.EMT_DLFR')
from trains.missingTask.EMT_DLFR import ReconLoss

class FakeBert(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(110, 768)
    def forward(self, x):
        return self.embedding(x[:, 0].long())

module.BertTextEncoder = FakeBert
torch.set_num_threads(2)
args = ConfigRegression(argparse.Namespace(modelName='emt-dlfr', datasetName='mosi',
    train_mode='regression', fusion_method='emt', seed='unseeded', seeds=['unseeded'],
    num_seeds=1, cur_time=0, use_normalized_data=False)).get_config()
args.update(feature_dims=(768, 3, 4), d_model=16, audio_out=16, video_out=16,
    a_lstm_hidden_size=16, v_lstm_hidden_size=16, fusion_layers=1, heads=2,
    a_lstm_dropout=0., v_lstm_dropout=0., emb_dropout=0., attn_dropout=0.,
    ff_dropout=0., post_fusion_dropout=0.)
records=[]
for cond in ('TAV',):
    args.protocol_condition=cond
    model=module.EMT_DLFR(args).eval()
    text=torch.zeros(2,3,8);text[:,0]=torch.randint(3,100,(2,8));text[:,1]=1
    audio=torch.randn(2,9,3);video=torch.randn(2,7,4)
    al=torch.tensor([9,5]);vl=torch.tensor([7,4])
    tm=text.clone();am=audio.clone();vm=video.clone()
    if 'T' not in cond:
        tm.zero_();tm[:,0,0]=101;tm[:,0,1]=102;tm[:,1,:2]=1
    if 'A' not in cond:am.zero_()
    if 'V' not in cond:vm.zero_()
    out=model((text,tm),(audio,am,al),(video,vm,vl))
    assert all(torch.isfinite(v).all() for v in out.values())
    assert out['audio_recon'].shape[1]==9 and out['video_recon'].shape[1]==7
    text2=text.clone();audio2=audio.clone();video2=video.clone()
    al2=al.clone();vl2=vl.clone()
    if 'T' not in cond:text2[:,0]=torch.randint(3,100,(2,8));text2[1,1,3:]=0
    if 'A' not in cond:audio2.normal_();al2=torch.tensor([2,3])
    if 'V' not in cond:video2.normal_();vl2=torch.tensor([2,3])
    altered=model((text2,tm),(audio2,am,al2),(video2,vm,vl2))
    delta=(out['pred_m']-altered['pred_m']).abs().max().item()
    assert delta==0, (cond,'hidden content/length leak',delta)
    grads={}
    for key,target,lengths in [('audio',audio,al),('video',video,vl)]:
        modality='A' if key=='audio' else 'V'
        if modality in cond:continue
        pred=out[key+'_recon'];pred.retain_grad()
        mask=(torch.arange(target.shape[1])[None,:]<lengths[:,None]).float()
        loss=ReconLoss('MSELoss')(pred,target,mask)
        model.zero_grad();loss.backward(retain_graph=True)
        assert pred.grad[:,1:].abs().sum()>0, 'Later reconstruction positions unsupervised'
        weight=getattr(model,'recon_'+key).weight
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum()>0
        grads[key]=float(weight.grad.norm())
    records.append({'condition':cond,'hidden_prediction_delta':delta,
        'reconstruction_lengths':[out['audio_recon'].shape[1],out['video_recon'].shape[1]],
        'missing_decoder_gradient_norm':grads})
result={'passed':True,'scope':'Actual EMT fusion/LSTM/projectors/decoder; fake BERT only; no benchmark test samples', 'checks':records}
# Check recomputation does not update BatchNorm twice or change dropout gradients.
model.train();other=copy.deepcopy(model);other.activation_checkpointing=True
rng=torch.get_rng_state()
a=model((text,tm),(audio,am,al),(video,vm,vl))
loss_a=sum(v.square().mean() for k,v in a.items() if not k.startswith('z_') and k!='text_for_recon')
loss_a.backward()
torch.set_rng_state(rng)
b=other((text,tm),(audio,am,al),(video,vm,vl))
loss_b=sum(v.square().mean() for k,v in b.items() if not k.startswith('z_') and k!='text_for_recon')
loss_b.backward()
assert torch.equal(a['pred_m'],b['pred_m'])
max_delta=0.
for (name,p),(name2,q) in zip(model.named_parameters(),other.named_parameters()):
    assert name==name2
    if p.grad is None:assert q.grad is None;continue
    assert torch.allclose(p.grad,q.grad,rtol=1e-4,atol=1e-6),name
    max_delta=max(max_delta,float((p.grad-q.grad).abs().max()))
for (name,p),(name2,q) in zip(model.named_buffers(),other.named_buffers()):
    assert name==name2 and torch.equal(p,q),name
result['fusion_checkpoint_gradient_parity']={'passed':True,'max_abs_delta':max_delta,'batchnorm_buffers_identical':True}
(ROOT/'emt_contract_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
