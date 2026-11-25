"""Frozen C5: unconstrained logits versus conservative neighboring mass flows."""
import sys
from pathlib import Path

import torch
from torch import nn
from .model import TextPresent

class DistributionAdapter(nn.Module):
    def __init__(self,ck,variant):
        super().__init__();self.variant=variant
        self.core=TextPresent(ck['dims'],ck['config'],ck['hidden']);self.core.load_state_dict(ck['model'])
        for p in self.core.parameters():p.requires_grad_(False)
        h=ck['hidden']; width=7 if variant=='D2_logits' else 6
        self.heads=nn.ModuleDict({c:nn.Sequential(nn.Linear(3*h+8,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,width)) for c in ('T','TA','TV')})
        for head in self.heads.values():nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def train(self,mode=True):
        super().train(mode);self.core.eval();return self
    def forward(self,b,c):
        got={}
        def capture(key):
            def fn(module,args,out):got[key]=out
            return fn
        def context(module,args):got['context']=args[0]
        hooks=[self.core.text.register_forward_hook(capture('text')),self.core.base.register_forward_hook(capture('base')),
               self.core.correct.register_forward_hook(capture('correct')),self.core.gate.register_forward_hook(capture('gate')),
               self.core.gate.register_forward_pre_hook(context)]
        try:
            with torch.no_grad():original=self.core(b,c)
        finally:
            for hook in hooks:hook.remove()
        t=got['text'];ctx=got.get('context',torch.cat((t,torch.zeros_like(t),torch.zeros_like(t)),-1))
        raw=got['base'] if c=='T' else got['base']+got['gate'].sigmoid()*got['correct']
        x=torch.cat((ctx,raw),-1);delta=self.heads[c](x);p=original['logits'].softmax(-1)
        if self.variant=='D2_logits':
            logits=original['logits']+delta;q=logits.softmax(-1)
        else:
            # A signed edge flow is bounded by 1/4 of each neighboring mass.
            # At most two outgoing edges: q_i >= p_i/2; sum(q)=sum(p).
            f=.25*delta.tanh()*p[:,:-1]*p[:,1:]/(p[:,:-1]+p[:,1:]+1e-12)
            q=p+torch.nn.functional.pad(f,(1,0))-torch.nn.functional.pad(f,(0,1))
            logits=q.clamp_min(1e-30).log()
        pred=((q*self.core.levels).sum(-1)+.5*raw[:,7].tanh()).clamp(-3,3)
        return dict(pred=pred,logits=logits,probabilities=q,base_pred=original['pred'])
