"""Trainable visible-feature evidence, anchored to a frozen C5 score distribution.

D4 is a concatenation control; D5 adds explicit rank-64 multiplicative evidence.
These are experimental designs, not a claim of literature novelty.
"""
from .adapters import DistributionAdapter
import torch
from torch import nn

def projection(d,h):
    return nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(.15))

class EvidenceHead(nn.Module):
    def __init__(self,dims,c,bilinear):
        super().__init__();self.c=c;self.bilinear=bilinear;h=64
        self.text=projection(dims['T'],h)
        if c!='T':
            width_in = dims['A'] if c=='TA' else dims['V']+dims['Q']+dims['timeline']
            if c=='TAV': width_in += dims['A']
            self.modality=projection(width_in,h)
        width=h if c=='T' else h*(4 if bilinear else 2)
        self.output=nn.Sequential(nn.Linear(width+8,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,8))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
    def forward(self,b,raw):
        t=self.text(b['T']);features=[t]
        if self.c!='T':
            x=b['A']-b['AS'] if self.c=='TA' else torch.cat((b['V']-b['VS'],b['Q'],b['timeline'].mean(1)),-1)
            if self.c=='TAV':x=torch.cat((b['A']-b['AS'],x),-1)
            m=self.modality(x);features.append(m)
            if self.bilinear:features.extend((t*m,t-m))
        return self.output(torch.cat(features+[raw],-1))

class RawEvidenceAdapter(DistributionAdapter):
    def __init__(self,ck,variant):
        super().__init__(ck,'D2_logits');self.variant=variant
        self.heads=nn.ModuleDict({c:EvidenceHead(ck['dims'],c,variant=='D5_bilinear') for c in ('T','TA','TV','TAV')})
    def forward(self,b,c):
        got={}
        def capture(k):
            def hook(module,args,out):got[k]=out
            return hook
        hooks=[getattr(self.core,k).register_forward_hook(capture(k)) for k in ('base','correct','gate')]
        try:
            with torch.no_grad():original=self.core(b,c)
        finally:
            for h in hooks:h.remove()
        raw=got['base'] if c=='T' else got['base']+got['gate'].sigmoid()*got['correct']
        delta=self.heads[c](b,raw)
        # Bound residual capacity, retaining an exact C5 candidate at zero.
        adjusted=raw+.5*delta.tanh()
        logits=adjusted[:,:7];q=logits.softmax(-1)
        pred=((q*self.core.levels).sum(-1)+.5*adjusted[:,7].tanh()).clamp(-3,3)
        return dict(pred=pred,logits=logits,probabilities=q,base_pred=original['pred'])
