"""Matched-encoder concatenation control: no shared/residual split, attention or ERU."""
import torch
from torch import nn
from .model import TextPresent
class SimpleFusion(TextPresent):
    def __init__(self,dims,config,h=128):
        # Keep the identical text/audio/visual encoders and 8-output distribution head.
        target=sum(p.numel() for p in TextPresent(dims,config,h).parameters())
        super().__init__(dims,{**config,'shared':False,'attention':False},h)
        del self.correct;del self.gate
        fixed=sum(p.numel() for p in self.parameters())
        # Match the removed parameter budget with one concatenation MLP, not extra evidence.
        width=max(16,round((target-fixed-8)/(2*h+11)))
        self.concat=nn.Sequential(nn.Linear(2*h,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,8))
        self.parameter_match={'c5_parameters':target,'simple_parameters':sum(p.numel() for p in self.parameters()),'concat_width':width}
    def forward(self,b,condition):
        if condition not in ('T','TA','TV'):raise ValueError(condition)
        t=self.text(b['T'])
        if condition=='T':raw=self.base(t)
        else:
            if condition=='TA':e=self.audio(b['A'])
            else:e=self.visual({**b,'VS':torch.zeros_like(b['V'])})
            raw=self.concat(torch.cat((t,e),-1))
        logits=raw[:,:7];pred=((logits.softmax(-1)*self.levels).sum(-1)+.5*raw[:,7].tanh()).clamp(-3,3)
        return {'pred':pred,'logits':logits}
