"""Independent supervised text anchor, retaining existing visible-modality increments."""
import torch
from torch import nn
from .model import mlp

class TextAnchor(nn.Module):
    def __init__(self,d=1024,h=128):
        super().__init__();self.text=mlp(d,h);self.base=nn.Linear(h,8)
        self.register_buffer('levels',torch.arange(-3.,4.))
    def forward(self,t):
        raw=self.base(self.text(t));logits=raw[:,:7]
        pred=((logits.softmax(-1)*self.levels).sum(-1)+.5*raw[:,7].tanh()).clamp(-3,3)
        return dict(pred=pred,logits=logits)

def combine(base,anchor,alpha):
    # Every condition uses the same replacement of the text anchor. Actual
    # audio/video contributions remain base[TA/TV]-base[T], then bounded output.
    delta=alpha*(anchor-base['T'])
    return {c:(p+delta).clip(-3,3) for c,p in base.items()}
