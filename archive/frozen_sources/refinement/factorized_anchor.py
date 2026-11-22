"""Supervised polarity/magnitude factorization; no teacher or hidden modalities."""
import torch
from torch import nn
from model import mlp

class FactorAnchor(nn.Module):
    def __init__(self,d=1024,h=128):
        super().__init__();self.text=mlp(d,h);self.head=nn.Linear(h,2)
    def forward(self,t):
        z=self.head(self.text(t));mag=3*z[:,1].sigmoid();pol=z[:,0]
        return dict(pred=mag*(pol/2).tanh(),polarity=pol,magnitude=mag)

def objective(o,y):
    loss=nn.functional.smooth_l1_loss(o['pred'],y,beta=.3)+.2*nn.functional.smooth_l1_loss(o['magnitude'],y.abs(),beta=.3)
    nz=y!=0
    if nz.any():loss=loss+.25*nn.functional.binary_cross_entropy_with_logits(o['polarity'][nz],(y[nz]>0).float())
    return loss
