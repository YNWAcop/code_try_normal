"""Ordinal category and within-category intensity, including zero-bin polarity."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import mlp

class CellTextAnchor(nn.Module):
    def __init__(self,d=1024,h=128):
        super().__init__();self.text=mlp(d,h);self.base=nn.Linear(h,15)
        lo=np.array([-3,-2.5,-1.5,-.5,.5,1.5,2.5],dtype='float32')
        hi=np.array([-2.5,-1.5,-.5,.5,1.5,2.5,3],dtype='float32')
        for j in (2,4,6):lo[j]=np.nextafter(lo[j],np.float32(np.inf))
        for j in (0,2,4):hi[j]=np.nextafter(hi[j],np.float32(-np.inf))
        self.register_buffer('lo',torch.from_numpy(lo));self.register_buffer('hi',torch.from_numpy(hi))
    def forward(self,t):
        raw=self.base(self.text(t));logits=raw[:,:7];rho=raw[:,7:14];z=raw[:,14]
        components=(self.lo+(self.hi-self.lo)*rho.sigmoid()).maximum(self.lo).minimum(self.hi)
        magnitude=(.5*rho[:,3].sigmoid()).clamp_min(1e-7)
        hard_zero=torch.where(z>=0,magnitude,-magnitude)
        components=torch.cat((components[:,:3],hard_zero[:,None],components[:,4:]),-1)
        soft_components=torch.cat((components[:,:3],(z.tanh()*magnitude)[:,None],components[:,4:]),-1)
        pred=components.gather(1,logits.argmax(-1,keepdim=True)).squeeze(1)
        return dict(pred=pred,soft_pred=(logits.softmax(-1)*soft_components).sum(-1),logits=logits,components=components,zero_logit=z,zero_magnitude=magnitude)

def cell_objective(out,y,config=None):
    cls=y.clip(-3,3).round().long()+3
    prob=out['logits'].softmax(-1);target=F.one_hot(cls,7).float()
    loss=F.smooth_l1_loss(out['soft_pred'],y,beta=.55)+.35*F.cross_entropy(out['logits'],cls)
    loss=loss+.1*F.mse_loss(prob.cumsum(-1)[:,:-1],target.cumsum(-1)[:,:-1])
    chosen=out['components'].gather(1,cls[:,None]).squeeze(1);zero=cls==3
    within=F.smooth_l1_loss(chosen,y,beta=.25,reduction='none')
    within=torch.where(zero,F.smooth_l1_loss(out['zero_magnitude'],y.abs(),beta=.25,reduction='none'),within)
    loss=loss+.5*within.mean()
    mask=zero&(y!=0)
    if mask.any():loss=loss+.2*F.binary_cross_entropy_with_logits(out['zero_logit'][mask],(y[mask]>0).float())
    nz=y!=0
    if nz.any():
        pos=prob[:,4:].sum(-1)+prob[:,3]*out['zero_logit'].sigmoid()
        neg=prob[:,:3].sum(-1)+prob[:,3]*(1-out['zero_logit'].sigmoid())
        loss=loss+.15*F.binary_cross_entropy_with_logits((pos.clamp_min(1e-8).log()-neg.clamp_min(1e-8).log())[nz],(y[nz]>0).float())
    return loss
