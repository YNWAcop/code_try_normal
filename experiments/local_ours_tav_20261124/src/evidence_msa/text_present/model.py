"""V1-inspired text-present branch. No upstream RoME/TMDC code is copied here.

Shared/residual evidence and shallow interaction are hypotheses, not claimed gains.
Every metric uses the same bounded scalar output. No absent input is evaluated.
"""
import torch
from torch import nn
from torch.nn import functional as F

def mlp(d, h, out=None):
    return nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(.2),
                         nn.Linear(h, h if out is None else out))

class TextPresent(nn.Module):
    def __init__(self, dims, config, h=128):
        super().__init__(); self.config = dict(config); self.dims = dims; self.h = h
        self.text = mlp(dims['T'], h)
        self.audio = mlp(dims['A'], h)
        self.vapp = mlp(dims['app'], h)
        self.vlex = mlp(dims['V'] - dims['app'], h)
        self.timeline = nn.Sequential(nn.Conv1d(dims['timeline'], h, 3, padding=1), nn.GELU())
        self.query = nn.Linear(h, h)
        self.visual_gate = nn.Sequential(nn.Linear(2*h + dims['Q'], h), nn.Sigmoid())
        if config['shared']:
            self.shared_a = mlp(dims['A'], h); self.shared_v = mlp(dims['V'], h)
        if config['attention']:
            self.interaction = nn.MultiheadAttention(h, 4, dropout=.1, batch_first=True)
            self.inter_norm = nn.LayerNorm(h)
        self.base = nn.Linear(h, 8 if config['distribution'] else 1)
        self.correct = mlp(3*h, h, 8 if config['distribution'] else 1)
        self.gate = nn.Linear(3*h, 1)
        # Begin as a text predictor. Trainable corrections can improve or hurt it.
        nn.init.zeros_(self.correct[-1].weight); nn.init.zeros_(self.correct[-1].bias)
        self.register_buffer('levels', torch.arange(-3., 4.))

    def visual(self, b):
        v = b['V'] - b['VS']; app = self.vapp(v[:, :self.dims['app']])
        lex = self.vlex(v[:, self.dims['app']:])
        states = self.timeline(b['timeline'].transpose(1, 2)).transpose(1, 2)
        weights = (self.query(lex)[:, None] * states).sum(-1).softmax(1)
        lex = lex + (weights[:, :, None] * states).sum(1)
        gate = self.visual_gate(torch.cat([app, lex, b['Q']], -1))
        return app*(1-gate) + lex*gate

    def forward(self, b, condition):
        if condition not in ('T', 'TA', 'TV', 'TAV'):
            raise ValueError('Route A/V/AV to the frozen V1 model, never this model')
        t = self.text(b['T']); base = self.base(t); raw = base
        if condition != 'T':
            residuals = []
            if 'A' in condition: residuals.append(self.audio(b['A']-b['AS']))
            if 'V' in condition: residuals.append(self.visual(b))
            residual = torch.stack(residuals).mean(0) if len(residuals)>1 else residuals[0]
            shared = torch.zeros_like(t)
            if self.config['shared']:
                shared_parts = []
                if 'A' in condition: shared_parts.append(self.shared_a(b['AS']))
                if 'V' in condition: shared_parts.append(self.shared_v(b['VS']))
                shared = torch.stack(shared_parts).mean(0) if len(shared_parts)>1 else shared_parts[0]
            context_t = t
            if self.config['attention']:
                # Only genuinely visible evidence participates; no zero placeholder token.
                tokens = [t] + residuals + (shared_parts if self.config['shared'] else [])
                tokens = torch.stack(tokens, dim=1)
                interacted, _ = self.interaction(t[:, None], tokens, tokens, need_weights=False)
                context_t = self.inter_norm(t + interacted[:, 0])
            context = torch.cat([context_t, residual, shared], -1)
            correction = self.gate(context).sigmoid() * self.correct(context)
            raw = base + correction
        if self.config['distribution']:
            logits, refine = raw[:, :7], raw[:, 7]
            pred = ((logits.softmax(-1) * self.levels).sum(-1) + .5*refine.tanh()).clamp(-3, 3)
        else:
            logits = None
            # Matched regression control, with bounded final prediction.
            pred = 3*torch.tanh(raw[:, 0]/3)
        return dict(pred=pred, logits=logits)

def objective(out, y, config):
    p = out['pred']
    value = F.smooth_l1_loss(p, y, beta=.55)
    if config['distribution']:
        # Triangular soft target retains fractional sentiment intensity.
        target = (1-(y[:, None]-torch.arange(-3., 4., device=y.device)).abs()).clamp_min(0)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
        logp = out['logits'].log_softmax(-1)
        ce = -(target*logp).sum(-1).mean()
        cdf = F.mse_loss(logp.exp().cumsum(-1)[:, :-1], target.cumsum(-1)[:, :-1])
        value = value + .3*ce + .2*cdf
    else:
        bounds = torch.arange(-2.5, 3., device=y.device)
        value = value + .3*F.binary_cross_entropy_with_logits(
            (p[:, None]-bounds)/.6, (y[:, None]>bounds).float())
    if config['polarity']:
        nz = y != 0
        if nz.any():
            value = value + .1*F.binary_cross_entropy_with_logits(p[nz]/.6, (y[nz]>0).float())
    return value
