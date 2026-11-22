"""Fold-local D24 postprocessors. No legacy checkpoint or prediction access.

OOF inputs train every fitted map. Only inner validation selects epochs.
Inference accepts feature dictionaries without labels. No seed is set.
"""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn

CS = ('T', 'TA', 'TV')

def signature(p):
    p = np.asarray(p)
    return np.column_stack((p > 0, p >= 0, np.rint(p.clip(-2, 2)), np.rint(p.clip(-3, 3))))

def cells(p):
    p = np.asarray(p, dtype='float64')
    assert np.isfinite(p).all() and (abs(p) <= 3).all()
    r = np.rint(p).astype(int)
    i = np.where(r < 0, r+3, np.where(r > 0, r+5, np.where(p < 0, 3, np.where(p > 0, 5, 4))))
    lo = np.array([-3, -2.5, np.nextafter(-1.5, np.inf), -.5, 0, np.nextafter(0., np.inf), np.nextafter(.5, np.inf), 1.5, np.nextafter(2.5, np.inf)])
    hi = np.array([np.nextafter(-2.5, -np.inf), -1.5, np.nextafter(-.5, -np.inf), np.nextafter(0., -np.inf), 0, .5, np.nextafter(1.5, -np.inf), 2.5, 3])
    assert ((p >= lo[i]) & (p <= hi[i])).all()
    return i, lo[i], hi[i]

def warp(p, parameters):
    p = np.asarray(p, dtype='float64'); i, lo, hi = cells(p)
    width = hi-lo; u = np.clip((p-lo)/np.maximum(width, 1e-30), 1e-10, 1-1e-10)
    a, b = np.asarray(parameters).T
    v = np.clip(np.exp(a[i])*np.log(u/(1-u))+b[i], -40, 40)
    q = np.clip(.5*p+.5*(lo+width/(1+np.exp(-v))), lo, hi); q[i == 4] = 0
    assert np.array_equal(signature(p), signature(q))
    return q

def fit_warp(p, y):
    i, lo, hi = cells(p); width = hi-lo
    u = np.clip((p-lo)/np.maximum(width, 1e-30), 1e-10, 1-1e-10)
    ix = torch.tensor(i); lower = torch.tensor(lo); w = torch.tensor(width)
    logit = torch.tensor(np.log(u/(1-u))); target = torch.tensor(y, dtype=torch.float64)
    par = torch.zeros((9, 2), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([par], lr=.02)
    weights = torch.tensor(np.bincount(i, minlength=9)/len(i), dtype=torch.float64)
    for _ in range(200):
        opt.zero_grad(); q = lower+w*torch.sigmoid(par[ix, 0].exp()*logit+par[ix, 1])
        loss = nn.functional.smooth_l1_loss(q, target, beta=.1)+.001*(weights*(par**2).sum(-1)).sum()
        assert torch.isfinite(loss); loss.backward(); opt.step()
        with torch.no_grad():
            par[:, 0].clamp_(-np.log(4), np.log(4)); par[:, 1].clamp_(-2, 2)
    return par.detach().numpy()

class Gate(nn.Module):
    def __init__(self, d=37):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Dropout(.15), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return 1+self.net(x).squeeze(-1).tanh()

class Intensity(nn.Module):
    def __init__(self, d=52):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Dropout(.1), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return 2*self.net(x).squeeze(-1).tanh()

def features(z, base, c):
    i, lo, hi = cells(base[c]); width = hi-lo
    u = np.clip((base[c]-lo)/np.maximum(width, 1e-30), 1e-7, 1-1e-7)
    x = np.concatenate((z['f_'+c], np.eye(9)[i], np.tile(np.eye(3)[CS.index(c)], (len(i), 1)), np.column_stack((base[c], u, width))), -1).astype('float32')
    return x, lo, hi, np.log(u/(1-u))

def correct(p, lo, hi, logit, delta):
    q = np.clip(lo+(hi-lo)/(1+np.exp(-np.clip(logit+delta, -40, 40))), lo, hi)
    assert np.array_equal(signature(p), signature(q))
    return q

def snapshot(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(value, tmp); os.replace(tmp, path)
def load(path): return torch.load(path, map_location='cpu', weights_only=False)
def rng():
    return dict(py=random.getstate(), np=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore(r):
    random.setstate(r['py']); np.random.set_state(r['np']); torch.set_rng_state(r['cpu'])
    if r['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(r['cuda'])
def resume(model, opt, path, device):
    if not path.exists(): return 0, None
    ck = load(path); model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); restore(ck['rng'])
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v): st[k] = v.to(device)
    return ck['epoch']+1, ck['best']

def gate_loss(p, y, g):
    loss = nn.functional.smooth_l1_loss(p, y, beta=.55)+.01*((g-1)**2).mean(); nz = y != 0
    if nz.any(): loss = loss+.2*nn.functional.binary_cross_entropy_with_logits(p[nz]/.5, (y[nz] > 0).float())
    bounds = torch.tensor([-2.5, -1.5, -.5, .5, 1.5, 2.5], device=p.device)
    return loss+.1*nn.functional.binary_cross_entropy_with_logits((p[:, None]-bounds)/.35, (y[:, None] > bounds).float())

@torch.no_grad()
def gated(z, model, mean, scale, pars, device):
    model.eval(); ps = {'T': np.asarray(z['pred_T'], dtype='float64')}
    for c in ('TA', 'TV'):
        x = torch.tensor((z['f_'+c]-mean[c])/scale[c], device=device)
        g = model[c](x).cpu().numpy().astype('float64')
        ps[c] = (ps['T']+(1+.5*(g-1))*(z['pred_'+c]-ps['T'])).clip(-3, 3)
    return {c: warp(ps[c], pars[c]) for c in CS}

def consensus(base, fold_mean):
    result = {}
    for c in CS:
        _, lo, hi = cells(base[c]); result[c] = np.clip(.5*base[c]+.5*fold_mean[c], lo, hi)
        assert np.array_equal(signature(base[c]), signature(result[c]))
    return result

@torch.no_grad()
def intensify(z, base, model, mean, scale, device):
    model.eval(); ps = {}
    for c in CS:
        x, lo, hi, lg = features(z, base, c)
        delta = model(torch.tensor((x-mean)/scale, device=device)).cpu().numpy().astype('float64')
        ps[c] = correct(base[c], lo, hi, lg, delta)
    return ps

def input_hash(z):
    h = hashlib.sha256()
    for k in ('ids', 'y')+tuple('pred_'+c for c in CS)+tuple('f_'+c for c in CS):
        a = np.asarray(z[k]); assert a.dtype.kind != 'O'
        h.update(k.encode()); h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()

def fit_postprocessors(train_z, valid_z, valid_fold_mean, output_dir, rank_fn, status_fn=None, device='cuda'):
    """Return portable CPU state. rank_fn(predictions, y) -> scalar, larger better.

    status_fn, if supplied, receives one dict. Valid/held scores are never printed.
    Epoch zero is a candidate. Resume includes RNG state, never seed reselection.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    assert not set(map(str, train_z['ids'])) & set(map(str, valid_z['ids']))
    contract = dict(schema=1, train_sha=input_hash(train_z), valid_sha=input_hash(valid_z),
                    fold_mean_sha=hashlib.sha256(b''.join(np.asarray(valid_fold_mean[c]).tobytes() for c in CS)).hexdigest(),
                    code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), no_manual_seed=True,
                    warp_penalty=.001, warp_strength=.5, gate_strength=.5, consensus_strength=.5, intensity_strength=1.)
    cp = out/'contract.json'
    if cp.exists(): assert json.loads(cp.read_text()) == contract, 'Postprocessor inputs or implementation changed'
    else: cp.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    if (out/'complete.pt').exists(): return load(out/'complete.pt')
    def notify(component, epoch, epochs):
        if status_fn: status_fn(dict(stage='calibration_training', component=component, epoch=epoch, epochs=epochs))
    wp = out/'warp.pt'
    if wp.exists(): pars = load(wp)
    else:
        pars = {c: fit_warp(train_z['pred_'+c], train_z['y']) for c in CS}; save(wp, pars)
    mean = {c: train_z['f_'+c].mean(0) for c in ('TA', 'TV')}
    scale = {c: train_z['f_'+c].std(0).clip(.05, None) for c in ('TA', 'TV')}
    gates = nn.ModuleDict({c: Gate() for c in ('TA', 'TV')}).to(device)
    opt = torch.optim.AdamW(gates.parameters(), lr=.001, weight_decay=.01)
    x = {c: torch.tensor((train_z['f_'+c]-mean[c])/scale[c], device=device) for c in ('TA', 'TV')}
    y = torch.tensor(train_z['y'], dtype=torch.float32, device=device)
    t = torch.tensor(train_z['pred_T'], dtype=torch.float32, device=device)
    inc = {c: torch.tensor(train_z['pred_'+c]-train_z['pred_T'], dtype=torch.float32, device=device) for c in ('TA', 'TV')}
    start, best = resume(gates, opt, out/'gate_last.pt', device)
    for epoch in range(start, 31):
        if epoch:
            gates.train()
            for ix in torch.randperm(len(y), device=device).split(128):
                opt.zero_grad(set_to_none=True)
                for c in ('TA', 'TV'):
                    g = gates[c](x[c][ix]); loss = gate_loss((t[ix]+g*inc[c][ix]).clamp(-3, 3), y[ix], g)/2
                    assert torch.isfinite(loss); loss.backward()
                torch.nn.utils.clip_grad_norm_(gates.parameters(), 5); opt.step()
        score = rank_fn(gated(valid_z, gates, mean, scale, pars, device), valid_z['y'])
        if best is None or score > best['rank']: best = dict(rank=score, epoch=epoch, model=snapshot(gates))
        save(out/'gate_last.pt', dict(model=snapshot(gates), opt=opt.state_dict(), rng=rng(), epoch=epoch, best=best))
        notify('increment_gate', epoch, 30)
    gates.load_state_dict(best['model']); gate_best = best
    # Original D24 uses warped OOF priors, NOT in-sample fitted gate/consensus
    # outputs for training the intensity head. Keep that distinction explicit.
    bt = {c: warp(train_z['pred_'+c], pars[c]) for c in CS}
    bv = consensus(gated(valid_z, gates, mean, scale, pars, device), valid_fold_mean)
    blocks = [features(train_z, bt, c) for c in CS]
    xx = np.concatenate([v[0] for v in blocks]); im = xx.mean(0); isc = xx.std(0).clip(.05, None)
    X = torch.tensor((xx-im)/isc, device=device)
    lo, hi, lg = [torch.tensor(np.concatenate([v[j] for v in blocks]), device=device) for j in (1, 2, 3)]
    Y = torch.tensor(np.tile(train_z['y'], 3), dtype=torch.float64, device=device)
    intensity = Intensity().to(device); selections = {}
    for name, epochs, lr in (('smooth_intensity', 40, .001), ('median_intensity', 20, .0002)):
        opt = torch.optim.AdamW(intensity.parameters(), lr=lr, weight_decay=.01)
        start, best = resume(intensity, opt, out/(name+'_last.pt'), device)
        for epoch in range(start, epochs+1):
            if epoch:
                intensity.train()
                for ix in torch.randperm(len(Y), device=device).split(128):
                    opt.zero_grad(set_to_none=True); delta = intensity(X[ix]); p = lo[ix]+(hi[ix]-lo[ix])*torch.sigmoid(lg[ix]+delta); yt = Y[ix]
                    if name == 'smooth_intensity':
                        pc, yc = p-p.mean(), yt-yt.mean(); corr = (pc*yc).sum()/((pc.square().sum()*yc.square().sum()).sqrt()+1e-8)
                        loss = nn.functional.smooth_l1_loss(p, yt, beta=.1)+.1*(1-corr)+.01*delta.square().mean()
                    else: loss = (p-yt).abs().mean()+.01*delta.square().mean()
                    assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(intensity.parameters(), 5); opt.step()
            score = rank_fn(intensify(valid_z, bv, intensity, im, isc, device), valid_z['y'])
            if best is None or score > best['rank']: best = dict(rank=score, epoch=epoch, model=snapshot(intensity))
            save(out/(name+'_last.pt'), dict(model=snapshot(intensity), opt=opt.state_dict(), rng=rng(), epoch=epoch, best=best))
            notify(name, epoch, epochs)
        intensity.load_state_dict(best['model']); selections[name] = dict(epoch=best['epoch'], rank=best['rank'])
    state = dict(warp=pars, gate_model=gate_best['model'], gate_mean=mean, gate_scale=scale,
                 intensity_model=snapshot(intensity), intensity_mean=im, intensity_scale=isc,
                 selection=dict(increment_gate=dict(epoch=gate_best['epoch'], rank=gate_best['rank']), **selections), contract=contract)
    save(out/'complete.pt', state)
    return state

@torch.no_grad()
def apply_postprocessors(z, fold_mean, state, device='cuda'):
    """Label-free inference through D23; root applies frozen D24 factor anchor."""
    gates = nn.ModuleDict({c: Gate() for c in ('TA', 'TV')}).to(device); gates.load_state_dict(state['gate_model'])
    intensity = Intensity().to(device); intensity.load_state_dict(state['intensity_model'])
    base = consensus(gated(z, gates, state['gate_mean'], state['gate_scale'], state['warp'], device), fold_mean)
    return intensify(z, base, intensity, state['intensity_mean'], state['intensity_scale'], device)

def selftest():
    edges = np.array([-3, -2.5, -1.5, -.5, 0, .5, 1.5, 2.5, 3.])
    p = np.unique(np.concatenate((np.linspace(-3, 3, 6001), edges, np.nextafter(edges, np.inf).clip(-3, 3), np.nextafter(edges, -np.inf).clip(-3, 3))))
    for a, b in ((-.9, -2), (.9, 2), (0, 0)):
        q = warp(p, [[a, b]]*9); assert np.array_equal(signature(p), signature(q)); assert (np.diff(q) >= 0).all()
        _, lo, hi = cells(p); u = np.clip((p-lo)/np.maximum(hi-lo, 1e-30), 1e-7, 1-1e-7)
        for delta in (-2., 2.):
            q = correct(p, lo, hi, np.log(u/(1-u)), delta); assert np.array_equal(signature(p), signature(q))
    # Inference must not consume labels, and projection must preserve signatures
    n = len(p); z = {**{'pred_'+c: p for c in CS}, **{'f_'+c: np.zeros((n, 37), dtype='float32') for c in CS}}
    g = nn.ModuleDict({c: Gate() for c in ('TA', 'TV')}); it = Intensity()
    st = dict(warp={c: [[0, 0]]*9 for c in CS}, gate_model=snapshot(g), gate_mean={c: np.zeros(37, dtype='float32') for c in ('TA', 'TV')}, gate_scale={c: np.ones(37, dtype='float32') for c in ('TA', 'TV')}, intensity_model=snapshot(it), intensity_mean=np.zeros(52, dtype='float32'), intensity_scale=np.ones(52, dtype='float32'))
    ps = apply_postprocessors(z, {c: -p for c in CS}, st, 'cpu')
    assert all(np.array_equal(signature(p), signature(ps[c])) for c in CS)
    print('PASS: exact boundaries, monotonic warp, intensity and consensus class preservation, label-free inference')

if __name__ == '__main__': selftest()
