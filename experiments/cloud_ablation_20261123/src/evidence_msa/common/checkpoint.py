"""Utilities for the isolated, validation-only V4 experiments."""
import contextlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .metrics import atomic_json, digest

HERE = Path(__file__).resolve().parent
CONDITIONS = ('T', 'TA', 'TV')
PROTECTED = ('A', 'V', 'AV')
PRIMARY = ('Acc7', 'Acc5', 'Acc2_nonzero', 'F1_weighted_nonzero',
           'Acc2_haszero', 'F1_weighted_haszero', 'MAE', 'Corr')
CONFIGS = {
    'C0_regression': dict(distribution=False, polarity=False, shared=False, attention=False),
    'C1_distribution': dict(distribution=True, polarity=False, shared=False, attention=False),
    'C2_polarity': dict(distribution=True, polarity=True, shared=False, attention=False),
    'C3_dual': dict(distribution=True, polarity=True, shared=True, attention=False),
    'C4_interaction': dict(distribution=True, polarity=True, shared=False, attention=True),
    'C5_combined': dict(distribution=True, polarity=True, shared=True, attention=True),
}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def dump(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(p, value)

def loadck(p):
    # Local trusted experiment checkpoints only. Compatible with Torch 1.12+.
    try:
        return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(p, map_location='cpu')

def tsave(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.partial'); torch.save(value, tmp); os.replace(tmp, p)

def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch'])
    if r['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(r['cuda'])

def event(out, stage, **kw):
    value = dict(time=time.time(), pid=os.getpid(), stage=stage, **kw)
    dump(Path(out) / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)

def code_hashes():
    return {p.name: digest(p) for p in sorted(HERE.glob('*.py'))}

@contextlib.contextmanager
def lock(path):
    """Kernel-released advisory lock; no pidfd_open or PID namespace assumptions."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    try:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        yield
    finally:
        f.close()

def selection_rank(current, baseline):
    """Fixed minimax improvement, then mean improvement, over 24 validation cells.

    Scales are predeclared comparison units, NOT standard deviations/significance.
    Higher is better after sign reversal for MAE. One checkpoint for all metrics.
    """
    gains = []
    for c in CONDITIONS:
        for k in PRIMARY:
            v, b = current[c][k], baseline[c][k]
            if v is None or b is None or not np.isfinite([v, b]).all():
                raise ValueError('Undefined primary validation metric: %s/%s' % (c, k))
            scale = .05 if k in ('MAE', 'Corr') else .02
            gains.append((b-v if k == 'MAE' else v-b) / scale)
    return (float(min(gains)), float(np.mean(gains)))

def dominates(a, b):
    av = np.array([(-a[c][k] if k == 'MAE' else a[c][k]) for c in CONDITIONS for k in PRIMARY])
    bv = np.array([(-b[c][k] if k == 'MAE' else b[c][k]) for c in CONDITIONS for k in PRIMARY])
    return bool(np.all(av >= bv) and np.any(av > bv))
