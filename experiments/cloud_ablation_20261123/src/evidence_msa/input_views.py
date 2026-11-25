"""Remove enhanced information before fitting transforms, including R3 inputs."""
import numpy as np
from .settings import INPUT_MODE
ENHANCED=('A_words','V_lip','V_words','V_expression','V_quality','V_timeline')
def select_inputs(data,mode=None):
    mode=INPUT_MODE if mode is None else mode
    assert mode in ('native','enhanced')
    out=dict(data)
    if mode=='native':
        for key in ENHANCED:out[key]=np.zeros_like(data[key])
    return out
