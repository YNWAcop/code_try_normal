"""Read-only checks: never start a training run or calculate test metrics."""
import json
import hashlib
from pathlib import Path
import zipfile
import numpy as np
import torch
from .settings import PROJECT,DATA_ROOT


def check():
    result={'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,
            'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'data_root':str(DATA_ROOT),'files':{},'code_verified':True}
    manifest=json.loads((PROJECT/'release/runtime_manifest.json').read_text())
    for n,h in manifest['files'].items():
        if hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Code changed: '+n)
    from .text_present.data import KEYS,LABEL_KEYS
    for ds in ('mosi','mosei'):
        for split in ('train','valid','test'):
            path=DATA_ROOT/'ready'/ds/(split+'.npz')
            if not path.is_file():raise FileNotFoundError(path)
            # ZIP member names only: no test labels or numerical features read.
            with zipfile.ZipFile(path) as z:
                present={Path(n).stem for n in z.namelist()}
                missing=set(KEYS+LABEL_KEYS)-present
                if missing:raise ValueError((str(path),sorted(missing)))
            result['files'][ds+'/'+split]={'bytes':path.stat().st_size,'schema':'complete'}
    return result
