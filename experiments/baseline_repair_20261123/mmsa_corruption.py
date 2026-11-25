"""Persist once, reuse missingness across compatible MMSA inputs. No fixed seed."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset


class CorruptedDataset(Dataset):
    def __init__(self, dataset, root, dataset_name, split, protocol, condition, rate, lnln=False):
        self.base = dataset
        self.protocol, self.condition, self.rate, self.lnln = protocol, condition, rate, lnln
        self.masks = {}
        if protocol == 'frame':
            ids = np.asarray([str(x) for x in dataset.ids])
            shapes = {m: np.asarray(getattr(dataset, k)).shape for m,k in [('T','text'),('A','audio'),('V','vision')]}
            signature = hashlib.sha256((repr(shapes)+'\n'+'\n'.join(ids)).encode()).hexdigest()[:16]
            folder = Path(root)/'baseline_protocol/masks'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{dataset_name}_{split}_frame{rate:.1f}_{signature}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    assert np.array_equal(ids,z['ids'])
                    self.masks = {m:z[m] for m in 'TAV'}
            else:
                rng = np.random.default_rng()
                self.masks = {m:rng.random((len(dataset), s[2] if m=='T' else s[1])) < rate for m,s in shapes.items()}
                np.savez_compressed(path, ids=ids, **self.masks)
            self.mask_path = str(path)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, i):
        item = self.base[i]
        item = dict(item)
        item['labels'] = dict(item['labels'])
        for m, key in [('T','text'),('A','audio'),('V','vision')]:
            x = item[key].clone()
            fraction = 0.
            if self.protocol == 'whole' and m not in self.condition:
                x.zero_()
                if m == 'T':
                    # Constant empty sentence; no original words/length are revealed.
                    x[0,0],x[0,1],x[1,0],x[1,1] = 101,102,1,1
                else:
                    lk = key+'_lengths'
                    if lk in item:
                        item[lk] = torch.ones_like(item[lk]) if torch.is_tensor(item[lk]) else 1
                fraction = 1.
            elif self.protocol == 'frame':
                mask = torch.as_tensor(self.masks[m][i], dtype=torch.bool)
                if m == 'T':
                    eligible = (x[1] > 0) & (x[0] != 101) & (x[0] != 102)
                    mask = mask & eligible
                    x[0,mask] = 100  # BERT UNK; preserve original padding and CLS/SEP
                    fraction = float(mask.sum()) / max(1,int(eligible.sum()))
                else:
                    x[mask] = 0
                    lk = key+'_lengths'
                    length = int(item[lk]) if lk in item else x.shape[0]
                    fraction = float(mask[:length].sum()) / max(1,length)
            if self.lnln:
                item[key+'_m'] = x
                labelkey = 'missing_rate_'+{'T':'l','A':'a','V':'v'}[m]
                old = item['labels'].get(labelkey)
                item['labels'][labelkey] = torch.full_like(old, fraction) if torch.is_tensor(old) else fraction
            else:
                item[key] = x
        return item


def wrap_loader_dataset(dataset, root, dataset_name, split, protocol, condition, rate, lnln=False):
    return CorruptedDataset(dataset, root, dataset_name, split, protocol, condition, rate, lnln)
