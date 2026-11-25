"""Synthetic fixture only: no benchmark score can be reported from this data."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
for ds in ('mosi','mosei'):
    out=a.destination/'ready'/ds;out.mkdir(parents=True,exist_ok=True)
    for split,n in [('train',24),('valid',12),('test',12)]:
        d={k:np.random.normal(size=(n,dim)).astype('float32') for k,dim in [('T',8),('A_native',5),('A_words',4),('V_native',6),('V_expression',3),('V_lip',4),('V_words',4),('V_quality',2)]}
        d['V_timeline']=np.random.normal(size=(n,4,3)).astype('float32');d['y']=np.tile(np.array([-1.5,.5,1.5],dtype='float32'),n//3)
        d['ids']=np.array([f'{ds}_{split}_{i}' for i in range(n)]);d['groups']=np.array([f'{ds}_{split}_video_{i//3}' for i in range(n)])
        np.savez_compressed(out/(split+'.npz'),**d)
print(a.destination)
