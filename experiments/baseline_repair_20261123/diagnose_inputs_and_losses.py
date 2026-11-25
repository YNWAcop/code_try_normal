"""Train/valid numerical integrity and batch-loss audit, no test evaluation."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'whole_priority_20261117/vendor'))
from mmsa_mapped_pickle import load_fields
from mmsa.trains.singleTask.MISA import CMD,DiffLoss
torch.set_num_threads(2)
result={'splits':{},'test_evaluated':False}
for ds in ('mosi','mosei'):
    blob=load_fields(Path('D:/MSA_Cloud_20261115/data/mmsa')/ds/'unaligned_50.pkl',
        ['audio','vision','audio_lengths','vision_lengths','regression_labels'],splits=('train','valid'))
    for split,data in blob.items():
        row={}
        for modality in ('audio','vision'):
            x=data[modality];counts=dict(nan=0,posinf=0,neginf=0)
            maximum=0.
            for i in range(0,len(x),64):
                a=x[i:i+64]
                counts['nan']+=int(np.isnan(a).sum());counts['posinf']+=int(np.isposinf(a).sum());counts['neginf']+=int(np.isneginf(a).sum())
                finite=a[np.isfinite(a)]
                if finite.size:maximum=max(maximum,float(np.abs(finite).max()))
            lengths=np.asarray(data[modality+'_lengths'])
            row[modality]={'shape':x.shape,**counts,'max_abs_finite':maximum,'min_length':int(lengths.min()),'max_length':int(lengths.max())}
        y=np.asarray(data['regression_labels']).reshape(-1)
        row['labels']={'n':len(y),'positive_nonzero_fraction':float(np.mean(y[y!=0]>0)),'mean':float(y.mean()),'median':float(np.median(y))}
        result['splits'][ds+'_'+split]=row
    del blob
# Actual saved MMSA loss classes. Microbatch averaging is not the original objective.
x=torch.randn(64,16,requires_grad=True);y=torch.zeros(64,16,requires_grad=True)
full=CMD()(x,y,5);micro=torch.stack([CMD()(a,b,5) for a,b in zip(x.split(4),y.split(4))]).mean()
gf=torch.autograd.grad(full,x,retain_graph=True)[0];gm=torch.autograd.grad(micro,x,retain_graph=True)[0]
df=DiffLoss()(x,y+torch.randn_like(y));other=torch.randn_like(y)
diff_full=DiffLoss()(x,other);diff_micro=torch.stack([DiffLoss()(a,b) for a,b in zip(x.split(4),other.split(4))]).mean()
result['misa_batch_objective']={'native_batch':64,'historical_microbatch':4,'CMD_full':float(full),'CMD_micro_mean':float(micro),
    'CMD_gradient_max_difference':float((gf-gm).abs().max()),'DiffLoss_full':float(diff_full),'DiffLoss_micro_mean':float(diff_micro),
    'conclusion':'CMD and orthogonality statistics must be computed on the native physical batch; accumulation alone is not equivalent.',
    'normalization_correction':'This actual MMSA MISA model uses LayerNorm, not BatchNorm. Earlier generic BatchNorm attribution was incorrect.'}
(ROOT/'input_and_batch_loss_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
