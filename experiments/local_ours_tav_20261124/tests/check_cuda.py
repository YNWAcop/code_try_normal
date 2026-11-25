"""One synthetic optimization step with actual feature widths; no real data training."""
import os,sys,json
from pathlib import Path
os.environ.update(MSA_DEVICE='cuda',MSA_SMOKE='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from evidence_msa.text_present.protocol import configure,CONFIG,CS
from evidence_msa.text_present.model import TextPresent,objective
from evidence_msa.text_present.raw_adapters import RawEvidenceAdapter
configure();torch.cuda.reset_peak_memory_stats()
dims=dict(T=1024,A=1280,V=3352,app=1048,timeline=768,Q=5)
b={k:torch.randn(64,n,device='cuda') for k,n in [('T',1024),('A',1280),('AS',1280),('V',3352),('VS',3352),('Q',5)]};b['timeline']=torch.randn(64,16,768,device='cuda');y=torch.linspace(-3,3,64,device='cuda')
m=TextPresent(dims,CONFIG,128).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003)
for c in CS:loss=objective(m(b,c),y,CONFIG)/len(CS);assert torch.isfinite(loss);loss.backward()
torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
ck=dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},dims=dims,config=CONFIG,hidden=128)
adapter=RawEvidenceAdapter(ck,'D5_bilinear').cuda();adapter.train();opt=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=.0001)
b={k:v[:32] for k,v in b.items()}
for c in CS:loss=objective(adapter(b,c),y[:32],CONFIG)/len(CS);assert torch.isfinite(loss);loss.backward()
opt.step();torch.cuda.synchronize()
print(json.dumps(dict(passed=True,synthetic_only=True,gpu=torch.cuda.get_device_name(0),max_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,max_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,scope='one C5 step batch64 + one frozen-core ERU adapter step batch32, real feature widths; not full training peak'),indent=2))
