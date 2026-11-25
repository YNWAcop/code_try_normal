"""Native batch resource/mechanism probe, synthetic only, no model selection."""
import argparse,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from adapter_misa_almt import Args,ModelWrapper
from mmsa.models.singleTask.MISA import MISA as Net
from mmsa.trains.singleTask.MISA import MISA as Trainer
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);c=p.parse_args()
torch.set_num_threads(4)
cfg=json.loads((ROOT/'vendor/mmsa/config/config_regression.json').read_text())
a=Args(cfg['datasetCommonParams'][c.dataset]['unaligned']);a.update(cfg['misa']['commonParams']);a.update(cfg['misa']['datasetParams'][c.dataset])
a.update(train_mode='regression',dataset_name=c.dataset,device=torch.device('cuda'),pretrained='D:/MSA_Cloud_20261115/models/bert-base-uncased')
result={'dataset':c.dataset,'batch':a.batch_size,'synthetic_only':True,'gradient_checkpointing':True};start=time.time()
try:
    model=ModelWrapper(Net(a),a).cuda().train();model.Model.bertmodel.model.gradient_checkpointing_enable()
    trainer=Trainer(a);trainer.model=model;opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
    b=a.batch_size;t=torch.zeros(b,3,50,device='cuda');t[:,0]=103;t[:,0,0]=101;t[:,0,-1]=102;t[:,1]=1
    x=torch.randn(b,500,a.feature_dims[1],device='cuda');v=torch.randn(b,500,a.feature_dims[2],device='cuda');y=torch.linspace(-2,2,b,device='cuda').view(-1,1)
    model.Model.visible_audio_lengths=torch.full((b,),500,dtype=torch.long);model.Model.visible_vision_lengths=torch.full((b,),500,dtype=torch.long)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);pred=model(t,x,v)['M']
        loss=(pred-y).square().mean()+a.diff_weight*trainer.get_diff_loss()+a.sim_weight*trainer.get_cmd_loss()+a.recon_weight*trainer.get_recon_loss()
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    torch.cuda.synchronize();result.update(passed=True,peak_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.time()-start)
except torch.cuda.OutOfMemoryError:
    result.update(passed=False,error='CUDA OOM',seconds=time.time()-start)
(ROOT/f'misa_memory_{c.dataset}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
