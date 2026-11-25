"""Synthetic full-length resource probe. Never a model-selection experiment."""
import argparse, json, os, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/emt_dlfr'))
os.environ['EMT_BERT_PATH']='D:/MSA_Cloud_20261115/models/bert-base-uncased'
os.environ['HF_HUB_OFFLINE']='1'
from config.config_regression import ConfigRegression
from models.AMIO import AMIO
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--activation-checkpointing',action='store_true')
c=p.parse_args();torch.set_num_threads(4)
args=ConfigRegression(argparse.Namespace(modelName='emt-dlfr',datasetName=c.dataset,train_mode='regression',fusion_method='emt',seed='unseeded',seeds=['unseeded'],num_seeds=1,cur_time=0,use_normalized_data=False)).get_config()
args.protocol_condition='TA';args.device=torch.device('cuda')
start=time.time();result={'dataset':c.dataset,'batch':c.batch,'purpose':'memory only, synthetic inputs, no selection','native_batch':args.batch_size}
try:
    model=AMIO(args).cuda().train();opt=torch.optim.Adam(model.parameters(),lr=1e-5)
    if c.activation_checkpointing:
        model.Model.activation_checkpointing=True
        model.Model.text_model.model.gradient_checkpointing_enable()
    b=c.batch;text=torch.zeros(b,3,50,device='cuda');text[:,0]=101;text[:,1]=1
    audio=torch.randn(b,500,args.feature_dims[1],device='cuda');video=torch.randn(b,500,args.feature_dims[2],device='cuda')
    lengths=torch.full((b,),500,device='cuda',dtype=torch.long)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out=model((text,text),(audio,audio,lengths),(video,torch.zeros_like(video),lengths))
        loss=sum(v.square().mean() for k,v in out.items() if not k.startswith('z_') and k!='text_for_recon')
        loss.backward();opt.step()
    torch.cuda.synchronize()
    result.update(passed=True,peak_gib=torch.cuda.max_memory_allocated()/2**30,elapsed=time.time()-start)
except torch.cuda.OutOfMemoryError as e:
    result.update(passed=False,error='CUDA OOM',elapsed=time.time()-start)
result['activation_checkpointing']=c.activation_checkpointing
(ROOT/f'memory_probe_{c.dataset}_{c.batch}_{int(c.activation_checkpointing)}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
