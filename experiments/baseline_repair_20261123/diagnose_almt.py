"""Replay the actual degenerate ALMT checkpoint on validation; inspect audio path."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from mmsa_mapped_pickle import load_fields
from metrics_common import digest
path=ROOT.parent/'whole_priority_20261117/adapter_misa_almt.py'
spec=importlib.util.spec_from_file_location('adapter',path);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
from mmsa.models.singleTask.ALMT import ALMT
run=ROOT.parent/'local_baselines_20261115/runs/results/ALMT_mosi_A'
args=adapter.Args(json.loads((run/'run_manifest.json').read_text(encoding='utf-8')))
args.device=torch.device('cpu');args.pretrained='D:/MSA_Cloud_20261115/models/bert-base-uncased'
torch.set_num_threads(2)
model=adapter.ModelWrapper(ALMT(args),args)
model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True,mmap=True),strict=True);model.eval()
blob=load_fields(Path('D:/MSA_Cloud_20261115/data/mmsa/mosi/unaligned_50.pkl'),
    ['text_bert','audio','vision','audio_lengths','vision_lengths','id','regression_labels'],splits=('valid',))
base=adapter.InputDataset(blob['valid']);dataset=adapter.CorruptedDataset(base,ROOT,'mosi','valid','whole','A',0.)
loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False)
pred=[];ids=[]
with torch.no_grad():
    for b in loader:
        pred.extend(model(b['text'],b['audio'],b['vision']).reshape(-1).tolist());ids.extend(b['id'])
archive=ROOT.parent/'existing_results_download_20261120_112150/reports/files/whole_priority_20261117/results/ALMT_mosi_A'
saved=np.load(archive/'predictions_full_valid.npz',allow_pickle=False)
archived_metrics=json.loads((archive/'metrics_full_valid.json').read_text(encoding='utf-8'))
assert digest(run/'best.pt')==archived_metrics['checkpoint_sha256']
assert ids==list(saved['ids'].astype(str))
delta=float(np.max(np.abs(np.array(pred)-saved['pred'].reshape(-1))))
assert delta<2e-5,delta
b=next(iter(loader));audio=b['audio'].clone().requires_grad_()
out=model(b['text'],audio,b['vision']);loss=(out.reshape(-1)-b['labels']['M'].reshape(-1)).square().mean();loss.backward()
grad=float(audio.grad.norm());assert np.isfinite(grad) and grad>0
parameter_grad=float(model.Model.embedding_a[0].weight.grad.norm());assert parameter_grad>0
with torch.no_grad():
    erased=model(b['text'],torch.zeros_like(audio),b['vision'])
effect=float((out-erased).abs().max());assert effect>0
result={'method':'ALMT','dataset':'mosi','condition':'A','checkpoint_sha256':digest(run/'best.pt'),
    'validation_replay_max_difference':delta,'validation_prediction_std':float(np.std(pred)),
    'validation_positive_fraction':float(np.mean(np.array(pred)>0)),
    'audio_input_gradient_norm':grad,'audio_embedding_gradient_norm':parameter_grad,'zero_audio_prediction_delta':effect,
    'conclusion':'Selected checkpoint is near-constant but the audio path is connected and differentiable; archived later training learns variation while validation MSE selected epoch2. No disconnected-input implementation bug demonstrated.',
    'test_evaluated':False,'fresh_training_performed':False}
(ROOT/'almt_mosi_A_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
