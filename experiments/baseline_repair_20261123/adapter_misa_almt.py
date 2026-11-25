"""Real MMSA MISA/ALMT, train/validation only; no seed sweep or test evaluation.

The protocol is an adaptation, not an exact reproduction of original full-input
paper tables. Original source/config hashes are in vendor/mmsa/source_manifest.json.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import argparse, hashlib, importlib, json, logging, math, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'vendor'))
from mmsa_corruption import CorruptedDataset
from mmsa_slim_pickle import load_fields

class Args(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class InputDataset(Dataset):
    def __init__(self, split):
        self.text = np.asarray(split['text_bert'], dtype=np.float32)
        self.audio = split['audio']
        self.vision = split['vision']
        self.ids = split['id']
        self.labels = np.asarray(split['regression_labels'], dtype=np.float32)
        self.audio_lengths = np.asarray(split['audio_lengths'], dtype=np.int64)
        self.vision_lengths = np.asarray(split['vision_lengths'], dtype=np.int64)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return {'text': torch.from_numpy(self.text[i].copy()),
                'audio': torch.from_numpy(np.nan_to_num(np.asarray(self.audio[i],dtype=np.float32), copy=True)),
                'vision': torch.from_numpy(np.nan_to_num(np.asarray(self.vision[i],dtype=np.float32), copy=True)), 'id': str(self.ids[i]),
                'audio_lengths': int(self.audio_lengths[i]), 'vision_lengths': int(self.vision_lengths[i]),
                'labels': {'M': torch.tensor(self.labels[i])}}

class ModelWrapper(torch.nn.Module):
    def __init__(self, actual, args):
        super().__init__()
        self.Model = actual
        self.args = args
    def forward(self, text, audio, vision):
        return self.Model(text, audio, vision)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=['MISA','ALMT'])
    p.add_argument('--dataset', required=True, choices=['mosi','mosei'])
    p.add_argument('--protocol', required=True, choices=['whole','frame'])
    p.add_argument('--condition', default='TAV')
    p.add_argument('--rate', type=float, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--data-root', type=Path, default=Path('D:/MSA_Cloud_20261115'))
    p.add_argument('--evaluate-only', action='store_true')
    cli = p.parse_args()
    condition = ''.join(m for m in 'TAV' if m in cli.condition.upper())
    if cli.protocol == 'whole' and condition not in ['A','V','T','AV','TA','TV']:
        p.error('whole condition must be A,V,T,AV,AT/TA,TV/VT')
    if cli.protocol == 'frame' and not any(abs(cli.rate-r/10)<1e-8 for r in range(1,8)):
        p.error('frame rate must be 0.1 through 0.7')
    if cli.batch_size is not None and cli.batch_size < 2: p.error('batch size must be >=2 for batch-distribution losses')
    out = cli.out.resolve()
    if ((out/'validation_lock.json').exists() or (out/'best.pt').exists()) and not cli.evaluate_only:
        raise RuntimeError(f'Refusing overwrite completed/checkpoint run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(out/'train.log', encoding='utf-8'),logging.StreamHandler()])
    start = time.time()
    if cli.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA not available')
    torch.set_num_threads(min(int(os.environ.get('PRIORITY_CPU_THREADS', '4')), os.cpu_count() or 1))
    cfg = json.loads((HERE/'vendor/mmsa/config/config_regression.json').read_text())
    method_cfg = cfg[cli.method.lower()]
    args = Args(cfg['datasetCommonParams'][cli.dataset]['unaligned'])
    args.update(method_cfg['commonParams'])
    args.update(method_cfg['datasetParams'][cli.dataset])
    original_batch = args.batch_size
    args.update(model_name=cli.method.lower(), dataset_name=cli.dataset, train_mode='regression',
                device=torch.device(cli.device), cur_seed='single_unseeded', smoke=cli.smoke,
                model_save_path=str(out/'best.pt'), pretrained=str(cli.data_root/'models/bert-base-uncased'))
    # Missing seed fields in upstream config are unused and removed to prevent confusion.
    args.pop('missing_seed', None)
    args.pop('missing_rate', None)
    args.batch_size = cli.batch_size or (2 if cli.smoke else original_batch)
    if cli.method == 'MISA' and not cli.smoke and args.batch_size != original_batch:
        p.error('MISA CMD/DiffLoss require the native physical batch; gradient accumulation is not equivalent')
    if original_batch % args.batch_size:
        p.error(f'batch size must divide original batch {original_batch}')
    original_update = args.get('update_epochs', 1)
    args.accumulation_steps = original_batch // args.batch_size
    args.gradient_scale = args.batch_size / original_batch
    args.update_epochs = original_update * args.accumulation_steps
    if cli.smoke:
        args.accumulation_steps = args.update_epochs = 1
        args.gradient_scale = 1.
    path = cli.data_root/'data/mmsa'/cli.dataset/'unaligned_50.pkl'
    logging.info('Reading train/valid from trusted local container %s', path)
    # Streaming reader skips large unused cached 768-dimensional text arrays.
    # Test data in the legacy container is discarded, never iterated/scored.
    fields = ['text_bert','audio','vision','id','regression_labels','audio_lengths','vision_lengths']
    container = load_fields(path,fields,splits=('train','valid'),audit_test_ids=True)
    id_sets = {s:set(map(str,container[s]['id'])) for s in ('train','valid')}
    id_sets['test'] = set(map(str,container['_audit_test_ids']))
    for left,right in [('train','valid'),('train','test'),('valid','test')]:
        if id_sets[left] & id_sets[right]:
            raise RuntimeError(f'{left}/{right} IDs overlap')
    id_audit = {'disjoint':True, 'split_counts':{k:len(v) for k,v in id_sets.items()},
                'test_usage':'IDs only for split audit; labels/features not iterated or scored'}
    del id_sets
    train_split, valid_split = container['train'], container['valid']
    del container
    base_train, base_valid = InputDataset(train_split), InputDataset(valid_split)
    del train_split, valid_split
    if set(map(str,base_train.ids)) & set(map(str,base_valid.ids)):
        raise RuntimeError('Train/validation IDs overlap')
    args.feature_dims = [768, base_train.audio.shape[-1],base_train.vision.shape[-1]]
    args.feature_length = [base_train.text.shape[-1],base_train.audio.shape[1],base_train.vision.shape[1]]
    loaders = {}
    masks = {}
    for name, base in [('train',base_train),('valid',base_valid)]:
        ds = CorruptedDataset(base, HERE, cli.dataset, name, cli.protocol, condition, cli.rate)
        masks[name] = getattr(ds, 'mask_path', None)
        if cli.smoke: ds = Subset(ds, range(min(max(4,args.batch_size*2),len(ds))))
        loaders[name] = DataLoader(ds, batch_size=args.batch_size, shuffle=name=='train',
                                  num_workers=0, drop_last=name=='train')
    model_cls = getattr(importlib.import_module(f'mmsa.models.singleTask.{cli.method}'), cli.method)
    trainer_cls = getattr(importlib.import_module(f'mmsa.trains.singleTask.{cli.method}'), cli.method)
    model = ModelWrapper(model_cls(args), args).to(args.device)
    if cli.gradient_checkpointing:
        model.Model.bertmodel.model.gradient_checkpointing_enable()
    history = []
    def on_valid(epoch,metrics):
        metrics = {k:float(v) for k,v in metrics.items()}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f'Non-finite validation metrics {metrics}')
        history.append({'epoch':epoch, 'metrics':metrics})
        (out/'validation_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    manifest = dict(args)
    trainer_file = HERE/'vendor/mmsa/trains/singleTask'/f'{cli.method}.py'
    manifest['trainer_source_sha256'] = hashlib.sha256(trainer_file.read_bytes()).hexdigest()
    if cli.method == 'MISA':
        manifest['cmd_numerics'] = 'vector_norm_zero_subgradient_v1; finite gradient check before clipping'
    manifest.update(device=str(args.device), method=cli.method, protocol=cli.protocol,
                    condition=condition, rate=cli.rate, original_batch_size=original_batch,
                    gradient_checkpointing=cli.gradient_checkpointing, input_path=str(path),
                    nominal_effective_batch=original_batch*original_update,
                    numerical_adaptation='Native physical batch required for MISA CMD and DiffLoss statistics; MISA uses LayerNorm.',
                    masks=masks, split_id_audit=id_audit, train_samples=len(loaders['train'].dataset), valid_samples=len(loaders['valid'].dataset),
                    training_seed=None, test_evaluated=False, input_type='unaligned cached features + finetuned BERT',
                    source_manifest=str(HERE/'vendor/mmsa/source_manifest.json'))
    if not cli.evaluate_only: (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    args.validation_callback = on_valid
    from full_four import evaluate_selected
    if cli.evaluate_only:
        evaluate_selected(model, loaders['valid'], cli, args)
        return
    trainer = trainer_cls(args)
    trainer.do_train(model,loaders,return_epoch_results=False)
    if not (out/'best.pt').is_file(): raise RuntimeError('No best checkpoint produced')
    best = min(history,key=lambda x:x['metrics'][args.KeyEval])
    from metrics_common import digest
    result = {'status':'validation_locked_no_test_evaluated', 'smoke':cli.smoke,
              'method':cli.method,'dataset':cli.dataset,'protocol':cli.protocol,'condition':condition,'rate':cli.rate,
              'selected_epoch':best['epoch'], 'valid_score':best['metrics'][args.KeyEval],
              'validation_metrics':best['metrics'],'checkpoint':str(out/'best.pt'),
              'epochs_completed':len(history),'elapsed_seconds':time.time()-start,
              'checkpoint_bytes':(out/'best.pt').stat().st_size,'test_evaluated':False,
              'checkpoint_sha256':digest(out/'best.pt'),
              'peak_cuda_bytes':torch.cuda.max_memory_allocated() if cli.device=='cuda' else None}
    (out/'validation_lock.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    logging.info('COMPLETED %s',json.dumps(result))
    evaluate_selected(model, loaders['valid'], cli, args)

if __name__ == '__main__': main()
