"""Native EMT-DLFR/TFR-Net, one condition, validation-selected, no test evaluation.

Complete training views are privileged reconstruction/distillation targets.
Validation never supplies withheld features to either model branch.
"""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import sys
import time

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['EMT_DLFR', 'TFR_NET'], required=True)
    p.add_argument('--dataset', choices=['mosi', 'mosei'], required=True)
    p.add_argument('--protocol', choices=['whole', 'frame'], required=True)
    p.add_argument('--condition', default='TAV')
    p.add_argument('--rate', type=float, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--activation-checkpointing', action='store_true')
    p.add_argument('--data-root', type=Path, default=Path('D:/MSA_Cloud_20261115'))
    p.add_argument('--evaluate-only', action='store_true')
    cli = p.parse_args()
    cli.condition = ''.join(m for m in 'TAV' if m in cli.condition.upper())
    if cli.protocol == 'whole' and cli.condition not in ('A','V','T','AV','TA','TV'):
        p.error('whole condition must be A,V,T,AV,TA,TV (AT/VT aliases accepted)')
    if cli.protocol == 'frame' and not any(abs(cli.rate-r/10)<1e-8 for r in range(1,8)):
        p.error('frame rate must be .1 through .7')
    if cli.batch_size is not None and cli.batch_size < 2:
        p.error('batch-size must be >=2 because native projection/fusion contains BatchNorm')
    out = cli.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'validation_lock.json').exists() and not cli.evaluate_only:
        raise RuntimeError('Output already locked. Use a new directory; never overwrite a completed run.')
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(out/'train.log', encoding='utf-8')], format='%(asctime)s %(message)s')
    os.environ['EMT_BERT_PATH'] = str(cli.data_root/'models/bert-base-uncased')
    os.environ['HF_HUB_OFFLINE'] = '1'
    from emt_vendor_prepare import prepare
    vendor = prepare()
    sys.path.insert(0, str(vendor))
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset
    from config.config_regression import ConfigRegression
    from models.AMIO import AMIO
    from trains.ATIO import ATIO
    from mmsa_corruption import CorruptedDataset
    from mmsa_slim_pickle import load_fields
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this interpreter; choose the verified CUDA environment.')
    if cli.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    start = time.time()
    args = ConfigRegression(argparse.Namespace(modelName={'EMT_DLFR':'emt-dlfr','TFR_NET':'tfr_net'}[cli.method], datasetName=cli.dataset, train_mode='regression', fusion_method='emt', seed='unseeded', seeds=['unseeded'], num_seeds=1, cur_time=0, use_normalized_data=False)).get_config()
    args.smoke = cli.smoke
    args.device = torch.device(cli.device)
    args.protocol_condition = cli.condition if cli.protocol == 'whole' else 'TAV'
    args.model_save_path = str(out/'best.pt')
    original_batch, original_update = args.batch_size, args.update_epochs
    args.batch_size = cli.batch_size or 2
    if original_batch % args.batch_size:
        p.error('batch-size must divide the original native batch size')
    args.update_epochs = max(1, math.ceil(original_batch*original_update/args.batch_size))
    args.gradient_scale = args.batch_size / original_batch
    args.pop('missing_seed', None)
    if cli.smoke:
        args.batch_size, args.update_epochs = cli.batch_size or 2, 1
        args.gradient_scale = 1.0
    # The native EMT default 'Loss' includes reconstruction on withheld targets.
    # Select the deployed incomplete-view validation MAE instead.
    args.KeyEval = 'Loss(pred_m)' if cli.method == 'EMT_DLFR' else 'Loss'
    data_path = cli.data_root/'data/mmsa'/cli.dataset/'unaligned_50.pkl'
    fields = ('text_bert','audio','vision','audio_lengths','vision_lengths','id','regression_labels')
    blob = load_fields(data_path, fields, splits=('train','valid'), audit_test_ids=True)
    test_ids_only = blob.pop('_audit_test_ids')
    split_ids = {s:set(map(str,d['id'])) for s,d in blob.items()}
    split_ids['test'] = set(map(str,test_ids_only))
    overlaps = {a+'_'+b:len(split_ids[a]&split_ids[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
    if any(overlaps.values()):
        raise RuntimeError('Split sample IDs overlap: '+str(overlaps))
    if not cli.evaluate_only: dump(out/'split_audit.json', {'counts':{s:len(v) for s,v in split_ids.items()}, 'pairwise_overlap':overlaps, 'test_usage':'ID metadata audit only; no prediction, label selection, or evaluation'})
    del test_ids_only

    class Base(Dataset):
        def __init__(self, split):
            d = blob[split]
            self.text = np.asarray(d['text_bert'], dtype=np.float32)
            self.audio = d['audio']
            self.vision = d['vision']
            self.ids = d['id']
            self.labels = np.asarray(d['regression_labels'], dtype=np.float32).reshape(-1,1)
            self.audio_lengths = np.clip(np.asarray(d['audio_lengths'], dtype=np.int64), 1, self.audio.shape[1])
            self.vision_lengths = np.clip(np.asarray(d['vision_lengths'], dtype=np.int64), 1, self.vision.shape[1])
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            return {'text':torch.from_numpy(self.text[i].copy()), 'audio':torch.from_numpy(np.nan_to_num(np.asarray(self.audio[i],dtype=np.float32), nan=0,posinf=0,neginf=0)), 'vision':torch.from_numpy(np.nan_to_num(np.asarray(self.vision[i],dtype=np.float32), nan=0,posinf=0,neginf=0)), 'audio_lengths':int(self.audio_lengths[i]), 'vision_lengths':int(self.vision_lengths[i]), 'labels':{'M':torch.from_numpy(self.labels[i].copy())}, 'id':str(self.ids[i]), 'index':i}

    class NativeMissing(Dataset):
        def __init__(self, split):
            self.split = split
            self.base = Base(split)
            self.corrupt = CorruptedDataset(self.base, HERE, cli.dataset, split, cli.protocol, cli.condition, cli.rate)
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            raw = self.base[i]
            visible = self.corrupt[i]
            # Only training is allowed original privileged reconstruction targets.
            item = raw if self.split == 'train' else dict(visible)
            for m,key in [('T','text'),('A','audio'),('V','vision')]:
                item[key+'_m'] = visible[key]
                if m == 'T':
                    available = visible[key][1].clone()
                    target_mask = item[key][1].clone()
                else:
                    length = item[key+'_lengths']
                    target_mask = (torch.arange(item[key].shape[0]) < length).float()
                    available = (torch.arange(item[key].shape[0]) < visible[key+'_lengths']).float()
                    item[key+'_mask'] = target_mask
                if cli.protocol == 'whole' and m not in cli.condition:
                    available.zero_()
                elif cli.protocol == 'frame':
                    dropped = torch.from_numpy(self.corrupt.masks[m][i].copy())
                    if m == 'T':
                        dropped &= (raw[key][1]>0) & (raw[key][0]!=101) & (raw[key][0]!=102)
                    available[dropped] = 0
                item[key+'_missing_mask'] = available
            return item

    train, valid = NativeMissing('train'), NativeMissing('valid')
    # No test dataset/labels are referenced. Release unrelated split objects.
    blob = None
    args.seq_lens = (train.base.text.shape[2], train.base.audio.shape[1], train.base.vision.shape[1])
    args.feature_dims = (768, train.base.audio.shape[2], train.base.vision.shape[2])
    data_hash = hashlib.sha256(('\n'.join(map(str,train.base.ids))+'\nVALID\n'+'\n'.join(map(str,valid.base.ids))).encode()).hexdigest()
    if set(map(str,train.base.ids)) & set(map(str,valid.base.ids)):
        raise RuntimeError('Train/validation IDs overlap.')
    info = {'cli':vars(cli), 'native_config':dict(args), 'original_batch_size':original_batch, 'original_update_epochs':original_update, 'memory_override':args.batch_size != original_batch, 'training_seed':None, 'split_id_sha256':data_hash, 'train_count':len(train), 'valid_count':len(valid), 'mask_paths':{s:str(getattr(d.corrupt,'mask_path','whole_fixed')) for s,d in [('train',train),('valid',valid)]}, 'validation_targets':'only sentiment labels; withheld feature inputs replaced with corrupted copies', 'protocol':'Whole T empty CLS/SEP; A/V zero with public length1. Training privileged complete targets for native losses.', 'checkpoint_selection':args.KeyEval, 'native_patch_manifest':str(vendor/'PATCH_MANIFEST.json')}
    if not cli.evaluate_only: dump(out/'config.json',info)
    loaders = {s:DataLoader(Subset(d,range(min(args.batch_size*2,len(d)))) if cli.smoke else d, batch_size=args.batch_size, shuffle=s=='train', num_workers=0, drop_last=s=='train', pin_memory=cli.device=='cuda') for s,d in [('train',train),('valid',valid)]}
    model = AMIO(args).to(args.device)
    if cli.activation_checkpointing:
        if cli.method != 'EMT_DLFR': raise ValueError('Checkpointing verified for EMT only')
        model.Model.activation_checkpointing = True
        model.Model.text_model.model.gradient_checkpointing_enable()
    from full_four import evaluate_selected
    if cli.evaluate_only:
        evaluate_selected(model, loaders['valid'], cli, args)
        return
    trainer = ATIO().getTrain(args)
    original_validate = trainer.do_test
    history = []
    def validate(*a, **kw):
        result = original_validate(*a, **kw)
        score = float(result[args.KeyEval])
        if not np.isfinite(score):
            raise RuntimeError('Non-finite validation score; no checkpoint may be locked.')
        history.append({'epoch':len(history)+1, 'validation':{k:float(v) for k,v in result.items()}, 'elapsed_seconds':time.time()-start})
        dump(out/'history.json',history)
        return result
    trainer.do_test = validate
    trainer.do_train(model,loaders)
    if not history or not (out/'best.pt').is_file():
        raise RuntimeError('Training did not save a validation-selected checkpoint.')
    # Match native tolerance-based checkpoint selection, not arbitrary rounded minima.
    best_score = float('inf'); best_epoch = None
    for row in history:
        score = row['validation'][args.KeyEval]
        if score <= best_score-1e-6:
            best_score,best_epoch=score,row['epoch']
    checkpoint = out/'best.pt'
    with checkpoint.open('rb') as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    lock={'status':'validation_locked_no_test_evaluated','smoke':cli.smoke,'method':cli.method,'dataset':cli.dataset,'protocol':cli.protocol,'condition':cli.condition,'rate':cli.rate,'selected_epoch':best_epoch,'valid_score':best_score,'checkpoint':str(checkpoint),'checkpoint_sha256':h.hexdigest(),'epochs_completed':len(history),'training_seed':None,'test_evaluated':False,'elapsed_seconds':time.time()-start, 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if cli.device=='cuda' else None}
    dump(out/'validation_lock.json',lock)
    print(json.dumps(lock,ensure_ascii=False,indent=2))

    evaluate_selected(model, loaders['valid'], cli, args)

if __name__ == '__main__':
    main()
