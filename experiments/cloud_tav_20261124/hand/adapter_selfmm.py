"""Self-MM fixed whole-modality/frame-corruption adaptation, validation-only.
Original architecture and auxiliary updates retained. No upstream seed loop.
"""
import argparse, ast, hashlib, json, logging, os, shutil, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=['mosi','mosei'],required=True);p.add_argument('--condition',default='TAV');p.add_argument('--protocol',choices=['whole','frame'],default='whole');p.add_argument('--rate',type=float,default=.3);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.condition=''.join(m for m in 'TAV' if m in a.condition.upper())
    if not a.condition or not 0<=a.rate<1:raise ValueError('Invalid condition or rate')
    root=a.root.resolve();src=root/'baselines/upstream/Self_MM'
    tag=a.condition if a.protocol=='whole' else f'frame{a.rate:.1f}'
    out=root/'baseline_results'/('Self_MM_'+a.dataset+'_'+tag+('_smoke' if a.smoke else ''));out.mkdir(parents=True,exist_ok=False)
    dst=root/'baselines/adapted'/out.name;shutil.copytree(src,dst)
    bert=dst/'models/subNets/BertTextEncoder.py';s=bert.read_text(encoding='utf-8')
    assert "'pretrained_model/bert_en'" in s
    s=s.replace("'pretrained_model/bert_en'",repr(str(root/'models/bert-base-uncased')));bert.write_text(s,encoding='utf-8')
    model_file=dst/'models/multiTask/SELF_MM.py';s=model_file.read_text(encoding='utf-8')
    assert 'pack_padded_sequence(x, lengths,' in s
    # Modern PyTorch requires CPU int64 sequence lengths even with CUDA inputs.
    s=s.replace('pack_padded_sequence(x, lengths,','pack_padded_sequence(x, lengths.detach().cpu().long(),')
    s=s.replace('final_states[0].squeeze()', 'final_states[0].squeeze(0)')
    model_file.write_text(s,encoding='utf-8')
    loader=dst/'data/load_data.py';s=loader.read_text(encoding='utf-8')
    s=s.replace("        'valid': MMDataset(args, mode='valid'),\n        'test': MMDataset(args, mode='test')","        'valid': MMDataset(args, mode='valid')")
    assert "'test': MMDataset" not in s
    loader.write_text(s,encoding='utf-8')
    trainerfile=dst/'trains/multiTask/SELF_MM.py';s=trainerfile.read_text(encoding='utf-8')
    # Instrument author trainer, preserving optimizer/self-label updates/early stopping.
    target="            cur_valid = val_results[self.args.KeyEval]"
    assert target in s
    s=s.replace(target,target+"\n            self.validation_history.append({'epoch': epochs, 'valid': val_results})")
    target='                best_valid, best_epoch = cur_valid, epochs';assert target in s
    s=s.replace(target,target+'\n                self.selected_epoch = epochs\n                self.selected_valid = val_results')
    target='        saved_labels = {}';assert target in s
    s=s.replace(target,target+'\n        self.validation_history = []')
    if a.smoke:s=s.replace('if epochs - best_epoch >= self.args.early_stop:', 'if epochs >= 1:')
    trainerfile.write_text(s,encoding='utf-8');ast.parse(s)
    os.chdir(dst);sys.path.insert(0,str(dst));logging.basicConfig(level=logging.INFO)
    import torch,numpy as np
    from torch.utils.data import DataLoader, Subset
    from config.config_regression import ConfigRegression
    from data.load_data import MMDataLoader
    from models.AMIO import AMIO
    from trains.ATIO import ATIO
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    args=ConfigRegression(argparse.Namespace(modelName='self_mm',datasetName=a.dataset,train_mode='regression',num_workers=0,cur_time=1,excludeZero=False)).get_config()
    args.dataPath=str(root/f'data/mmsa/{a.dataset}/unaligned_50.pkl');args.device=torch.device('cuda:0')
    args.model_save_path=str(out/'best.pt');args.res_save_dir=str(out)
    loaders=MMDataLoader(args);ds=loaders['train'].dataset;args.train_samples=len(ds)
    if [768,ds.audio.shape[2],ds.vision.shape[2]]!=list(args.feature_dims):raise RuntimeError('Feature dimension mismatch')
    args.seq_lens=ds.get_seq_len()
    from corruption import wrap_loader_dataset
    loaders={k:DataLoader(wrap_loader_dataset(v.dataset,root,a.dataset,k,a.protocol,a.condition,a.rate),batch_size=args.batch_size,shuffle=(k=='train'),num_workers=0) for k,v in loaders.items()}
    (out/'mask_protocol.json').write_text(json.dumps({'protocol':a.protocol,'condition':a.condition,'rate':a.rate,'mask_paths':{k:getattr(v.dataset,'mask_path',None) for k,v in loaders.items()}},indent=2))
    if a.smoke:
        # Keep author label-map indices; only iterate the first samples.
        loaders={k:DataLoader(Subset(v.dataset,range(min(16,len(v.dataset)))),batch_size=8,shuffle=(k=='train')) for k,v in loaders.items()}
    (out/'effective_config.json').write_text(json.dumps(dict(args),default=str,indent=2))
    model=AMIO(args).to(args.device);trainer=ATIO().getTrain(args)
    trainer.do_train(model,loaders)
    if not (out/'best.pt').exists():raise RuntimeError('No validation checkpoint produced')
    if not np.isfinite(float(trainer.selected_valid['Loss'])):raise RuntimeError('Nonfinite validation loss')
    (out/'history.json').write_text(json.dumps(trainer.validation_history,default=float,indent=2))
    lock={'model':'Self_MM','dataset':a.dataset,'condition':a.condition,'protocol':a.protocol,'rate':a.rate,'smoke':a.smoke,'status':'validation_locked_no_test_evaluated','selected_epoch':trainer.selected_epoch,'valid':trainer.selected_valid,'selection':'author validation Loss and early stopping under same corruption protocol','adaptation':'whole-modality zero-input or fixed-rate frame-missing adaptation, not native Self-MM reproduction; original auxiliary-head losses and pseudo-label updates retained, even for missing heads','seed_policy':'no manual seeds; one run','upstream_trainer_sha256':hashlib.sha256((src/'trains/multiTask/SELF_MM.py').read_bytes()).hexdigest()}
    (out/'validation_lock.json').write_text(json.dumps(lock,default=float,indent=2));print(json.dumps(lock,default=float,indent=2))

if __name__=='__main__':main()
