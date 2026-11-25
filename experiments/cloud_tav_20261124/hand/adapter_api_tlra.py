"""Single upstream training run; test data are not passed to the trainer."""
import argparse
import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--method', choices=['API', 'TLRA'], required=True)
    p.add_argument('--dataset', choices=['mosi', 'mosei'], required=True)
    p.add_argument('--rate', type=float, default=.3)
    p.add_argument('--protocol', choices=['whole', 'frame'], default='whole')
    p.add_argument('--condition', choices=['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'], default='TAV')
    p.add_argument('--smoke', action='store_true')
    o = p.parse_args()
    root = o.root.resolve()
    tag = f'frame{o.rate:.1f}' if o.protocol == 'frame' else o.condition
    if o.protocol == 'frame' and o.rate not in [i/10 for i in range(1,8)]:
        raise ValueError('Frame rate must be 0.1 ... 0.7')
    if o.protocol == 'frame':
        o.condition = 'TAV'
    name = f'{o.method}_{o.dataset}_{tag}' + ('_smoke' if o.smoke else '')
    out = root / 'baseline_results' / name
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'source'
    upstream = root / 'baselines/upstream' / o.method
    shutil.copytree(upstream, work, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    relative = 'trains/singleTask/APRIL.py' if o.method == 'API' else 'trains/singleTask/TLRA.py'
    path = work / relative
    old = path.read_text(encoding='utf-8')
    new = old
    if o.method == 'API':
        assert 'verbose=True, ' in new
        new = new.replace('verbose=True, ', '')  # torch 2.7 removed this argument
        new = new.replace("(np.round(len(dataloader['train']) / 10) * 10)", "max(1, np.round(len(dataloader['train']) / 10) * 10)")
        new = new.replace('(np.round(len(dataloader) / 10) * 10)', 'max(1, np.round(len(dataloader) / 10) * 10)')
    else:
        target = 'self._eval_missing(model[0], dataloader["test"], mode="VAL", keep=keep)'
        assert new.count(target) == 1
        new = new.replace(target, target.replace('["test"]', '["valid"]'))
        new = new.replace('epoch_results["test"].append(val_results)', '# No test evaluation during training')
    if o.smoke:
        target = '            if epochs - best_epoch >= self.args.early_stop:' if o.method == 'API' else '            no_improve_epochs = epochs - best_epoch'
        assert target in new
        new = new.replace(target, '            if epochs >= 1:\n                return epoch_results if return_epoch_results else None\n' + target)
    path.write_text(new, encoding='utf-8')
    (out / 'source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=relative, tofile=relative)), encoding='utf-8')
    (out / 'source_sha256.json').write_text(json.dumps({relative: hashlib.sha256(old.encode()).hexdigest()}, indent=2))
    if o.method == 'API' and o.protocol == 'whole':
        mp = work / 'trains/singleTask/model/april.py'
        ms = mp.read_text(encoding='utf-8')
        # Upstream mapping is 0=text, 1=vision, 2=audio (not T,A,V).
        indices = [i for i,m in enumerate('TVA') if m in o.condition]
        needle = '        ava_modal_idx = sample(modal_idx, num_modal)'
        assert ms.count(needle) == 1
        patched = ms.replace(needle, f'        ava_modal_idx = {indices!r}\n        num_modal = len(ava_modal_idx)')
        mp.write_text(patched, encoding='utf-8')
        (out / 'fixed_mask.patch').write_text(''.join(difflib.unified_diff(ms.splitlines(True), patched.splitlines(True))), encoding='utf-8')
    sys.path.insert(0, str(work))
    os.chdir(work)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from config import get_config_regression
    from data_loader import MMDataset
    from trains import ATIO
    logging.basicConfig(level=logging.INFO)
    assert torch.cuda.is_available(), 'CUDA is required for this paid cloud queue'
    torch.set_num_threads(4)
    model_name = 'april' if o.method == 'API' else 'TLRA'
    configfile = work / 'config' / ('config_april.json' if o.method == 'API' else 'config.json')
    args = get_config_regression(model_name, o.dataset, configfile)
    args.update(featurePath=str(root / 'data/api_aligned' / o.dataset / 'aligned_50.pkl'),
                pretrained=str(root / 'models/bert-base-uncased'),
                model_save_path=out / 'best.pt', device=torch.device('cuda:0'),
                train_mode='regression', mode='train', is_training=True,
                feature_T='', feature_A='', feature_V='', cur_seed='single_unseeded',
                mr=0.0, missing_keep_mode=o.condition, missing_missing_mode=o.condition)
    datasets = {s: MMDataset(args, mode=s) for s in ('train', 'valid')}
    from corruption import wrap_loader_dataset
    # Whole-modality methods retain their original complete training views for
    # prototype/teacher learning. Held-out inference sees only available inputs.
    datasets = {s: wrap_loader_dataset(d, root, o.dataset, s, o.protocol, o.condition, o.rate)
                if (o.protocol == 'frame' or s == 'valid') else d for s,d in datasets.items()}
    if o.smoke:
        datasets = {s: Subset(d, range(min(len(d), 8))) for s, d in datasets.items()}
        args.batch_size = 4
        args.update_epochs = 1
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == 'train'), num_workers=0)
               for s, d in datasets.items()}
    # No train/test seed-setting call, and no multi-run loop.
    if o.method == 'API':
        from trains.singleTask.model.april import APRIL
        model = APRIL(args).to(args.device)
    else:
        from trains.singleTask.model.TLRA import TLRA
        model = TLRA(args).to(args.device)
    trainer = ATIO().getTrain(args)
    history = []
    original = trainer.do_test if o.method == 'API' else trainer._eval_missing
    def record(*a, **kw):
        result = original(*a, **kw)
        history.append(result)
        (out / 'validation_history.json').write_text(json.dumps(history, indent=2, default=float))
        return result
    if o.method == 'API':
        trainer.do_test = record
    else:
        trainer._eval_missing = record
    trainer.do_train(model if o.method == 'API' else [model], loaders, return_epoch_results=False)
    assert (out / 'best.pt').exists(), 'Training returned without a validation checkpoint'
    lock = dict(status='validation_locked_no_test_evaluated', method=o.method,
                dataset=o.dataset, condition=tag, smoke=o.smoke,
                selection=args.KeyEval, validation_epochs=len(history),
                seed_policy='no manually fixed training seed; one run',
                protocol='frame-drop adaptation on aligned MMSA sequences' if o.protocol == 'frame' else 'whole modality fixed combination adaptation',
                note='Not the original API missing-rate protocol; pretrained input resource differs across baseline families.',
                checkpoint_sha256=hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest(),
                config={k: str(v) if isinstance(v, (Path, torch.device)) else v for k,v in args.items()})
    (out / 'validation_lock.json').write_text(json.dumps(lock, indent=2, default=str))
    print('COMPLETED', out, flush=True)


if __name__ == '__main__':
    main()
