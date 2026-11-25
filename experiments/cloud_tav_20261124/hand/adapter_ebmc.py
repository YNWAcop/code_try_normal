"""EBMC author model; single-run, validation-selected whole-modality adapter.

Not a claim of reproducing the paper: selection, padding-aware metrics and
teacher-eval handling are corrected explicitly. No test split is evaluated.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile


def unpack(root, dataset):
    dest = root / 'data/ebmc_author' / dataset
    marker = dest / '.extracted'
    # The queue independently validates every required clip before training.
    # A transferred, fully extracted feature directory need not have this marker.
    if marker.exists() or ((dest / (dataset + '_features_raw_2way.pkl')).is_file() and
                           all((dest / 'features' / f).is_dir() for f in
                               ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT'))):
        return dest
    if dest.exists():
        raise RuntimeError(f'Incomplete extraction at {dest}; inspect before retrying.')
    archive = root / 'data/author_archives' / (dataset + '.zip')
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for m in members:
            path = PurePosixPath(m.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in m.filename:
                raise RuntimeError('Unsafe zip member ' + m.filename)
        need = sum(m.file_size for m in members) + 2 * 1024**3
        if shutil.disk_usage(root).free < need:
            raise RuntimeError('Insufficient disk space for EBMC feature extraction and checkpoint reserve.')
        dest.mkdir(parents=True)
        z.extractall(dest)
    label = dest / (dataset + '_features_raw_2way.pkl')
    assert label.exists(), f'Unexpected author archive layout: {label}'
    marker.write_text('complete\n')
    return dest


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/MSA_Cloud_20261115'))
    p.add_argument('--dataset', choices=['mosi', 'mosei'], required=True)
    p.add_argument('--condition', choices=['A','T','V','TA','AV','TV','TAV','a','t','v','at','av','tv','atv'], default='TAV')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda')
    opt = p.parse_args()
    root = opt.root.resolve()
    ds = 'CMU' + opt.dataset.upper()
    opt.condition = ''.join(c for c in 'atv' if c in opt.condition.lower())
    display_condition = ''.join(c for c in 'TAV' if c.lower() in opt.condition)
    out = root / 'baseline_results' / ('EBMC_' + opt.dataset + '_' + display_condition + ('_smoke' if opt.smoke else ''))
    if out.exists():
        raise RuntimeError(f'Output exists, refusing to overwrite: {out}')
    source = root / 'baselines/upstream/EBMC'
    working = root / 'baselines/adapted/EBMC'
    if not working.exists():
        shutil.copytree(source, working)
    # Model files stay byte-identical. The adapter replaces the training entry.
    feature_dir = unpack(root, ds)
    out.mkdir(parents=True)
    os.environ['EBMC_DATASET_ROOT'] = str(feature_dir.parent)
    os.environ['EBMC_SAVED_ROOT'] = str(out / 'author_saved')
    sys.path[:0] = [str(working / 'EBMC'), str(working)]
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    import config
    from dataloader_cmumosi import CMUMOSIDataset
    from ebmc import build_model
    from loss import MaskedMSELoss
    from utils import generate_inputs

    torch.set_num_threads(4)
    if opt.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing unintended full CPU training.')
    device = torch.device(opt.device)
    args = argparse.Namespace(dataset=ds, hidden=256, n_classes=1, n_speakers=2,
        depth=4, num_heads=2, drop_rate=.55 if opt.dataset == 'mosi' else .6,
        attn_drop_rate=0., no_cuda=device.type != 'cuda', device=device,
        test_condition=opt.condition, lambda_msd=.5, lambda_cce=.1,
        lambda_emc=.1, lambda_imtd=.1)
    dsobj = CMUMOSIDataset(str(feature_dir / (ds + '_features_raw_2way.pkl')),
        str(feature_dir / 'features/wav2vec-large-c-UTT'),
        str(feature_dir / 'features/deberta-large-4-UTT'), str(feature_dir / 'features/manet_UTT'))
    train_ids, valid_ids, test_ids = map(set, (dsobj.trainVids, dsobj.valVids, dsobj.testVids))
    assert train_ids and valid_ids and not (train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids)
    tr = [i for i, vid in enumerate(dsobj.vids) if vid in train_ids]
    va = [i for i, vid in enumerate(dsobj.vids) if vid in valid_ids]
    if opt.smoke:
        tr, va = tr[:4], va[:4]
    batch = (64 if opt.dataset == 'mosi' else 32) if not opt.smoke else 2
    train_loader = DataLoader(Subset(dsobj, tr), batch_size=batch, shuffle=True, collate_fn=dsobj.collate_fn)
    valid_loader = DataLoader(Subset(dsobj, va), batch_size=batch, shuffle=False, collate_fn=dsobj.collate_fn)
    model = build_model(args, *dsobj.get_featDim()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    regression = MaskedMSELoss()
    stage_epochs = 1 if opt.smoke else (150 if opt.dataset == 'mosi' else 100)
    provenance = dict(method='EBMC_author_model_validation_adapter', dataset=opt.dataset,
        condition=opt.condition, seed_policy='no seed calls; one run per condition',
        protocol='seven independently trained whole-modality conditions; video context retained',
        first_stage_epochs=stage_epochs, second_stage_epochs=stage_epochs, batch_size=batch,
        stage1_selection='minimum validation mean MAE of A/T/V unimodal heads',
        stage2_selection='minimum validation MAE of fused prediction',
        differences=['official test-selected epoch replaced with validation selection',
                     'padding excluded; MAE/Corr on all valid utterances, Acc2/F1 exclude zero',
                     'Stage-I random fusion logits never used to select teacher',
                     'teacher kept eval/frozen after model.train()',
                     'no test evaluation; no seed sweeps'], smoke=opt.smoke,
        source_hashes={str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in source.rglob('*.py')})
    save(out / 'protocol.json', provenance)

    def epoch(loader, training, first_stage):
        model.train(training)
        if hasattr(model, 'teacher_model'):
            model.teacher_model.eval()
        truth, fused, uni, ids = [], [], [], []
        for data in loader:
            tensors = [x.to(device) for x in data[:9]]
            ah, th, vh, ag, tg, vg, qmask, umask, label = tensors
            length, size = ah.shape[:2]
            condition = 'atv' if first_stage else opt.condition
            masks = [torch.full((length, size, 1), float(c in condition), device=device) for c in 'atv']
            am, tm, vm = masks
            x = generate_inputs(ah*am, th*tm, vh*vm, ag*am, tg*tm, vg*vm, qmask)[0]
            mask = generate_inputs(am, tm, vm, am, tm, vm, qmask)[0]
            # Eval never passes labels into the network or computes label-based EMC.
            with torch.set_grad_enabled(training):
                _, pc, pa, pt, pv, losses, _ = model(x, mask, umask, first_stage,
                    label if training else None, do_cf=training)
                if training:
                    objective = sum(regression(v.reshape(-1, 1), label, umask) for v in (pa, pt, pv)) if first_stage else regression(pc.reshape(-1, 1), label, umask)
                    objective = objective + .5*losses['disentangle'] + .1*losses['cfd']
                    if not first_stage:
                        objective = objective + .1*losses['emc'] + .1*losses['imtd']
                    if not torch.isfinite(objective):
                        raise RuntimeError('Nonfinite EBMC training loss; output preserved, no automatic retry.')
                    optimizer.zero_grad()
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            good = umask.bool()
            truth.append(label[good].detach().cpu().numpy())
            fused.append(pc.squeeze(-1)[good].detach().cpu().numpy())
            uni.append(torch.stack([pa.squeeze(-1)[good], pt.squeeze(-1)[good], pv.squeeze(-1)[good]], 1).detach().cpu().numpy())
            for vid in data[9]:
                ids.extend(dsobj.videoIDs[vid])
        y, pred, unip = np.concatenate(truth), np.concatenate(fused), np.concatenate(uni)
        assert len(y) == len(ids)
        value = float(np.abs(unip-y[:,None]).mean()) if first_stage else float(np.abs(pred-y).mean())
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite validation metric')
        return value, dict(y=y, pred=pred, unimodal=unip, ids=np.asarray(ids))

    history = []
    for stage in (1, 2):
        first = stage == 1
        best, chosen = float('inf'), None
        for ep in range(stage_epochs):
            epoch(train_loader, True, first)
            score, predictions = epoch(valid_loader, False, first)
            history.append(dict(stage=stage, epoch=ep+1, valid_mae=score))
            save(out / 'history.json', history)
            save(out / 'heartbeat.json', dict(time=time.time(), stage=stage, epoch=ep+1))
            print(history[-1], flush=True)
            if score < best:
                best, chosen = score, ep+1
                torch.save(model.state_dict(), out / f'stage{stage}_best.pt')
                np.savez_compressed(out / f'stage{stage}_validation_predictions.npz', **predictions)
        if first:
            stage_path = Path(config.MODEL_DIR) / 'stage_1'
            stage_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / 'stage1_best.pt', stage_path / f'{ds}_{opt.condition}_stage1_best.pth')
            # Student continues its last Stage-I state; teacher uses best valid checkpoint.
            # Preload teacher before optimizer updates in Stage-II; keep it frozen.
            import ebmc
            model.teacher_model = ebmc.build_model(args, *dsobj.get_featDim()).to(device)
            model.teacher_model.load_state_dict(torch.load(out / 'stage1_best.pt', map_location=device, weights_only=True))
            model.teacher_model.eval()
            for parameter in model.teacher_model.parameters():
                parameter.requires_grad_(False)
        else:
            provenance.update(status='validation_locked_no_test_evaluated', selected_epoch=chosen, valid_score=best,
                              checkpoint=str(out / 'stage2_best.pt'))
            save(out / 'validation_lock.json', provenance)
    print('EBMC VALIDATION LOCKED:', out, flush=True)


if __name__ == '__main__':
    main()
