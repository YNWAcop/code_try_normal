"""Synthetic CPU checks; never substitutes for a cloud data run."""
import argparse, importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
import av_ablation as av

def main():
    p = argparse.ArgumentParser(); p.add_argument('--work', type=Path, required=True); a = p.parse_args()
    work = a.work.resolve(); work.mkdir(parents=True, exist_ok=False)
    data = work / 'data'; rng = np.random.default_rng()
    for ds in av.DATASETS:
        for split in ('train', 'valid', 'test'):
            out = data / 'ready' / ds; out.mkdir(parents=True, exist_ok=True)
            values = dict(ids=np.array([f'{ds}_{split}_{i}' for i in range(16)]),
                          groups=np.array([f'{ds}_{split}_g{i//2}' for i in range(16)]),
                          y=np.linspace(-3, 3, 16).astype('float32'))
            for k in av.v1.KEYS:
                shape = (16, 2, 768) if k == 'V_timeline' else (16, 5 if k == 'V_quality' else 7)
                values[k] = rng.normal(size=shape).astype('float32')
            np.savez_compressed(out / (split + '.npz'), **values)
    args = argparse.Namespace(data_root=data, run_root=work / 'resumed', smoke=True, device='cpu', stop_after_epoch=1)
    torch.set_num_threads(2)
    contract = av.check(args)
    try: av.train_unit(args, 'mosi', 'A', 'native', contract)
    except RuntimeError as e: assert 'Intentional' in str(e)
    else: raise AssertionError('Expected synthetic interruption')
    assert not (av.unit(args,'mosi','A','native') / 'selection_lock.json').exists()
    # An uninterrupted control gets the exact same sampled initialization and RNG.
    direct = argparse.Namespace(**vars(args)); direct.run_root = work / 'uninterrupted'; direct.stop_after_epoch = 0
    direct_contract = av.check(direct); assert direct_contract == contract
    du = av.unit(direct, 'mosi', 'A', 'native'); du.parent.mkdir(parents=True)
    shutil.copy2(av.unit(args,'mosi','A','native').parent / 'pair_initial.pt', du.parent / 'pair_initial.pt')
    av.train_unit(direct, 'mosi', 'A', 'native', direct_contract)
    args.stop_after_epoch = 0; av.train(args)
    c1 = av.load_pt(du / 'last.pt'); c2 = av.load_pt(av.unit(args,'mosi','A','native') / 'last.pt')
    for k in c1['model']: torch.testing.assert_close(c1['model'][k], c2['model'][k], rtol=0, atol=0)
    assert c1['best'] == c2['best'] and c1['selected_epoch'] == c2['selected_epoch']
    assert av.read(args.run_root / 'status.json')['stage'] == 'training_complete'
    assert not list(args.run_root.rglob('predictions_test.npz'))
    for ds in av.DATASETS:
        for condition in av.CONDITIONS:
            n, e = [av.verify_lock(av.unit(args,ds,condition,m)) for m in av.MODES]
            assert n['parameters'] == e['parameters'] and n['pair_sha'] == e['pair_sha']
            d = av.load_data(args, ds, 'train', 'native', condition)
            for k in ('A_words','V_lip','V_words','V_expression','V_quality','V_timeline','T'):
                assert not np.count_nonzero(d[k]), (condition, k)
            if 'A' not in condition: assert not np.count_nonzero(d['A_native'])
            if 'V' not in condition: assert not np.count_nonzero(d['V_native'])
    av.evaluate(args)
    saved = {str(p): av.digest(p) for p in args.run_root.rglob('predictions_test.npz')}
    av.evaluate(args)
    assert saved == {str(p): av.digest(p) for p in args.run_root.rglob('predictions_test.npz')}
    assert len(saved) == 12
    # Exercise lock rejection rather than merely checking a digest helper.
    target = av.unit(args,'mosi','A','native') / 'predictions_valid.npz'
    original = target.read_bytes(); target.write_bytes(original + b'changed')
    try: av.verify_lock(target.parent)
    except AssertionError: pass
    else: raise AssertionError('Changed lock artifact was accepted')
    target.write_bytes(original)
    # Check retained V1 forward against its exact text-absent algebra, including gradients.
    for condition in av.CONDITIONS:
        u = av.unit(args,'mosi',condition,'enhanced'); ck = av.load_pt(u / 'best.pt')
        model = av.v1.Evidence(ck['dims'],ck['appdim']); model.load_state_dict(ck['model']); model.eval()
        d = av.load_data(args,'mosi','valid','enhanced',condition); av.v1.apply_scalers(d,av.joblib.load(u/'scalers.joblib'))
        b = av.v1.tensors(av.make_arrays(d),slice(0,8),'cpu'); bits = torch.tensor([[False,'A' in condition,'V' in condition]]).expand(8,-1)
        full = model(b,bits)
        ax = model.a(b['A'] * bits[:,1:2]) * bits[:,1:2]
        vx = model.visual(b['V']*bits[:,2:3],b['timeline']*bits[:,2:3,None],b['quality']*bits[:,2:3]) * bits[:,2:3]
        direct_pred = (3 * torch.tanh(model.no_text(torch.cat([ax,vx,bits[:,1:].float()],1)).squeeze(-1)/3)).clamp(-3,3)
        torch.testing.assert_close(full,direct_pred,rtol=0,atol=0)
        model.zero_grad(); av.v1.loss(full,torch.as_tensor(d['y'][:8])).backward()
        grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(); av.v1.loss(direct_pred,torch.as_tensor(d['y'][:8])).backward()
        for n,p in model.named_parameters():
            if p.grad is not None: torch.testing.assert_close(grads[n],p.grad,atol=1e-7,rtol=1e-7)
    report = dict(passed=True, synthetic_only=True, formal_cloud_run=False, checks=[
        '12 synthetic train/evaluate/export jobs', 'native/enhanced matching dimensions and paired initialization',
        'unavailable sources zeroed before transforms', 'forward unavailable-source invariance',
        'V1 text-absent algebra and gradients', 'epoch checkpoint resume equals uninterrupted CPU training',
        'no test predictions before all training locks', 'completed tests idempotent', 'changed artifacts rejected'],
        rows=12, runtime=av.read(args.run_root/'preflight.json'))
    av.save(work/'LOCAL_TEST_REPORT.json',report); print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
