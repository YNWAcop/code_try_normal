"""Frozen V1 A/V/AV input ablation: paired initialization, validation-only selection."""
import argparse, contextlib, csv, json, os, platform, shutil, sys, time, zipfile
from pathlib import Path
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P / 'src'))
import joblib
import numpy as np
import sklearn
import torch
from evidence_msa.text_absent import v1
from evidence_msa.input_views import select_inputs
from evidence_msa.common.metrics import atomic_json, digest, scores

DATASETS = ('mosi', 'mosei')
CONDITIONS = ('A', 'V', 'AV')
MODES = ('native', 'enhanced')
METRICS = ('Acc7', 'Acc5', 'Acc2_nonzero', 'F1_weighted_nonzero', 'MAE', 'Corr')
COUNTS = {'mosi': (1284, 229, 686), 'mosei': (16326, 1871, 4659)}
PROTOCOL = dict(epochs=40, batch_size=64, lr=3e-4, weight_decay=1e-3,
                gradient_clip=5, selection='minimum validation MAE; earliest tie',
                conditions=list(CONDITIONS), modes=list(MODES),
                pairing='same initial weights and starting RNG per dataset/condition; no manual seed',
                scope='V1 text-absent branch; fresh paired input ablation, not historical weight reproduction')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); atomic_json(p, x)

def torch_save(p, x):
    tmp = p.with_suffix('.tmp'); torch.save(x, tmp); os.replace(tmp, p)

def rng_get(device):
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if device == 'cuda' else None)

def rng_set(rng):
    torch.set_rng_state(rng['cpu'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng['cuda'])

def load_pt(p):
    return torch.load(p, map_location='cpu', weights_only=False)

def code_check():
    manifest = read(P / 'manifest.json')
    for n, h in manifest.items():
        if digest(P / n) != h:
            raise RuntimeError('Package source changed: ' + n)
    return digest(P / 'manifest.json')

def state(a, stage, **fields):
    save(a.run_root / 'status.json', dict(stage=stage, updated=time.time(), pid=os.getpid(), smoke=a.smoke, **fields))

def check(a):
    source = code_check()
    if not a.smoke and not torch.cuda.is_available():
        raise RuntimeError('Formal training requires a working CUDA Python environment')
    a.run_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(a.run_root).free < (0.1 if a.smoke else 6) * 1024**3:
        raise RuntimeError('Insufficient disk reserve (formal minimum 6 GiB)')
    expected = read(P / 'expected_data.json')['files']; inputs = {}
    for ds in DATASETS:
        ids, groups = {}, {}
        for idx, split in enumerate(('train', 'valid', 'test')):
            p = a.data_root / 'ready' / ds / (split + '.npz')
            h = digest(p)
            if not a.smoke and h != expected[ds + '/' + split + '.npz']['sha256']:
                raise RuntimeError('Ready cache hash differs: ' + str(p))
            with np.load(p, allow_pickle=False) as z:
                assert set(v1.KEYS + ['ids', 'groups', 'y']).issubset(z.files)
                ids[split] = set(z['ids'].astype(str)); groups[split] = set(z['groups'].astype(str))
                n = len(z['ids']); assert n == len(ids[split])
                if not a.smoke: assert n == COUNTS[ds][idx]
            inputs[ds + '/' + split] = dict(sha256=h, n=n)
        for s, t in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
            assert not ids[s] & ids[t] and not groups[s] & groups[t], 'Split overlap'
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   sklearn=sklearn.__version__, device=a.device,
                   gpu=torch.cuda.get_device_name(0) if a.device == 'cuda' else None)
    contract = dict(source_sha=source, inputs=inputs, protocol=PROTOCOL, smoke=a.smoke, runtime=runtime)
    cp = a.run_root / 'contract.json'
    if cp.exists():
        assert read(cp) == contract, 'Source/data/runtime changed; preserve this run and inspect the difference'
    else: save(cp, contract)
    save(a.run_root / 'preflight.json', dict(passed=True, test_usage='File hashes and IDs/groups only; no test labels read', **runtime))
    return digest(cp)

def load_data(a, ds, split, mode, condition):
    with np.load(a.data_root / 'ready' / ds / (split + '.npz'), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if a.smoke: d = {k: v[:16] for k, v in d.items()}
    d = select_inputs(d, mode)
    # Eliminate all unavailable raw sources before scaling, in addition to V1's forward mask.
    for k in v1.KEYS:
        if k == 'T' or (k.startswith('A_') and 'A' not in condition) or (k.startswith('V_') and 'V' not in condition):
            d[k] = np.zeros_like(d[k])
    for k in v1.KEYS + ['y']:
        assert np.isfinite(d[k]).all(), (ds, split, k)
    return d

def make_arrays(d):
    x = v1.arrays(d, 'reliable')
    x.update(anchor=np.zeros(len(d['y']), dtype='float32'), A_res=x['A'].copy(), V_res=x['V'].copy())
    return x

def predict(model, x, condition, device):
    model.eval(); out = []
    bits = torch.tensor([[m in condition for m in 'TAV']], dtype=torch.bool)
    with torch.no_grad():
        for i in range(0, len(x['T']), 256):
            b = v1.tensors(x, slice(i, i + 256), device)
            out.append(model(b, bits.expand(len(b['T']), -1)).cpu().numpy())
    return np.concatenate(out)

def visibility_check(model, x, condition, device):
    model.eval(); b = v1.tensors(x, slice(0, 8), device)
    alt = {k: v.clone() for k, v in b.items()}
    for k in ('T', 'anchor', 'A_res', 'V_res'): alt[k][:] = 99
    if 'A' not in condition: alt['A'][:] = 99
    if 'V' not in condition:
        for k in ('V', 'timeline', 'quality'): alt[k][:] = 99
    bits = torch.tensor([[m in condition for m in 'TAV']], dtype=torch.bool).expand(len(b['T']), -1)
    with torch.no_grad():
        torch.testing.assert_close(model(b, bits), model(alt, bits), atol=1e-6, rtol=1e-6)

def unit(a, ds, condition, mode):
    return a.run_root / 'runs' / (ds + '_' + condition) / mode

def verify_lock(u):
    lock = read(u / 'selection_lock.json')
    for name, h in lock['artifacts'].items():
        assert digest(u / name) == h, ('Locked artifact changed', u, name)
    return lock

def jobs():
    return [(d, c, m) for d in DATASETS for c in CONDITIONS for m in MODES]

def train_unit(a, ds, condition, mode, contract):
    u = unit(a, ds, condition, mode); u.mkdir(parents=True, exist_ok=True)
    if (u / 'selection_lock.json').exists():
        assert verify_lock(u)['contract_sha'] == contract
        return
    state(a, 'preparing', job=f'{ds}_{condition}_{mode}',
          completed=sum((unit(a,*j) / 'selection_lock.json').exists() for j in jobs()), expected=12)
    tr, va = [load_data(a, ds, s, mode, condition) for s in ('train', 'valid')]
    scalers = v1.scaling(tr, [va]); x, z = make_arrays(tr), make_arrays(va)
    dims = {k: x[k].shape[-1] for k in ('T', 'A', 'V')}
    appdim = tr['V_native'].shape[-1] + tr['V_expression'].shape[-1]
    model = v1.Evidence(dims, appdim).to(a.device)
    pair = u.parent / 'pair_initial.pt'
    if not pair.exists():
        torch_save(pair, dict(model=model.state_dict(), rng=rng_get(a.device), dims=dims, appdim=appdim))
    init = load_pt(pair); assert init['dims'] == dims and init['appdim'] == appdim
    model.load_state_dict(init['model']); opt = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['lr'], weight_decay=PROTOCOL['weight_decay'])
    rng_set(init['rng'])
    history = []; best = float('inf'); best_state = None; selected = 0; start = 0
    last = u / 'last.pt'
    if last.exists():
        ck = load_pt(last); assert ck['contract_sha'] == contract and ck['pair_sha'] == digest(pair)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); rng_set(ck['rng'])
        history, best, best_state, selected, start = ck['history'], ck['best'], ck['best_model'], ck['selected_epoch'], ck['epoch']
    epochs = 2 if a.smoke else PROTOCOL['epochs']
    bits = torch.tensor([[m in condition for m in 'TAV']], dtype=torch.bool)
    for ep in range(start, epochs):
        model.train(); total = 0.; started = time.time()
        for ix in torch.randperm(len(tr['y'])).split(PROTOCOL['batch_size']):
            ix = ix.numpy(); opt.zero_grad()
            pred = model(v1.tensors(x, ix, a.device), bits.expand(len(ix), -1))
            loss = v1.loss(pred, torch.as_tensor(tr['y'][ix], device=a.device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            total += loss.item() * len(ix)
        pred = predict(model, z, condition, a.device)
        score = float(np.abs(pred - va['y']).mean()); assert np.isfinite(score)
        if score < best:
            best = score; selected = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=ep + 1, train_loss=total / len(tr['y']), valid_MAE=score, seconds=time.time() - started))
        torch_save(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_get(a.device),
                             history=history, best=best, best_model=best_state, selected_epoch=selected,
                             epoch=ep + 1, contract_sha=contract, pair_sha=digest(pair)))
        save(u / 'history.json', history)
        state(a, 'training', job=f'{ds}_{condition}_{mode}', epoch=ep + 1, epochs=epochs,
              completed=sum((unit(a,*j) / 'selection_lock.json').exists() for j in jobs()), expected=12)
        print(ds, condition, mode, history[-1], flush=True)
        if a.stop_after_epoch and ep + 1 >= a.stop_after_epoch:
            raise RuntimeError('Intentional synthetic resume-test interruption')
    model.load_state_dict(best_state); visibility_check(model, z, condition, a.device)
    pred = predict(model, z, condition, a.device)
    torch_save(u / 'best.pt', dict(model=best_state, dims=dims, appdim=appdim))
    joblib.dump(scalers, u / 'scalers.joblib')
    np.savez_compressed(u / 'predictions_valid.npz', ids=va['ids'], y=va['y'], pred=pred)
    save(u / 'metrics_valid.json', scores(va['y'], pred))
    save(u / 'selection_lock.json', dict(dataset=ds, condition=condition, mode=mode, contract_sha=contract,
        pair_sha=digest(pair), selected_epoch=selected, valid_MAE=best, epochs=epochs,
        parameters=sum(p.numel() for p in model.parameters()), test_opened=False,
        artifacts={n: digest(u / n) for n in ('best.pt', 'scalers.joblib', 'predictions_valid.npz', 'metrics_valid.json', 'history.json')}))

def train(a):
    state(a, 'checking')
    contract = check(a)
    for ds, condition, mode in jobs():
        train_unit(a, ds, condition, mode, contract)
    for j in jobs(): verify_lock(unit(a, *j))
    summarize(a, 'valid')
    save(a.run_root / 'training_done.json', dict(contract_sha=contract, expected=12, test_opened=False))
    evaluated = (a.run_root / 'evaluation_done.json').exists()
    state(a, 'complete' if evaluated else 'training_complete', completed=12, expected=12, test_opened=evaluated)

def evaluate(a):
    contract = check(a); assert read(a.run_root / 'training_done.json')['contract_sha'] == contract
    # Verify every model before opening any test features or labels.
    for j in jobs(): assert verify_lock(unit(a, *j))['contract_sha'] == contract
    for ds, condition, mode in jobs():
        u = unit(a, ds, condition, mode); done = u / 'test_done.json'
        if done.exists():
            d = read(done); assert d['selection_sha'] == digest(u / 'selection_lock.json')
            assert d['prediction_sha'] == digest(u / 'predictions_test.npz'); continue
        state(a, 'evaluating', job=f'{ds}_{condition}_{mode}')
        d = load_data(a, ds, 'test', mode, condition)
        v1.apply_scalers(d, joblib.load(u / 'scalers.joblib')); x = make_arrays(d)
        ck = load_pt(u / 'best.pt'); model = v1.Evidence(ck['dims'], ck['appdim']).to(a.device); model.load_state_dict(ck['model'])
        visibility_check(model, x, condition, a.device); pred = predict(model, x, condition, a.device)
        np.savez_compressed(u / 'predictions_test.npz', ids=d['ids'], y=d['y'], pred=pred)
        save(u / 'metrics_test.json', scores(d['y'], pred))
        save(done, dict(selection_sha=digest(u / 'selection_lock.json'), prediction_sha=digest(u / 'predictions_test.npz')))
    summarize(a, 'test'); save(a.run_root / 'evaluation_done.json', dict(contract_sha=contract, rows=12))
    state(a, 'complete', completed=12, expected=12, test_opened=True)
    export(a)

def write_csv(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def summarize(a, split):
    dest = a.run_root / 'tables'; dest.mkdir(exist_ok=True); rows = []; refs = {}; audit = []
    for ds, condition, mode in jobs():
        u = unit(a, ds, condition, mode); verify_lock(u); p = u / ('predictions_' + split + '.npz')
        if split == 'test':
            done = read(u / 'test_done.json')
            assert done['selection_sha'] == digest(u / 'selection_lock.json') and done['prediction_sha'] == digest(p)
        with np.load(p, allow_pickle=False) as z:
            key = (ds, condition)
            if key in refs:
                assert np.array_equal(z['ids'], refs[key][0]) and np.array_equal(z['y'], refs[key][1])
            else: refs[key] = (z['ids'], z['y'])
            result = scores(z['y'], z['pred'])
        row = dict(dataset=ds, condition=condition, variant='B0' if mode == 'native' else 'B1', input=mode, n=result['n_all'])
        row.update({k: result['metrics'][k] for k in METRICS}); rows.append(row)
        audit.append(dict(path=str(p.relative_to(a.run_root)), sha256=digest(p)))
    write_csv(dest / (split + '_results.csv'), rows)
    gains = []
    for ds in DATASETS:
        for condition in CONDITIONS:
            base, enhanced = [r for r in rows if r['dataset'] == ds and r['condition'] == condition]
            g = dict(dataset=ds, condition=condition, contrast='B1-B0; positive=improvement')
            for k in METRICS:
                g[k] = None if base[k] is None or enhanced[k] is None else (enhanced[k] - base[k]) * (-1 if k == 'MAE' else 1)
            gains.append(g)
    write_csv(dest / (split + '_gains.csv'), gains)
    means = []
    for ds in DATASETS:
        for mode in MODES:
            sub = [r for r in rows if r['dataset'] == ds and r['input'] == mode]
            means.append(dict(dataset=ds, input=mode, condition='Avg(A,V,AV)', **{
                k: None if any(r[k] is None for r in sub) else float(np.mean([r[k] for r in sub])) for k in METRICS}))
    write_csv(dest / (split + '_means.csv'), means)
    save(dest / (split + '_audit.json'), dict(predictions=audit, rows=12, smoke=a.smoke))

def export(a):
    assert read(a.run_root / 'evaluation_done.json')['contract_sha'] == digest(a.run_root / 'contract.json')
    summarize(a, 'test')
    target = a.run_root / ('SMOKE_results.zip' if a.smoke else 'AV_ablation_results.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in a.run_root.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.csv', '.npz'):
                z.write(p, p.relative_to(a.run_root))
    target.with_suffix('.zip.sha256').write_bytes((digest(target) + '  ' + target.name + '\n').encode('ascii'))
    print(target, flush=True)

@contextlib.contextmanager
def queue_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['check', 'train', 'evaluate', 'export', 'status'])
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/MSA_Cloud_20261115/data'))
    p.add_argument('--run-root', type=Path, default=P / 'outputs/formal')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after-epoch', type=int, default=0, help=argparse.SUPPRESS)
    a = p.parse_args(); a.device = 'cpu' if a.smoke else 'cuda'
    assert not a.stop_after_epoch or a.smoke, 'Synthetic interruption only'
    torch.set_num_threads(2 if a.smoke else 4)
    if a.action == 'status':
        d = read(a.run_root / 'status.json'); pid = d.get('pid')
        if os.name == 'nt':
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x1000, False, pid or 0)
            alive = bool(handle)
            if handle:
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]; kernel.CloseHandle(handle)
        else:
            try: os.kill(pid, 0); alive = True
            except (OSError, TypeError): alive = False
        d['recorded_process_alive'] = alive; d['seconds_since_update'] = time.time() - d['updated']
        print(json.dumps(d, ensure_ascii=False, indent=2)); return
    with queue_lock(a.run_root):
        try:
            if a.action == 'check': print(json.dumps(dict(passed=True, contract_sha=check(a))))
            elif a.action == 'train': train(a)
            elif a.action == 'evaluate': evaluate(a)
            else: check(a); export(a)
        except Exception as e:
            state(a, 'failed', error=repr(e)); raise

if __name__ == '__main__': main()
