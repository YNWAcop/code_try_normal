"""Resume-safe MOSEI visual speech extraction, using RGB from public ZIP ranges.

CPU workers fetch and align mouths; only the main thread performs GPU inference.
Matches extract_visual_speech.py's MOSI aggregation and CTC decoding exactly.
No dataset transcript, audio decoder, sentiment label, or neighboring clip is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import uuid
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent
sys.path.insert(0, str(RUN / 'vendor'))
import cv2
import inflate64
import numpy as np
import requests
import torch
from extract_visual_speech import mouth_chunks, text_from_ids, E2E, CODE

URL = 'https://hf-mirror.com/datasets/tamb2203579/CMU-MOSEI/resolve/main/Raw.zip'
ZIP_BYTES = 27280713458
AUTHOR_IDS = Path(r'E:\UserData\Desktop\idea_final\data\processed\ebmc_features_20261111\all_author\mosei_features.npz')
NETWORK = threading.local()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def fetch(offset, length):
    if offset < 0 or length <= 0 or offset + length > ZIP_BYTES:
        raise ValueError('Invalid ZIP range')
    errors = []
    for attempt in range(3):
        try:
            if not hasattr(NETWORK, 'session'):
                NETWORK.session = requests.Session()
            # Redirect URLs remain in memory, never in persisted logs.
            endpoint = getattr(NETWORK, 'cdn', URL) if attempt == 0 else URL
            headers = {'Range': f'bytes={offset}-{offset + length - 1}', 'Accept-Encoding': 'identity'}
            with NETWORK.session.get(endpoint, headers=headers, stream=True, timeout=(15, 70)) as response:
                if response.status_code != 206:
                    raise ValueError(f'Expected HTTP 206, got {response.status_code}')
                expected = f'bytes {offset}-{offset + length - 1}/{ZIP_BYTES}'
                if response.headers.get('Content-Range') != expected:
                    raise ValueError('Unexpected Content-Range')
                data = response.raw.read(length + 1)
                if len(data) != length:
                    raise ValueError('Incomplete range payload')
                NETWORK.cdn = response.url
                return data
        except Exception as exc:
            # requests exceptions can contain temporary signed URLs.
            errors.append(type(exc).__name__)
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError('ZIP range fetch failed: ' + ', '.join(errors))


def decode_member(name, info):
    length = min(int(info['compressed']) + 4096, ZIP_BYTES - int(info['offset']))
    payload = fetch(int(info['offset']), length)
    if len(payload) < 30 or payload[:4] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP local header')
    header = struct.unpack('<4s5H3L2H', payload[:30])
    flags, method = header[2], header[3]
    name_len, extra_len = header[-2:]
    begin = 30 + name_len + extra_len
    if begin + info['compressed'] > len(payload):
        # A legitimate long local extra field needs one larger range.
        payload = fetch(int(info['offset']), begin + int(info['compressed']))
    local_name = payload[30:30 + name_len].decode('utf-8' if flags & 0x800 else 'cp437')
    if local_name != name or method != info['method'] or flags & 1:
        raise ValueError('ZIP name/method mismatch or encrypted member')
    compressed = payload[begin:begin + info['compressed']]
    if method == 9:
        raw = inflate64.Inflater().inflate(compressed)
    elif method == 8:
        raw = zlib.decompress(compressed, -15)
    elif method == 0:
        raw = compressed
    else:
        raise ValueError(f'Unsupported ZIP compression: {method}')
    if len(raw) != info['size'] or zlib.crc32(raw) != info['crc']:
        raise ValueError('ZIP CRC32 or uncompressed size mismatch')
    return raw


def cached_ok(path, ident, fingerprint):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z['id']) != ident or str(z['fingerprint']) != fingerprint:
                return False
            for key, shape in {'pooled': (1536,), 'temporal': (16, 768), 'quality': (5,)}.items():
                if z[key].shape != shape or not np.isfinite(z[key]).all():
                    return False
            return z['transcript'].shape == () and z['transcript'].dtype.kind == 'U' and int(z['frames']) > 0
    except (ValueError, OSError, EOFError, KeyError, TypeError, zipfile.BadZipFile):
        return False


def preprocess(job, tmp_root):
    i, ident, name, info = job
    started = time.monotonic()
    path = tmp_root / f'{i:05d}.{os.getpid()}.{uuid.uuid4().hex}.mp4'
    try:
        raw = decode_member(name, info)
        digest = hashlib.sha256(raw).hexdigest()
        if path.resolve().parent != tmp_root.resolve():
            raise ValueError('Temporary media escapes designated directory')
        # Unique, exclusively created file: never overwrite preexisting media.
        with path.open('xb') as f:
            f.write(raw)
        size = len(raw)
        del raw
        chunks, detection, frames = mouth_chunks(path)
        return dict(ok=True, i=i, id=ident, path=path, chunks=chunks, detection=detection,
                    frames=frames, media_sha256=digest, bytes=size,
                    preprocess_seconds=time.monotonic() - started)
    except Exception as exc:
        # Retain newly downloaded media on failure, allowing diagnosis.
        return dict(ok=False, i=i, id=ident, stage='fetch_or_preprocess',
                    error=f'{type(exc).__name__}: {str(exc)[:400]}', retained_tmp=str(path) if path.exists() else None)


def infer(chunks, detection, nframes, model, tokens):
    embeddings, entropy, blank, texts, present = [], [], [], [], []
    for gray, ok in chunks:
        if not ok:
            embeddings.append(np.zeros((len(gray), 768), np.float32))
            present.extend([False] * len(gray))
            continue
        x = torch.as_tensor(gray.copy(), device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        x = (x / 255 - .421) / .165
        with torch.inference_mode():
            enc, _ = model.encoder(model.proj_encoder(model.frontend(x)), None)
            lp = model.ctc.log_softmax(enc)[0]
            p = lp.exp()
            entropy.append((-(p * lp).sum(-1) / np.log(len(tokens))).cpu().numpy())
            blank.append(p[:, 0].cpu().numpy())
            texts.append(text_from_ids(lp.argmax(-1).cpu().numpy(), tokens))
            embeddings.append(enc[0].cpu().numpy())
        present.extend([True] * len(gray))
    h = np.concatenate(embeddings)
    present = np.asarray(present, dtype=bool)
    if h.shape != (nframes, 768) or len(present) != nframes:
        raise ValueError('Encoder sequence length does not match 25 fps input')
    usable = h[present]
    pooled = np.concatenate((usable.mean(0), usable.std(0))) if len(usable) else np.zeros(1536, np.float32)
    temporal = np.stack([a.mean(0) if len(a) else np.zeros(768) for a in np.array_split(h, 16)]).astype(np.float32)
    quality = np.asarray([detection, present.mean(), np.concatenate(entropy).mean() if entropy else 1,
                          np.concatenate(blank).mean() if blank else 1, nframes / 25], np.float32)
    return dict(pooled=pooled.astype(np.float32), temporal=temporal, quality=quality,
                transcript=' '.join(texts), frames=nframes)


def main(args):
    if args.workers < 1 or args.start < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError('workers/limit must be positive; start must be nonnegative')
    os.chdir(RUN)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    out = ROOT / 'mosei_vsr'
    parts, tmp_root = out / 'parts', out / 'tmp'
    parts.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    index_path = RUN / 'rgb_mosei/zip_index.json'
    rows = json.loads(index_path.read_text(encoding='utf-8'))
    mapping = {}
    for name, info in rows.items():
        pieces = name.replace('\\', '/').split('/')
        ident = pieces[-2] + '$_$' + Path(pieces[-1]).stem
        if ident in mapping:
            raise ValueError(f'Duplicate ZIP identifier {ident}')
        mapping[ident] = (name, info)
    with np.load(AUTHOR_IDS, allow_pickle=False) as z:
        # Open only ID arrays; labels and all modality arrays are never loaded.
        split_sizes = {s: len(z[f'{s}_ids']) for s in ('train', 'valid', 'test')}
        ids = np.concatenate([z[f'{s}_ids'] for s in split_sizes]).astype(str)
    if len(ids) != 22856 or len(set(ids)) != len(ids):
        raise ValueError('Expected 22856 unique official MOSEI IDs')
    missing = [ident for ident in ids if ident not in mapping]
    if missing or args.start >= len(ids):
        raise ValueError(f'Missing ZIP IDs {missing[:10]} or invalid start')
    weights = ROOT / 'vsr_trlrs2lrs3vox2avsp_base.pth'
    weights_hash = sha256(weights)
    verified = json.loads((ROOT / 'weights_verified.json').read_text(encoding='utf-8'))
    if weights_hash != verified['sha256'] or weights.stat().st_size != verified['bytes']:
        raise ValueError('Weights differ from verified official checkpoint')
    protocol = dict(schema=1, source=URL, zip_bytes=ZIP_BYTES, index_sha256=sha256(index_path),
        weights_sha256=weights_hash, weights_bytes=weights.stat().st_size,
        code_sha256=sha256(ROOT / 'extract_mosei_visual_speech.py'), mouth_code_sha256=sha256(ROOT / 'extract_visual_speech.py'),
        yunet_sha256=sha256(RUN / 'face_detection_yunet_2023mar.onnx'),
        upstream_preprocess_sha256=sha256(CODE / 'preparation/detectors/mediapipe/video_process.py'),
        units_sha256=sha256(CODE / 'spm/unigram/unigram5000_units.txt'),
        ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(), split_sizes=split_sizes,
        input='current-utterance RGB only; no audio decoding, dataset transcript, labels, neighbors',
        source_model='official Auto-AVSR 3291-hour VSR; external pretraining',
        preprocessing='25fps YuNet largest face, 4-point mouth approximation, gray96 center88, .421/.165',
        chunk_frames=300, chunk_overlap=0, decoding='CTC greedy, repeated tokens collapse before blank removal',
        aggregation='MOSI-compatible mean/std pooled1536; 16 equal-time encoder summaries 16x768',
        quality_order=['face_detection_fraction', 'usable_chunk_frame_fraction', 'normalized_ctc_entropy', 'ctc_blank_probability', 'duration_seconds'],
        no_face='zero visual vectors, empty transcript, quality reflects unavailable chunks; retained as sample',
        cleanup='Only unique MP4 freshly created under mosei_vsr/tmp, after verified feature output')
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol['fingerprint'] = fingerprint
    if (out / 'protocol.json').exists():
        prior = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
        if prior.get('fingerprint') != fingerprint:
            raise ValueError('Existing output protocol differs; preserve it and choose a new output version')
    atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / f'scheduler_{os.getpid()}.json', dict(scheduler='completion order with final ID-ordered assembly', scheduler_source_sha256=sha256(Path(__file__)), algorithm_source_sha256=protocol['code_sha256'], fingerprint=fingerprint, note='Only future scheduling differs; original cache protocol remains exact. Download/CRC/mouth/inference/aggregation functions are AST-identical.'))
    indices = list(range(args.start, len(ids)))
    if args.limit is not None:
        indices = indices[:args.limit]
    todo = [i for i in indices if not cached_ok(parts / f'{i:05d}.npz', ids[i], fingerprint)]
    tokens = ['<blank>'] + [s.split()[0] for s in (CODE / 'spm/unigram/unigram5000_units.txt').read_text(encoding='utf-8').splitlines()] + ['<eos>']
    model = None
    if todo:
        model = E2E(len(tokens), 'video', ctc_weight=.1)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        model.cuda().eval()
    tag = f'{args.start}_{os.getpid()}_{time.time_ns()}'
    status_path = out / f'status_{tag}.json'
    started, succeeded, failed = time.time(), len(indices) - len(todo), 0
    atomic_json(status_path, dict(stage='starting', selected=len(indices), resumed=succeeded, start=args.start, limit=args.limit))
    with (out / f'events_{tag}.jsonl').open('a', encoding='utf-8', buffering=1) as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        depth = args.workers * 2
        def submit(i):
            name, info = mapping[ids[i]]
            return pool.submit(preprocess, (i, ids[i], name, info), tmp_root)
        pending = {submit(i): i for i in todo[:depth]}
        next_slot = depth
        finished = 0
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                i = pending.pop(future)
                item = future.result()
                if next_slot < len(todo):
                    j = todo[next_slot]
                    next_slot += 1
                    pending[submit(j)] = j
                finished += 1
                if item['ok']:
                    try:
                        record = infer(item.pop('chunks'), item['detection'], item['frames'], model, tokens)
                        path = parts / f'{i:05d}.npz'
                        temp = parts / f'{i:05d}.{tag}.tmp.npz'
                        np.savez_compressed(temp, id=ids[i], fingerprint=fingerprint, media_sha256=item['media_sha256'], **record)
                        if not cached_ok(temp, ids[i], fingerprint):
                            raise ValueError('New feature cache failed schema/finiteness check')
                        if path.exists():
                            os.replace(path, parts / f'{i:05d}.invalid_{time.time_ns()}.npz')
                        os.replace(temp, path)
                        if not cached_ok(path, ids[i], fingerprint):
                            raise ValueError('Committed feature cache failed validation')
                        media = item.pop('path')
                        if media.resolve().parent != tmp_root.resolve() or not media.name.startswith(f'{i:05d}.{os.getpid()}.'):
                            raise ValueError('Refusing cleanup outside this run temporary file')
                        media.unlink()
                        item['no_detected_faces'] = bool(item['detection'] == 0)
                        succeeded += 1
                    except Exception as exc:
                        item.pop('chunks', None)
                        item['ok'] = False
                        item['stage'] = 'inference_or_commit'
                        item['error'] = f'{type(exc).__name__}: {str(exc)[:400]}'
                        if 'path' in item:
                            item['retained_tmp'] = str(item.pop('path'))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                if not item['ok']:
                    failed += 1
                log.write(json.dumps(item, ensure_ascii=False) + '\n')
                status = dict(stage='extracting', succeeded=succeeded, failed=failed, selected=len(indices),
                              official_total=len(ids), elapsed=time.time() - started)
                atomic_json(status_path, status)
                if finished % 10 == 0 or not item['ok']:
                    print(status, flush=True)
    # A sharded/resumed run can assemble only when every official ID is valid.
    incomplete = [i for i, ident in enumerate(ids) if not cached_ok(parts / f'{i:05d}.npz', ident, fingerprint)]
    if incomplete:
        atomic_json(status_path, dict(stage='partial_or_failed', succeeded=succeeded, failed=failed,
            official_total=len(ids), missing_count=len(incomplete), missing_indices=incomplete,
            elapsed=time.time() - started))
        if failed:
            raise RuntimeError(f'{failed} selected clips failed; inspect events_{tag}.jsonl and rerun to resume')
        print(f'Partial extraction complete; {len(incomplete)} of {len(ids)} official IDs still missing', flush=True)
        return
    blocks = {k: [] for k in ('pooled', 'temporal', 'quality', 'transcript', 'frames')}
    for i, ident in enumerate(ids):
        with np.load(parts / f'{i:05d}.npz', allow_pickle=False) as z:
            if str(z['id']) != ident:
                raise ValueError('ID changed during final assembly')
            for key in blocks:
                blocks[key].append(z[key])
    arrays = {key: np.stack(value) for key, value in blocks.items()}
    temp = out / f'features.{tag}.tmp.npz'
    np.savez_compressed(temp, ids=ids, fingerprint=fingerprint, **arrays)
    with np.load(temp, allow_pickle=False) as z:
        if not np.array_equal(z['ids'], ids) or z['temporal'].shape != (22856, 16, 768):
            raise ValueError('Final feature archive verification failed')
    os.replace(temp, out / 'features.npz')
    atomic_json(status_path, dict(stage='completed', total=len(ids), failed_this_invocation=failed, elapsed=time.time() - started))
    if failed:
        raise RuntimeError('Features assembled, but cleanup failures occurred; inspect invocation events')
    print('COMPLETED all 22856 MOSEI visual speech features', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--workers', type=int, default=6)
    main(parser.parse_args())
