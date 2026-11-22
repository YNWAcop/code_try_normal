from pathlib import Path
import sys,threading,time,struct,zlib
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import inflate64
URL='https://hf-mirror.com/datasets/tamb2203579/CMU-MOSEI/resolve/main/Raw.zip'
ZIP_BYTES=27280713458
NETWORK=threading.local()
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
