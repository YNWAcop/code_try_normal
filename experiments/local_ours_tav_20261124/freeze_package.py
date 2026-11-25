"""Freeze runtime sources before starting any real experiment."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=list((P/'src').rglob('*.py'))+[P/'local_tav.py',P/'Run-Ours-TAV.ps1',P/'expected_data.json',P/'requirements.txt']
files={p.relative_to(P).as_posix():sha(p) for p in sorted(paths)}
(P/'release/runtime_manifest.json').write_text(json.dumps(dict(version='TAV-extension-v1',files=files),indent=2),encoding='utf-8')
print('Frozen',len(files),'runtime sources')
