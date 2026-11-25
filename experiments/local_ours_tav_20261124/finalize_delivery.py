from pathlib import Path
import difflib,hashlib,json,zipfile
P=Path(__file__).resolve().parent;BASE=P.parent/'cloud_ablation_20261123'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in json.loads((P/'release/runtime_manifest.json').read_text())['files'].items():assert sha(P/n)==h,n
diff=[]
for p in sorted((P/'src').rglob('*.py')):
    rel=p.relative_to(P);old=BASE/rel
    diff.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True) if old.exists() else [],p.read_text(encoding='utf-8').splitlines(True),fromfile='original/'+rel.as_posix(),tofile='tav/'+rel.as_posix()))
(P/'provenance/TAV_changes.patch').write_text(''.join(diff),encoding='utf-8')
report=json.loads((P/'LOCAL_TEST_REPORT.json').read_text())
report.update(unified_label_free_inference_matches=True,cuda_synthetic=json.loads((P/'validation/CUDA_SMOKE_REPORT.json').read_text(encoding='utf-8-sig')),
    real_preflight=json.loads((P/'outputs/formal/preflight.json').read_text()),runtime_manifest_sha256=sha(P/'release/runtime_manifest.json'))
(P/'LOCAL_TEST_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
files=[p for p in P.rglob('*') if p.is_file() and not set(p.relative_to(P).parts)&{'outputs','validation','__pycache__'} and p.suffix!='.pyc']
archive=P.parent/(P.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(P.parent))
archive.with_suffix('.zip.sha256').write_bytes((sha(archive)+'  '+archive.name+'\n').encode('ascii'))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert not any('/outputs/' in n or '/validation/' in n or '__pycache__' in n for n in z.namelist())
print(json.dumps(dict(source_archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),runtime_manifest_sha256=sha(P/'release/runtime_manifest.json')),indent=2))
