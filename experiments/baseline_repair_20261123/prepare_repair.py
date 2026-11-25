"""Isolate the actual trained EMT source; preserve old code and result files."""
from pathlib import Path
import hashlib
import json
import shutil
import difflib

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'whole_priority_20261117'
ROOT.mkdir(exist_ok=True)
for name in ('adapter_emt_tfr.py', 'mmsa_corruption.py', 'mmsa_slim_pickle.py',
             'full_four.py', 'metrics_common.py'):
    shutil.copy2(OLD/name, ROOT/name)
shutil.copytree(OLD/'vendor/emt_dlfr', ROOT/'vendor/emt_dlfr', dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__'))
# Never regenerate from an ambient, possibly different desktop checkout.
(ROOT/'emt_vendor_prepare.py').write_text(
    'from pathlib import Path\ndef prepare(force=False):\n'
    '    if force: raise RuntimeError("Repair source is frozen")\n'
    '    return Path(__file__).resolve().parent / "vendor/emt_dlfr"\n', encoding='utf-8')
path = ROOT/'vendor/emt_dlfr/models/missingTask/EMT_DLFR.py'
old = path.read_text(encoding='utf-8')
new = old.replace(
    "missing_text_lengths = text_m[:, 1, :].sum(1).int() - 2\n"
    "        missing_audio_lengths = audio_lengths if 'A' in self.protocol_condition else torch.ones_like(audio_lengths)\n"
    "        missing_video_lengths = video_lengths if 'V' in self.protocol_condition else torch.ones_like(video_lengths)",
    "# Whole-absent streams retain a PUBLIC fixed reconstruction grid.\n"
    "        # Its length is the dataset tensor width, never a hidden sample length.\n"
    "        # All supplied missing inputs remain constant empty text / zero A,V.\n"
    "        missing_text_lengths = (text_m[:, 1, :].sum(1).int() - 2 if 'T' in self.protocol_condition\n"
    "                                else torch.full_like(text_lengths, text_m.shape[2] - 1))\n"
    "        missing_audio_lengths = (audio_lengths if 'A' in self.protocol_condition\n"
    "                                 else torch.full_like(audio_lengths, audio_m.shape[1]))\n"
    "        missing_video_lengths = (video_lengths if 'V' in self.protocol_condition\n"
    "                                 else torch.full_like(video_lengths, video_m.shape[1]))")
assert new != old, 'Expected historical source patch not found'
path.write_text(new, encoding='utf-8')
(ROOT/'emt_fixed_grid.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), fromfile='historical/EMT_DLFR.py',
    tofile='repair/EMT_DLFR.py')), encoding='utf-8')
manifest = {
    'source': str(OLD), 'scope': 'EMT_DLFR only; TFR source not altered',
    'reason': 'Historical missing A/V length1 truncated low-level reconstruction supervision.',
    'repair': 'Public dataset-width grid for fully missing streams, including text; original EMT, SimSiam and reconstruction objectives/weights retained.',
    'claim': 'Whole-modality protocol adaptation, not exact original random-feature-missing experiment.',
    'training': 'One unseeded run per dataset/condition; train/valid only; original optimizer, early stop and validation selection; no test-driven retries.',
    'physical_batch': 'Resource-tested microbatch; gradient accumulation does not reproduce native BatchNorm statistics.',
    'old_source_sha256': hashlib.sha256(old.encode()).hexdigest(),
    'repaired_source_sha256': hashlib.sha256(new.encode()).hexdigest(),
    'results_policy': 'Old table immutable until replacement predictions, checkpoint replay, sample identities and metrics pass.'}
(ROOT/'repair_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
