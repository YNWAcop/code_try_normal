"""One-time explicit TAV extension of a copied frozen C5+ERU source tree."""
from pathlib import Path
import hashlib,json,difflib
P=Path(__file__).resolve().parent
BASE=P.parent/'cloud_ablation_20261123'
def patch(name,old,new):
    p=P/'src/evidence_msa/text_present'/name
    s=p.read_text(encoding='utf-8');assert old in s,(name,old[:60]);p.write_text(s.replace(old,new),encoding='utf-8')
patch('model.py',"('T', 'TA', 'TV')", "('T', 'TA', 'TV', 'TAV')")
patch('model.py',"residual = self.audio(b['A']-b['AS']) if condition == 'TA' else self.visual(b)",
"""residuals = []
            if 'A' in condition: residuals.append(self.audio(b['A']-b['AS']))
            if 'V' in condition: residuals.append(self.visual(b))
            residual = torch.stack(residuals).mean(0) if len(residuals)>1 else residuals[0]""")
patch('model.py',"shared = self.shared_a(b['AS']) if condition == 'TA' else self.shared_v(b['VS'])",
"""shared_parts = []
                if 'A' in condition: shared_parts.append(self.shared_a(b['AS']))
                if 'V' in condition: shared_parts.append(self.shared_v(b['VS']))
                shared = torch.stack(shared_parts).mean(0) if len(shared_parts)>1 else shared_parts[0]""")
patch('model.py',"tokens = [t, residual] + ([shared] if self.config['shared'] else [])",
      "tokens = [t] + residuals + (shared_parts if self.config['shared'] else [])")
for name in ('adapters.py','raw_adapters.py','protocol.py'):
    patch(name,"('T','TA','TV')","('T','TA','TV','TAV')")
patch('raw_adapters.py',"if c!='T':self.modality=projection(dims['A'] if c=='TA' else dims['V']+dims['Q']+dims['timeline'],h)",
"""if c!='T':
            width_in = dims['A'] if c=='TA' else dims['V']+dims['Q']+dims['timeline']
            if c=='TAV': width_in += dims['A']
            self.modality=projection(width_in,h)""")
patch('raw_adapters.py',"m=self.modality(x);features.append(m)","if self.c=='TAV':x=torch.cat((b['A']-b['AS'],x),-1)\n            m=self.modality(x);features.append(m)")
patch('protocol.py',"    for c in CS:\n        m=scores", "    for c in ('TAV',):\n        m=scores")
patch('protocol.py',"POLICY.update(ablation_package='cloud_ablation_20261123',input_mode=INPUT_MODE,backbone=BACKBONE,synthetic_smoke=SMOKE,device=DEVICE)",
"""POLICY.update(experiment='local C5+complete ERU TAV extension v1',
    package='local_ours_tav_20261124',input_mode=INPUT_MODE,backbone=BACKBONE,synthetic_smoke=SMOKE,device=DEVICE,
    own_checkpoint_selection='TAV only: mean of six scaled validation metrics, earliest tie',
    report='TAV primary; T/TA/TV auxiliary diagnostics from this new run, never replacements for old six-condition results',
    tav_core='text query over [T,A_res,V_res,A_shared,V_shared]; arithmetic mean residual/shared slots preserve head width',
    tav_eru='raw evidence head projects concatenated A_res,V_res,Q,mean_timeline; all ERU stages retained',
    training_conditions=list(CS),primary_condition='TAV',loss_average='four visible conditions equally weighted')""")
patch('train.py',"loss=objective(model(b,c),y,CONFIG)/3","loss=objective(model(b,c),y,CONFIG)/len(CS)")
patch('calibration.py',"CS = ('T', 'TA', 'TV')","CS = ('T', 'TA', 'TV', 'TAV')")
patch('calibration.py',"('TA', 'TV')","CS[1:]")
patch('calibration.py',"def __init__(self, d=52):","def __init__(self, d=37+9+len(CS)+3):")
patch('calibration.py',"np.eye(3)[CS.index(c)]","np.eye(len(CS))[CS.index(c)]")
patch('calibration.py',"y[ix], g)/2","y[ix], g)/len(CS[1:])")
patch('calibration.py',"np.tile(train_z['y'], 3)","np.tile(train_z['y'], len(CS))")
patch('calibration.py',"np.zeros(52, dtype='float32')","np.zeros(37+9+len(CS)+3, dtype='float32')")
patch('calibration.py',"np.ones(52, dtype='float32')","np.ones(37+9+len(CS)+3, dtype='float32')")
patch('runner.py',"'V_words','V_quality')}[c]","'V_words','V_quality'),'TAV':('T','A_native','A_words','V_native','V_expression','V_lip','V_words','V_quality')}[c]")
patch('runner.py',"assert sha(dest/'results.json')==read(dest/'done.json')['results_sha'];reports.extend(read(dest/'results.json'));continue",
"""done=read(dest/'done.json')
            for name,key in [('results.json','results_sha'),('predictions_test.npz','predictions_sha'),('predictions_c5_test.npz','c5_predictions_sha')]:
                assert sha(dest/name)==done[key]
            assert sha(ROOT/'runs'/ds/'fold_0/selection_lock.json')==done['selection_lock_sha']
            assert sha(cv_data.SOURCE/ds/'test.npz')==done['test_sha']
            reports.extend(read(dest/'results.json'));continue""")
# Save an exact reviewable diff; do not mutate any original package.
changes=[];original={}
for p in sorted((P/'src').rglob('*.py')):
    rel=p.relative_to(P);old=BASE/rel
    if old.exists():
        original[rel.as_posix()]=hashlib.sha256(old.read_bytes()).hexdigest()
        changes.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile='original/'+rel.as_posix(),tofile='tav/'+rel.as_posix()))
(P/'provenance/original_hashes.json').write_text(json.dumps(original,indent=2),encoding='utf-8')
(P/'provenance/TAV_changes.patch').write_text(''.join(changes),encoding='utf-8')
print('TAV source extension complete; original files untouched')
