import argparse,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
os.environ.update(MSA_RUN_ROOT=str(a.run),MSA_DATA_ROOT=str(a.data),MSA_SMOKE='1',MSA_DEVICE='cpu')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evidence_msa.inference import features,predict_text,canonical
assert canonical('TAV')=='TAV'
for ds in ('mosi','mosei'):
    d=features(a.data/'ready'/ds/'test.npz');assert 'y' not in d
    pred=predict_text(d,a.run/'runs'/ds/'fold_0')
    with np.load(a.run/'evaluation'/ds/'predictions_test.npz') as z:np.testing.assert_allclose(pred['TAV'],z['pred_TAV'],rtol=0,atol=1e-12)
print('Unified label-free TAV inference matches evaluated predictions for both datasets')
