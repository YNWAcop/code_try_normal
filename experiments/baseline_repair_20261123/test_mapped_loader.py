import json,pickle,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from mmsa_mapped_pickle import load_fields
fields=['audio','vision','text_bert','audio_lengths','vision_lengths','id','regression_labels']
# Compare real MOSI against historical loader (before replacing its entry point).
sys.path.insert(0,str(ROOT.parent/'whole_priority_20261117'))
from mmsa_slim_pickle import load_fields as old_load
path=Path('D:/MSA_Cloud_20261115/data/mmsa/mosi/unaligned_50.pkl')
old=old_load(path,fields,splits=('train','valid'));new=load_fields(path,fields,splits=('train','valid'))
checks=[]
for split in old:
    for key in fields:
        a,b=old[split][key],new[split][key]
        assert np.array_equal(a,b),(split,key)
        if key in ('audio','vision'):
            # Adapter per-sample conversion equals previous whole-array conversion.
            a32=np.asarray(a,dtype=np.float32)
            for start in range(0,len(b),64):assert np.array_equal(a32[start:start+64],np.asarray(b[start:start+64],dtype=np.float32))
        checks.append(split+'/'+key)
with tempfile.TemporaryDirectory() as folder:
    for protocol in (3,4):
        blob={'train':{'x':np.arange(500000,dtype=np.float64).reshape(1000,500,order='F')}}
        file=Path(folder)/f'test{protocol}.pkl';file.write_bytes(pickle.dumps(blob,protocol=protocol))
        mapped=load_fields(file,['x'],splits=('train',))['train']['x']
        assert np.array_equal(mapped,blob['train']['x'])
        assert mapped.dtype==blob['train']['x'].dtype
        del mapped
result={'passed':True,'exact_real_MOSI_fields':checks,'synthetic_protocols':[3,4],'cast_float32_equal':True,'test_data_used':False}
(ROOT/'mapped_loader_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
