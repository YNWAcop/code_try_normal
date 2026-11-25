import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evidence_msa.input_views import select_inputs,ENHANCED
from evidence_msa.text_present.model import TextPresent,objective
from evidence_msa.text_present.simple_fusion import SimpleFusion
CFG=dict(shared=True,attention=True,distribution=True,polarity=True)
DIMS=dict(T=8,A=9,V=17,app=9,timeline=3,Q=2)
class Contracts(unittest.TestCase):
    def test_native_removes_all_extra_channels_before_transforms(self):
        d={k:np.random.randn(3,5).astype('float32') for k in ENHANCED+('T','A_native','V_native')}
        n=select_inputs(d,'native');e=select_inputs(d,'enhanced')
        for k in ENHANCED:self.assertTrue(np.all(n[k]==0));self.assertTrue(np.array_equal(e[k],d[k]))
        for k in ('T','A_native','V_native'):self.assertTrue(np.array_equal(n[k],d[k]))
    def test_core_and_simple_missing_modality_isolation_and_backward(self):
        torch.set_num_threads(2)
        for cls in (TextPresent,SimpleFusion):
            m=cls(DIMS,CFG,128).eval()
            b={k:torch.randn(3,d) for k,d in [('T',8),('A',9),('AS',9),('V',17),('VS',17),('Q',2)]};b['timeline']=torch.randn(3,4,3)
            for c in ('T','TA','TV'):
                hidden=('A','AS','V','VS','Q','timeline') if c=='T' else (('V','VS','Q','timeline') if c=='TA' else ('A','AS'))
                altered={k:v.clone() for k,v in b.items()}
                for k in hidden:altered[k].normal_()
                self.assertTrue(torch.equal(m(b,c)['pred'],m(altered,c)['pred']))
                loss=objective(m(b,c),torch.tensor([-1.,0.,1.]),CFG);self.assertTrue(torch.isfinite(loss));m.zero_grad();loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            if cls is SimpleFusion:
                self.assertFalse(hasattr(m,'shared_a'));self.assertFalse(hasattr(m,'interaction'));self.assertFalse(hasattr(m,'correct'))
                a=m.parameter_match;self.assertLess(abs(a['c5_parameters']-a['simple_parameters'])/a['c5_parameters'],.005)
    def test_native_policy_and_no_seed_calls(self):
        import ast
        for path in (ROOT/'src').rglob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,('manual_seed','manual_seed_all','seed'),str(path))
if __name__=='__main__':unittest.main()
