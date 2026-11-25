import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from evidence_msa.text_present.model import TextPresent,objective
from evidence_msa.text_present.raw_adapters import RawEvidenceAdapter
from evidence_msa.text_present import calibration as cal
from evidence_msa.text_present.protocol import utility
CFG=dict(shared=True,attention=True,distribution=True,polarity=True)
DIMS=dict(T=8,A=9,V=17,app=9,timeline=3,Q=2)
def batch():
    b={k:torch.randn(6,d) for k,d in [('T',8),('A',9),('AS',9),('V',17),('VS',17),('Q',2)]}
    b['timeline']=torch.randn(6,4,3);return b
class Tests(unittest.TestCase):
    def test_old_conditions_preserved(self):
        path=P.parent/'cloud_ablation_20261123/src/evidence_msa/text_present/model.py'
        spec=importlib.util.spec_from_file_location('original_core',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        x=old.TextPresent(DIMS,CFG,128).eval();m=TextPresent(DIMS,CFG,128).eval();m.load_state_dict(x.state_dict())
        # Nonzero correction avoids a vacuous equality at the text-only initialization.
        with torch.no_grad():x.correct[-1].weight.normal_(std=.03)
        m.load_state_dict(x.state_dict());b=batch()
        for c in ('T','TA','TV'):
            torch.testing.assert_close(x(b,c)['pred'],m(b,c)['pred'],rtol=0,atol=0)
    def test_tav_uses_both_sources_and_five_tokens(self):
        m=TextPresent(DIMS,CFG,128).eval()
        with torch.no_grad():m.correct[-1].weight.normal_(std=.03)
        b={k:v.requires_grad_() for k,v in batch().items()};tokens=[]
        hook=m.interaction.register_forward_pre_hook(lambda mod,args:tokens.append(args[1].shape[1]))
        objective(m(b,'TAV'),torch.linspace(-2,2,6),CFG).backward();hook.remove()
        self.assertEqual(tokens,[5])
        for k in ('T','A','V','timeline','Q'):
            self.assertGreater(b[k].grad.abs().sum().item(),0,k)
        for c,hidden in [('T',('A','AS','V','VS','timeline','Q')),('TA',('V','VS','timeline','Q')),('TV',('A','AS'))]:
            alt={k:v.detach().clone() for k,v in b.items()}
            for k in hidden:alt[k].normal_()
            torch.testing.assert_close(m(b,c)['pred'],m(alt,c)['pred'],rtol=0,atol=0)
    def test_full_adapter_zero_identity_and_frozen_core(self):
        core=TextPresent(DIMS,CFG,128).eval();m=RawEvidenceAdapter(dict(model=core.state_dict(),dims=DIMS,config=CFG,hidden=128),'D5_bilinear').eval()
        b=batch()
        for c in cal.CS:torch.testing.assert_close(core(b,c)['pred'],m(b,c)['pred'],rtol=0,atol=0)
        before={k:v.clone() for k,v in m.core.state_dict().items()}
        m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        objective(m(b,'TAV'),torch.linspace(-2,2,6),CFG).backward();opt.step()
        for k,v in before.items():self.assertTrue(torch.equal(v,m.core.state_dict()[k]))
        with torch.no_grad():m.heads['TAV'].output[-1].weight.normal_(std=.03)
        m.eval();b={k:v.requires_grad_() for k,v in batch().items()};m(b,'TAV')['pred'].sum().backward()
        for k in ('A','V'):self.assertGreater(b[k].grad.abs().sum().item(),0)
    def test_calibration_width_and_primary_selection(self):
        n=10;p=np.linspace(-2.8,2.8,n)
        z={**{'pred_'+c:p.copy() for c in cal.CS},**{'f_'+c:np.zeros((n,37),np.float32) for c in cal.CS}}
        base={c:p.copy() for c in cal.CS};x,*_=cal.features(z,base,'TAV');self.assertEqual(x.shape,(n,53))
        self.assertEqual(cal.Intensity()(torch.as_tensor(x)).shape,(n,))
        one=utility(base,p)
        changed={c:(-p if c!='TAV' else p) for c in cal.CS};self.assertEqual(one,utility(changed,p))
        self.assertGreater(one,utility({c:-p for c in cal.CS},p))
        cal.selftest()
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
