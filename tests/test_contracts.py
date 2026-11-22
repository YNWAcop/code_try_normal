from pathlib import Path
import ast
import importlib
import pkgutil
import sys
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
torch.set_num_threads(2)

class Contracts(unittest.TestCase):
    def test_runtime_imports(self):
        import evidence_msa
        for item in pkgutil.walk_packages(evidence_msa.__path__,evidence_msa.__name__+'.'):
            importlib.import_module(item.name)

    def test_metric_definitions(self):
        from evidence_msa.common.metrics import scores
        y=np.array([-2.5,-.5,0,.5,2.5]);m=scores(y,y)['metrics']
        for name in ('Acc7','Acc5','Acc2_nonzero','F1_weighted_nonzero','Corr'):
            self.assertAlmostEqual(m[name],1)
        self.assertEqual(m['MAE'],0)
        self.assertIsNone(scores(y,np.zeros_like(y))['metrics']['Corr'])
        with self.assertRaises(ValueError):scores(y,np.ones(3))

    def test_interval_boundaries(self):
        from evidence_msa.text_present.calibration import signature,cells,warp
        cuts=np.array([-2.5,-1.5,-.5,0,.5,1.5,2.5])
        p=np.concatenate([np.linspace(-3,3,1001),cuts,np.nextafter(cuts,-np.inf),np.nextafter(cuts,np.inf)])
        _,lo,hi=cells(p)
        self.assertTrue(np.array_equal(signature(p),signature(lo)))
        self.assertTrue(np.array_equal(signature(p),signature(hi)))
        for a,b in ((0,0),(1,2),(-1,-2)):
            self.assertTrue(np.array_equal(signature(p),signature(warp(p,np.tile([a,b],(9,1))))))

    def test_repair_scope(self):
        from evidence_msa.text_present.runner import repair
        p=np.array([-.09,.09,-.2,.2,0.]);q=np.array([.9,.1,.9,.1,.9])
        r=repair(p,q)
        self.assertTrue(r[0]>0 and r[1]<0)
        np.testing.assert_array_equal(r[2:4],p[2:4])
        for clip in (2,3):np.testing.assert_array_equal(np.rint(p.clip(-clip,clip)),np.rint(r.clip(-clip,clip)))

    def test_c5_missing_inputs(self):
        from evidence_msa.text_present.model import TextPresent
        dims=dict(T=8,A=6,V=10,app=4,Q=5,timeline=7)
        m=TextPresent(dims,dict(shared=True,attention=True,distribution=True,polarity=True),h=16).eval()
        b={k:torch.randn(3,dims[k]) for k in ('T','A','V','Q')}
        b.update(AS=torch.randn(3,6),VS=torch.randn(3,10),timeline=torch.randn(3,4,7))
        with torch.no_grad():
            for c,unused in [('T',('A','AS','V','VS','Q','timeline')),('TA',('V','VS','Q','timeline')),('TV',('A','AS'))]:
                alt={k:v.clone() for k,v in b.items()}
                for k in unused:alt[k].fill_(123)
                torch.testing.assert_close(m(b,c)['pred'],m(alt,c)['pred'],rtol=0,atol=0)

    def test_v1_missing_inputs(self):
        from evidence_msa.text_absent.v1 import Evidence
        m=Evidence(dict(T=8,A=6,V=10),4).eval()
        b={k:torch.randn(3,n) for k,n in dict(T=8,A=6,V=10,A_res=6,V_res=10,quality=5).items()}
        b.update(anchor=torch.randn(3),timeline=torch.randn(3,4,768))
        with torch.no_grad():
            for c in ('A','V','AV'):
                bits=torch.tensor([[m in c for m in 'TAV']]*3)
                alt={k:v.clone() for k,v in b.items()}
                unused=['T','anchor','A_res','V_res']
                if 'A' not in c:unused+=['A']
                if 'V' not in c:unused+=['V','quality','timeline']
                for k in unused:alt[k].fill_(123)
                torch.testing.assert_close(m(b,bits),m(alt,bits),rtol=0,atol=0)

    def test_algorithm_source_preserved(self):
        pairs=[('final/train_ours.py','text_present/train.py'),('final/runner.py','text_present/runner.py'),
               ('final/calibration.py','text_present/calibration.py'),('c5/model.py','text_present/model.py'),
               ('v1/train_ours_fixed.py','text_absent/v1.py')]
        class StripImports(ast.NodeTransformer):
            def visit_Import(self,n):return None
            def visit_ImportFrom(self,n):return None
        def definitions(p):
            tree=StripImports().visit(ast.parse(p.read_text(encoding='utf-8-sig')))
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for old,new in pairs:
            self.assertEqual(definitions(ROOT/'archive/frozen_sources'/old),definitions(ROOT/'src/evidence_msa'/new),new)

    def test_original_cell_rule_equivalence(self):
        from evidence_msa.text_present.calibration import cells,signature
        tree=ast.parse((ROOT/'archive/frozen_sources/refinement/cell_warp.py').read_text(encoding='utf-8-sig'))
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('cells','signature')]
        space={'np':np};exec(compile(ast.Module(body=funcs,type_ignores=[]),'original','exec'),space)
        p=np.concatenate([np.linspace(-3,3,6001),np.array([-2.5,-1.5,-.5,0,.5,1.5,2.5])])
        np.testing.assert_array_equal(signature(p),space['signature'](p))
        for a,b in zip(cells(p),space['cells'](p)):np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
