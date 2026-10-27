"""CPU synthetic integration through the real pilot runner structure."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
from experiments.utility_feasibility_v1.test_slice_api import CPUBackend
from experiments.utility_feasibility_v1.slice_api import Geometry,FrozenConfig
from experiments.route_b_preflight_v1.runner import pilot_predict,evaluate_saved,RunWriter

class Backend(CPUBackend):
    def __init__(self):super().__init__();self.support_cache={};self.support_encoder_calls=0;self.generations=0
    def prepare_support(self,s,m):
        key=s.tobytes()+m.tobytes()
        if key not in self.support_cache:
            mm=m[0].astype(bool);self.support_cache[key]=np.stack([mm,~mm]).astype(float);self.support_encoder_calls+=1
        self.support_features=self.support_cache[key]
    def generate(self,q,s,m):self.generations+=1;return super().generate(q,s,m)
    def encode(self,q):super().encode(q);self.query_features=self.support_features.copy()

class IntegrationTests(unittest.TestCase):
    def inputs(self):
        m=np.zeros((1,257,257),np.float32);m[:,80:180,80:180]=1
        return dict(query_image=np.ones((1,3,257,257),np.float32),support_image=np.ones((1,3,257,257),np.float32),
                    support_mask=m,geometry=Geometry(),configuration=FrozenConfig())
    def test_gt_is_outside_prediction(self):
        legitimate=self.inputs();outputs=[]
        for gt in (None,np.zeros((257,257),bool),np.ones((257,257),bool)):
            external={**legitimate,'query_gt':gt};args={k:external[k] for k in legitimate}
            a,r=pilot_predict(**args,backend=Backend());outputs.append((a,r))
        for a,r in outputs[1:]:
            self.assertEqual(r,outputs[0][1])
            for k in a:np.testing.assert_array_equal(a[k],outputs[0][0][k])
        with self.assertRaises(TypeError):pilot_predict(**legitimate,backend=Backend(),query_gt=None)
        a=outputs[0][0];g0=evaluate_saved(a,np.zeros((257,257),bool));g1=evaluate_saved(a,np.ones((257,257),bool))
        self.assertNotEqual(g0,g1)
    def test_encoding_accounting_and_cache_reuse(self):
        b=Backend();a,r=pilot_predict(**self.inputs(),backend=b)
        self.assertEqual(r['logical_requests'],20);self.assertLessEqual(b.calls,20)
        self.assertEqual(b.calls,r['configurations_decoded']);self.assertEqual(b.encodes,1);self.assertEqual(b.generations,1)
        self.assertEqual(sum(not x['cache_hit'] for x in r['requests']),b.calls)
        pilot_predict(**self.inputs(),backend=b);self.assertEqual(b.support_encoder_calls,1);self.assertEqual(b.encodes,2)
        self.assertEqual(r['generation_order'],list(range(10)))
    def test_duplicate_configurations_reused(self):
        b=Backend()
        # Invalid descriptors force support arms to reuse original full10, not extra decodes.
        orig=b.prepare_support
        def invalid(s,m):orig(s,m);b.support_features=np.ones((2,257,257))
        b.prepare_support=invalid
        a,r=pilot_predict(**self.inputs(),backend=b)
        self.assertEqual(r['selected_ids']['signed'],list(range(10)))
        self.assertTrue(any(x['cache_hit'] for x in r['requests']))
        self.assertLess(r['configurations_decoded'],20)
    def test_writer_partial_complete_and_never_overwrite(self):
        a,r=pilot_predict(**self.inputs(),backend=Backend())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new';w=RunWriter(path)
            self.assertEqual(json.loads((path/'status.json').read_text())['status'],'partial')
            with self.assertRaises(FileExistsError):RunWriter(path)
            w.save('case0',a,r)
            with self.assertRaises(FileExistsError):w.save('case0',a,r)
            with self.assertRaises(RuntimeError):w.complete()
            for i in range(1,8):w.save('case'+str(i),a,r)
            w.complete();s=json.loads((path/'status.json').read_text())
            self.assertEqual(s['status'],'completed');self.assertLessEqual(s['configurations_decoded'],160)
            with np.load(path/'case0.npz') as z:np.testing.assert_array_equal(z['full10_raw'],a['full10_raw'])
    def test_interruption_is_explicit_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=RunWriter(Path(tmp)/'interrupted')
            self.assertEqual(json.loads((w.path/'status.json').read_text())['status'],'partial')

if __name__=='__main__':unittest.main(verbosity=2)