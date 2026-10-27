"""CPU boundary tests; not model quality/parity or whole-volume GT independence."""
import unittest
from dataclasses import FrozenInstanceError
import numpy as np
from experiments.utility_feasibility_v1.slice_api import Geometry,FrozenConfig,predict_slice

class CPUBackend:
    def __init__(self):self.encodes=0;self.calls=0
    def generate(self,q,s,m):
        self.signal=float(q.mean()+s.mean()+m.mean())
        return np.array([[i*23.,i*17.] for i in range(10)],np.float32),np.array([[128.,128.]],np.float32)
    def encode(self,q):self.encodes+=1
    def decode(self,neg,pos):
        self.calls+=1
        y,x=np.indices((257,257));r=35+int(self.signal*5)
        out=(x-128)**2+(y-128)**2<r*r
        for a,b in neg:out &= (x-a)**2+(y-b)**2>25
        return out
    def cc(self,mask,pos):return mask.copy()

FIELDS=('query_image','support_image','support_mask','geometry','configuration')
def outside_benchmark_adapter(record,backend):
    # Explicit legitimate-field projection: no query-label lookup or slab choice.
    return predict_slice(**{k:record[k] for k in FIELDS},backend=backend)

class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(query_image=np.ones((1,3,257,257),np.float32),
            support_image=np.ones((1,3,257,257),np.float32),
            support_mask=np.ones((1,257,257),np.float32),geometry=Geometry(),
            configuration=FrozenConfig())
    def assertSame(self,a,b):
        self.assertEqual(a.keys(),b.keys())
        for k in a:
            if isinstance(a[k],np.ndarray):np.testing.assert_array_equal(a[k],b[k])
            else:self.assertEqual(a[k],b[k])
    def test_removed_replaced_shuffled_external_gt(self):
        gt=np.arange(257*257).reshape(257,257)%2
        original=outside_benchmark_adapter(dict(self.record,query_gt=gt),CPUBackend())
        for changed in [self.record,dict(self.record,query_gt=1-gt),
                        dict(self.record,query_gt=np.random.default_rng(42).permutation(gt)),
                        dict(self.record,query_gt=object(),oracle_budget=0,reference=object())]:
            self.assertSame(original,outside_benchmark_adapter(changed,CPUBackend()))
    def test_api_rejects_gt_and_oracle_keywords(self):
        for key in ('query_gt','gt','oracle_budget','diagnostic_reference'):
            with self.assertRaises(TypeError):
                predict_slice(**self.record,backend=CPUBackend(),**{key:None})
    def test_legitimate_input_changes_prediction(self):
        a=outside_benchmark_adapter(self.record,CPUBackend())
        b=outside_benchmark_adapter(dict(self.record,query_image=self.record['query_image']*2),CPUBackend())
        self.assertFalse(np.array_equal(a['raw'],b['raw']))
    def test_single_encoder_and_restoration(self):
        backend=CPUBackend();out=outside_benchmark_adapter(self.record,backend)
        self.assertEqual((backend.encodes,backend.calls),(1,1))
        np.testing.assert_allclose(out['negative_restored_xy'],
            (out['negative_source_xy']+.5)*257/256-.5,atol=1e-5)
        self.assertEqual(out['selected_ids'],list(range(10)))
    def test_frozen_config_and_invalid_input(self):
        with self.assertRaises(FrozenInstanceError):self.record['configuration'].token=1
        with self.assertRaises(ValueError):
            outside_benchmark_adapter(dict(self.record,support_mask=np.zeros((1,257,257))),CPUBackend())
        with self.assertRaises(ValueError):
            outside_benchmark_adapter(dict(self.record,configuration=FrozenConfig(token=1)),CPUBackend())

    def test_near2_external_gt_invariance(self):
        record=dict(self.record,configuration=FrozenConfig(arm='boundary_near2'))
        a=outside_benchmark_adapter(record,CPUBackend())
        for gt in (None,np.zeros((257,257)),np.ones((257,257)),object()):
            self.assertSame(a,outside_benchmark_adapter(dict(record,query_gt=gt),CPUBackend()))
        self.assertEqual(len(a['selected_ids']),2)
        self.assertEqual(a['decoder_configurations'],2)

if __name__=='__main__':unittest.main(verbosity=2)