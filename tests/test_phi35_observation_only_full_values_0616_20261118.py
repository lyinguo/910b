"""Boundary checks for observer call transparency and complete-value storage."""
import unittest
import numpy as np
import torch
from universal_eval.model.phi35_own_native_value_0616_20261118.worker import OwnPhi35Capture

class OrdinaryCallable:
    def __init__(self):self.calls=0;self.result=object()
    def compute(self,value):self.calls+=1;return self.result

class ObserverBoundaries(unittest.TestCase):
    def capture(self):
        c=OwnPhi35Capture.__new__(OwnPhi35Capture)
        c.active=True;c.original_methods=[];c.arrays={};c.same_values={};c.seq=0;c.occurrences=0;c.logical_bytes=0;c.bitwise_reuses=0
        return c
    def test_wrapper_calls_original_once_and_returns_identical_result(self):
        c=self.capture();obj=OrdinaryCallable();seen=[]
        c.wrap(obj,'compute',lambda args,kwargs:seen.append(('before',args[0])),lambda args,kwargs,result,context:seen.append(('after',result)))
        result=obj.compute(7)
        self.assertIs(result,obj.result);self.assertEqual(obj.calls,1);self.assertEqual(seen,[('before',7),('after',obj.result)])
    def test_inactive_wrapper_does_not_capture(self):
        c=self.capture();c.active=False;obj=OrdinaryCallable()
        c.wrap(obj,'compute',lambda *a:self.fail('Inactive observer called'),lambda *a:self.fail('Inactive observer called'))
        self.assertIs(obj.compute(8),obj.result);self.assertEqual(obj.calls,1)
    def test_all_duplicate_occurrences_are_counted_and_raw_values_preserved(self):
        c=self.capture();value=torch.tensor([1.,-2.,0.],dtype=torch.bfloat16);before=value.clone()
        a=c.save(value,'first');b=c.save(value,'second')
        self.assertEqual(a,b);self.assertEqual(c.occurrences,2);self.assertEqual(c.bitwise_reuses,1)
        self.assertEqual(c.logical_bytes,2*value.numel()*value.element_size());self.assertTrue(torch.equal(value,before))
        self.assertTrue(np.array_equal(c.arrays[a['array']].view(np.uint8).reshape(-1),value.view(torch.uint8).numpy().reshape(-1)))
    def test_shape_dtype_and_signed_zero_are_not_aliased(self):
        c=self.capture()
        refs=[c.save(x,'edge') for x in [torch.tensor([0.],dtype=torch.float32),torch.tensor([-0.],dtype=torch.float32),torch.tensor([[0.]],dtype=torch.float32),torch.tensor([0.],dtype=torch.bfloat16)]]
        self.assertEqual(len({r['array'] for r in refs}),4);self.assertEqual(c.occurrences,4);self.assertEqual(c.bitwise_reuses,0)

if __name__=='__main__':unittest.main()
