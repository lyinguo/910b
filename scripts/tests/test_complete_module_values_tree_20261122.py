"""Lossless artifact interoperability and rejection tests; no model checkpoint/grade."""
import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from transformers import Glm4vTextConfig
from transformers.cache_utils import DynamicCache
from universal_eval.utils.complete_module_values_tree_20261122 import CompleteValueTree,reconstruct,compare_reference
class CompleteValueArtifactTests(unittest.TestCase):
 def test_standard_NPZ_roundtrip_preserves_strided_BF16_and_every_occurrence(self):
  cap=CompleteValueTree('cpu');x=torch.arange(24,dtype=torch.float32).bfloat16().reshape(4,6).t();tree=cap.encode((x,{'again':x.clone(),'ids':torch.tensor([1,151343])},None));self.assertEqual(cap.occurrences,3);self.assertEqual(len(cap.arrays),2)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'complete.npz';np.savez_compressed(p,**cap.arrays)
   with np.load(p,allow_pickle=False) as z:
    a=reconstruct(tree,z);self.assertTrue(torch.equal(a[0],x));self.assertTrue(torch.equal(a[1]['again'],x));self.assertTrue(torch.equal(a[1]['ids'],torch.tensor([1,151343])))
 def test_corrupt_value_is_rejected_before_reference_math(self):
  cap=CompleteValueTree('cpu');tree=cap.encode(torch.tensor([1.,2.]));a={k:v.copy() for k,v in cap.arrays.items()};a[tree['tensor']][0]=3.
  with self.assertRaises(AssertionError):reconstruct(tree,a)
 def test_wrong_dtype_or_shape_is_rejected(self):
  cap=CompleteValueTree('cpu');tree=cap.encode(torch.tensor([1.,2.],dtype=torch.bfloat16));bad=copy.deepcopy(tree);bad['shape']=[1,2]
  with self.assertRaises(AssertionError):reconstruct(bad,cap.arrays)
  with self.assertRaises(AssertionError):compare_reference(torch.ones(2,dtype=torch.float32),tree,cap.arrays,[],'dtype')
 def test_real_fresh_DynamicCache_and_nonempty_request_rejection(self):
  config=Glm4vTextConfig(num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,hidden_size=512,head_dim=128);cache=DynamicCache(config=config);cap=CompleteValueTree('cpu');tree=cap.encode(cache,layer_index=1);other=reconstruct(tree,cap.arrays,config);self.assertIsInstance(other,DynamicCache);self.assertEqual(other.get_seq_length(1),0)
  cache.update(torch.ones(1,2,3,128),torch.zeros(1,2,3,128),1)
  with self.assertRaises(AssertionError):cap.encode(cache,layer_index=1)
 def test_real_CPU_module_reference_detects_wrong_numeric_outputs(self):
  cap=CompleteValueTree('cpu');tree=cap.encode(torch.tensor([1.,2.]));checks=[];compare_reference(torch.tensor([1.,2.]),tree,cap.arrays,checks,'exact');self.assertEqual(checks[0]['all_values'],2)
  with self.assertRaises(AssertionError):compare_reference(torch.tensor([1.,3.]),tree,cap.arrays,[],'bad',.02)
 def test_nonfinite_output_and_unknown_schema_are_rejected(self):
  cap=CompleteValueTree('cpu');tree=cap.encode(torch.tensor([float('nan')]));
  with self.assertRaises(AssertionError):compare_reference(torch.tensor([float('nan')]),tree,cap.arrays,[],'nonfinite')
  with self.assertRaises(ValueError):reconstruct({'unsupported':1},cap.arrays)
if __name__=='__main__':unittest.main()
