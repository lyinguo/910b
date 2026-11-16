"""Path/retention boundaries only; no model, fake scores, or filesystem deletion."""
import importlib.util
from pathlib import Path
import unittest
SOURCE=Path(__file__).resolve().parents[1]/'retain_nine_complete_verified_runs_nonexample_matrices_for_RedHatE2B_capacity_20261116.py'
if not SOURCE.exists():SOURCE=Path(__file__).with_name(SOURCE.name)
spec=importlib.util.spec_from_file_location('retention_boundary_module',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RetentionBoundaryTests(unittest.TestCase):
 def setUp(self):self.h=Path('/owned/formal/home')
 def test_original_batch10_groups(self):
  for i in [0,9,19]:self.assertEqual(m.classify(self.h/f'groups/{i:03}/full_vocab_KV_inputs_visual.npz',self.h,'batch10_grouped','heretic-org/gemma-3-4b-it-heretic'),(i,True))
  self.assertEqual(m.classify(self.h/'groups/008/full_vocab_KV_inputs_visual.npz',self.h,'batch10_grouped','heretic-org/gemma-3-4b-it-heretic'),(8,False))
 def test_weights_logs_and_unrecognized_names_never_candidates(self):
  for n in ['groups/008/weights.npz','groups/008/source.py','groups/008/run.log','groups/008/full_vocab_KV_inputs_visual.npy','AA/sample_00010.npz']:
   self.assertIsNone(m.classify(self.h/n,self.h,'batch10_grouped','owner'))
 def test_outside_root_rejected(self):
  with self.assertRaises(ValueError):m.classify(Path('/foreign/sample_00010.npz'),self.h,'sample_serial','owner')
 def test_google_batch2_first_both_middle_and_last_groups(self):
  for i in [0,1,2,3,96,97,98,99,100,101,102,103,196,197,198,199]:self.assertEqual(m.classify(self.h/f'sample_{i:05}.npz',self.h,'sample_serial','google/gemma-4-E4B-it'),(i,True))
  self.assertEqual(m.classify(self.h/'sample_00004.npz',self.h,'sample_serial','google/gemma-4-E4B-it'),(4,False))
 def test_batch10_serial_complete_original_examples(self):
  for i in [0,9,90,99,190,199]:self.assertEqual(m.classify(self.h/f'sample_{i:05}.npz',self.h,'sample_serial','RedHatAI/gemma-4-E4B-it'),(i,True))
  self.assertEqual(m.classify(self.h/'sample_00100.npz',self.h,'sample_serial','RedHatAI/gemma-4-E4B-it'),(100,False))
 def test_legacy_per_sample_visual_control_examples(self):
  for sub in ['full_vocab_and_visual','visual_features']:
   for i in [0,1,2,9,90,99,100,109,190,199]:self.assertEqual(m.classify(self.h/f'{sub}/{i:06}.npz',self.h,sub,'owner'),(i,True))
   self.assertEqual(m.classify(self.h/f'{sub}/000010.npz',self.h,sub,'owner'),(10,False))
 def test_out_of_range_indices_rejected(self):
  with self.assertRaises(AssertionError):m.classify(self.h/'sample_00200.npz',self.h,'sample_serial','owner')
  with self.assertRaises(AssertionError):m.classify(self.h/'groups/020/full_vocab_KV_inputs_visual.npz',self.h,'batch10_grouped','owner')
if __name__=='__main__':unittest.main()
