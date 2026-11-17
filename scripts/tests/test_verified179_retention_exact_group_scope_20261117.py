"""Exercise exact cleanup classification and immutable example boundaries."""
import sys
sys.dont_write_bytecode=True
import unittest,tempfile,os,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'/root/projects/comperror/scripts')
import retain_verified179_accepted_current_nonexample_matrices_user_request_20261117 as m
class ScopeTests(unittest.TestCase):
 def test_logical_examples(self):self.assertEqual(m.KEEP,set(range(10))|set(range(90,110))|set(range(190,200)))
 def test_serial(self):self.assertEqual(m.indices_for(Path('sample_00199.npz'),{}),{199})
 def test_native_full_group(self):self.assertEqual(m.indices_for(Path('full_native_group_09.npz'),{}),set(range(90,100)))
 def test_old_group_arrays(self):self.assertEqual(m.indices_for(Path('group_10_arrays.npz'),{}),set(range(100,110)))
 def test_pair_logits_visual(self):
  for n in ['0008_visual.npz','0008_full_logits.npz']:self.assertEqual(m.indices_for(Path('samples')/n,{}),{8})
 def test_hovle(self):self.assertEqual(m.indices_for(Path('samples/099.npz'),{}),{99})
 def test_actual_native_variable_batch(self):self.assertEqual(m.indices_for(Path('groups/0046/actual_visual_positions_cache_full_vocab.npz'),{46:{97,98,99}}),{97,98,99})
 def test_actual_idefics_variable_batch(self):self.assertEqual(m.indices_for(Path('generation_groups/0003_full_logits.npz'),{3:{10,11}}),{10,11})
 def test_static_probe_never_classified(self):self.assertIsNone(m.indices_for(Path('native_GELU_probe.npz'),{}))
 def test_out_of_range_never_classified(self):self.assertIsNone(m.indices_for(Path('sample_00200.npz'),{}))
 def test_unknown_group_never_classified(self):self.assertIsNone(m.indices_for(Path('groups/0001/actual_visual_positions_cache_full_vocab.npz'),{}))
 def test_actual_reordered_execution_keeps_original_indices(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'trace.jsonl';rows=[dict(index=i,group=(i//2)+1) for i in reversed(range(200))];f.write_text('\n'.join(json.dumps(x) for x in rows));groups,_=m.sample_group_index({'sample_trace':str(f)});self.assertEqual(groups[50],{98,99});self.assertEqual(set.union(*groups.values()),set(range(200)))
 def test_regular_link_and_path_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);home=root/'own';home.mkdir();f=home/'sample_00011.npz';f.write_bytes(b'original numerical example')
   with patch.object(m,'R',root):
    self.assertTrue(m.removable_file(f,home));link=root/'linked.npz';os.link(f,link);self.assertFalse(m.removable_file(f,home));link.unlink()
    symlink=home/'alias.npz';symlink.symlink_to(f);self.assertFalse(m.removable_file(symlink,home));self.assertFalse(m.removable_file(f,root/'other'))
if __name__=='__main__':unittest.main()
