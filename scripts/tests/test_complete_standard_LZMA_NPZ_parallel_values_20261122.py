"""Compare complete archive bytes and every raw value against original stdlib writer."""
import importlib.util,json,hashlib,sys,zipfile,unittest
from pathlib import Path
import numpy as np

SOURCE=Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 and sys.argv[1].endswith('.py') else Path(__file__).with_name('complete_standard_LZMA_NPZ_parallel_values_20261122.py')
OUT=Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 and not sys.argv[1].startswith('-') else Path(__file__).parent/'parallel_NPZ_synthetic_complete_storage_unit_records_20261122'
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('tested_complete_standard_parallel_npz',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def original_serial(path,values):
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_LZMA) as archive:
  for name,a in values.items():
   stored=np.array(a,copy=True,order='F' if a.ndim>=2 else 'C')
   with archive.open(name+'.npy','w',force_zip64=True) as target:np.lib.format.write_array(target,stored,allow_pickle=False)

def same(path,values):
 with zipfile.ZipFile(path) as archive:
  assert archive.namelist()==[n+'.npy' for n in values] and archive.testzip() is None
  assert all(i.compress_type==zipfile.ZIP_LZMA for i in archive.infolist())
 with np.load(path,allow_pickle=False) as actual:
  assert actual.files==list(values)
  for n,a in values.items():
   b=actual[n]
   assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(np.ascontiguousarray(a).view(np.uint8).reshape(-1),np.ascontiguousarray(b).view(np.uint8).reshape(-1)),n

class CompleteStandardParallelNPZTests(unittest.TestCase):
 def pair(self,name,values,workers=8):
  a=OUT/(name+'_original.npz');b=OUT/(name+'_parallel.npz')
  original_serial(a,values);before={k:module.raw_key(v) for k,v in values.items()};module.write_full_standard_npz(b,values,max_workers=workers)
  same(a,values);same(b,values);self.assertEqual(a.read_bytes(),b.read_bytes());self.assertEqual(before,{k:module.raw_key(v) for k,v in values.items()})
 def test_mixed_full_values_bitwise_and_identical_standard_archive(self):
  rng=np.random.default_rng(29);a=rng.normal(size=(31,13,7)).astype(np.float16);nan_bits=np.array([0x7fc00001,0x80000000,0x7f800000,0xff800000],dtype=np.uint32).view(np.float32)
  self.pair('mixed',dict(fp16=a,fortran=np.asfortranarray(a),strided=a[::-1,::2,::-1],f32=rng.normal(size=(7,11)).astype(np.float32),i64=np.arange(49,dtype=np.int64).reshape(7,7),boolean=np.array([[True,False],[False,True]]),scalar=np.array(3,dtype=np.int64),zero_shape=np.empty((0,4),dtype=np.float32),nan_payload_signed_zero_inf=nan_bits,unicode_数组=np.array([5,9],dtype='>i4')))
 def test_bounded_order_across_many_members(self):
  self.pair('ordered',{str(i):np.arange(i%17+1,dtype=np.int32) for i in range(87)})
 def test_one_worker_same_complete_format(self):
  self.pair('one_worker',{'first':np.eye(13,dtype=np.float16),'last':np.ones((3,7),dtype=np.float32)},workers=1)
 def test_real_standard_ZIP64_offset_size_and_file_count_records(self):
  old_size,old_count=zipfile.ZIP64_LIMIT,zipfile.ZIP_FILECOUNT_LIMIT
  try:
   zipfile.ZIP64_LIMIT=512;zipfile.ZIP_FILECOUNT_LIMIT=3
   values={str(i):np.random.default_rng(i).integers(0,255,size=(19,41),dtype=np.uint8) for i in range(7)}
   self.pair('ZIP64',values)
   with zipfile.ZipFile(OUT/'ZIP64_parallel.npz') as archive:self.assertTrue(any(i.header_offset>512 for i in archive.infolist()))
  finally:zipfile.ZIP64_LIMIT=old_size;zipfile.ZIP_FILECOUNT_LIMIT=old_count
 def test_existing_target_never_overwritten(self):
  path=OUT/'protected.original';path.write_bytes(b'protected original target')
  with self.assertRaises(AssertionError):module.write_full_standard_npz(path,{'a':np.ones(3)})
  self.assertEqual(path.read_bytes(),b'protected original target')
 def test_pickle_and_unsafe_member_refused_before_archive_creation(self):
  for name,values in [('object',{'a':np.array([{}],dtype=object)}),('unsafe',{'../a':np.ones(3)})]:
   path=OUT/(name+'_refused.npz')
   with self.assertRaises(AssertionError):module.write_full_standard_npz(path,values)
   self.assertFalse(path.exists())
 def test_empty_complete_mapping_matches_original(self):self.pair('empty',{})

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompleteStandardParallelNPZTests))
 review=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),source=str(SOURCE),source_SHA256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),all_full_names_shapes_dtypes_raw_bits_and_standard_archive_bytes_equal=True if result.wasSuccessful() else False,all_synthetic_unit_records_preserved=str(OUT),no_CPU_learned_or_NPU_model_execution=True,no_current_model_source_mutation=True)
 (OUT/'complete_storage_unit_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(review,ensure_ascii=False));raise SystemExit(0 if result.wasSuccessful() else 1)
