import sys
sys.dont_write_bytecode=True
from pathlib import Path
import tempfile,json,zipfile,hashlib
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[2])]
from universal_eval.utils.lossless_complete_alias_npz_fortran_head_20261117 import write_complete_arrays,CompleteAliasNPZ,copy_complete_arrays
with tempfile.TemporaryDirectory(prefix='prithiv_exact_NPY_isolated_') as tmp:
 p=Path(tmp);a=np.tile(np.arange(262144,dtype=np.uint16),(3,1));a[1,100]^=7;original=a.copy();v={'all_original_full_head_logits':a,'same_full_head':a.copy(),'equal_flat_but_shape_different':a.reshape(-1).copy(),'different_dtype':a.view(np.int16).copy(),'negative_zero_and_positive_zero':np.array([-0.,0.],dtype=np.float32)}
 out=p/'complete.npz';m=write_complete_arrays(out,v);assert m['original_occurrences']==5 and m['physical_unique_arrays']==4;assert m['aliases']['same_full_head']=='all_original_full_head_logits'
 with CompleteAliasNPZ(out) as d:
  assert set(d.files)==set(v)
  for name,value in v.items():assert d[name].shape==value.shape and d[name].dtype==value.dtype and np.array_equal(np.ascontiguousarray(d[name]).view(np.uint8).reshape(-1),np.ascontiguousarray(value).view(np.uint8).reshape(-1))
  assert d['all_original_full_head_logits'].flags.f_contiguous and not d['all_original_full_head_logits'].flags.writeable
 assert np.array_equal(a,original)
 before=out.read_bytes()
 try:write_complete_arrays(out,v)
 except AssertionError:pass
 else:raise AssertionError('Existing archive overwritten')
 assert out.read_bytes()==before
 bad=p/'bad.npz'
 try:write_complete_arrays(bad,{'object':np.array([object()],dtype=object)})
 except AssertionError:pass
 else:raise AssertionError('Unsafe pickle allowed')
 assert not bad.exists()
 collision=p/'collision.npz';collision.with_suffix('.aliases.json').write_text('existing')
 try:write_complete_arrays(collision,v)
 except AssertionError:pass
 else:raise AssertionError('Existing alias metadata overwritten')
 assert not collision.exists() and collision.with_suffix('.aliases.json').read_text()=='existing'
 src=p/'source.npz';np.savez_compressed(src,**v);sh=hashlib.sha256(src.read_bytes()).hexdigest();new=p/'copy.npz';r=copy_complete_arrays(src,new);assert r['all_logical_arrays_bitwise_reconstructed'] and hashlib.sha256(src.read_bytes()).hexdigest()==sh
 with CompleteAliasNPZ(new) as reader:
  assert reader['same_full_head'].flags.f_contiguous and np.array_equal(reader['same_full_head'],a)
 assert all(x['raw_C_order_value_SHA256'] for x in m['all_original_values'].values())
print(json.dumps(dict(passed=True,test_count=10,full_shape_dtype_and_original_logical_bits_preserved=True,standard_NPY_Fortran_storage_only=True,exact_duplicate_only_merged_and_all_occurrences_retained=True,unsafe_existing_files_and_pickle_rejected_before_mutation=True)))
