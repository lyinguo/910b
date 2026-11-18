"""Standard lossless ZIP-LZMA with full NPY values; no tensor or math change."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
def raw_key(a):
 c=np.ascontiguousarray(a);return dict(dtype=c.dtype.str,shape=list(c.shape),raw_C_order_SHA256=hashlib.sha256(c.view(np.uint8).reshape(-1).tobytes()).hexdigest())
def write_full_standard_npz(path,values):
 path=Path(path);assert not path.exists();assert all(isinstance(n,str) and n and isinstance(v,np.ndarray) and not v.dtype.hasobject for n,v in values.items())
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_LZMA) as z:
  for n,a in values.items():
   stored=np.array(a,copy=True,order='F' if a.ndim>=2 else 'C')
   with z.open(n+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,stored,allow_pickle=False)
 return path
def copy_and_check_all_values(source,destination):
 source=Path(source);destination=Path(destination);assert source.resolve()!=destination.resolve()
 with ReadOnlyNPZ(source) as old:values={n:old[n].copy() for n in old.files}
 original={n:raw_key(a) for n,a in values.items()};write_full_standard_npz(destination,values)
 with ReadOnlyNPZ(destination) as new:
  assert set(new.files)==set(values)
  for n,a in values.items():
   actual=new[n];assert raw_key(actual)==original[n] and np.array_equal(np.ascontiguousarray(a).view(np.uint8).reshape(-1),np.ascontiguousarray(actual).view(np.uint8).reshape(-1)),n
 return dict(all_original_semantic_array_names=list(values),all_original_complete_values_SHA256=original,all_names_shapes_dtypes_and_every_raw_bit_preserved=True,no_alias_approximation_sampling_or_math_change=True,original_bytes=source.stat().st_size,new_bytes=destination.stat().st_size)
