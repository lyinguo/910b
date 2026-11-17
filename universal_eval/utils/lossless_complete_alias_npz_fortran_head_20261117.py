"""Keep every logical array; share exact complete values and store full heads F-order.

NumPy's standard NPY fortran_order changes only serialized traversal. Shapes,
dtypes, logical values and semantic occurrence names are all retained.
"""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ

def key(a):
 c=np.ascontiguousarray(a)
 return (c.dtype.str,tuple(c.shape),hashlib.sha256(c.view(np.uint8).reshape(-1).tobytes()).hexdigest())

def write_complete_arrays(path,values):
 path=Path(path);index=path.with_suffix('.aliases.json');assert not path.exists() and not index.exists()
 physical={};seen={};aliases={};records={}
 for name,a in values.items():
  assert isinstance(name,str) and name and isinstance(a,np.ndarray) and not a.dtype.hasobject
  k=key(a)
  if k in seen:
   owner=seen[k];assert np.array_equal(np.ascontiguousarray(a).view(np.uint8).reshape(-1),np.ascontiguousarray(physical[owner]).view(np.uint8).reshape(-1))
  else:
   owner=name;seen[k]=owner
   if name=='all_original_full_head_logits':
    assert a.ndim==2 and a.shape[1]==262144 and a.dtype==np.uint16
    physical[name]=np.array(a,copy=True,order='F')
   else:physical[name]=np.array(a,copy=True,order='C')
  aliases[name]=owner;records[name]=dict(dtype=a.dtype.str,shape=list(a.shape),raw_C_order_value_SHA256=k[2],nbytes=a.nbytes)
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for name,a in physical.items():
   with z.open(name+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,a,allow_pickle=False)
 with ReadOnlyNPZ(path) as z:
  assert set(z.files)==set(physical)
  for name,owner in aliases.items():assert key(z[owner])==(records[name]['dtype'],tuple(records[name]['shape']),records[name]['raw_C_order_value_SHA256'])
 metadata=dict(format='complete_NPY_exact_aliases_optional_full_head_standard_Fortran_order_v1',aliases=aliases,all_original_values=records,original_occurrences=len(aliases),physical_unique_arrays=len(physical),all_logical_arrays_bitwise_reconstructed=True,no_semantic_occurrence_discarded=True,full_head_original_shape_dtype_and_C_order_bits_preserved=True,standard_NPY_Fortran_order_not_math_or_dtype_change=True)
 index.write_bytes(json.dumps(metadata,ensure_ascii=False,indent=2).encode())
 return metadata

def copy_complete_arrays(source,destination):
 source=Path(source);destination=Path(destination);assert source.resolve()!=destination.resolve()
 with ReadOnlyNPZ(source) as z:values={n:z[n].copy() for n in z.files}
 result=write_complete_arrays(destination,values)
 with ReadOnlyNPZ(source) as old,ReadOnlyNPZ(destination) as new:
  assert set(old.files)==set(result['aliases'])
  for name,owner in result['aliases'].items():
   a=old[name];b=new[owner];assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(np.ascontiguousarray(a).view(np.uint8).reshape(-1),np.ascontiguousarray(b).view(np.uint8).reshape(-1))
 return result

class CompleteAliasNPZ:
 def __init__(self,path,index=None):
  self.path=Path(path);self.index=json.loads((Path(index) if index else self.path.with_suffix('.aliases.json')).read_bytes());self.aliases=self.index['aliases'];self.files=list(self.aliases);self._data=ReadOnlyNPZ(self.path)
  assert set(self.aliases.values())==set(self._data.files) and self.index['all_logical_arrays_bitwise_reconstructed'] and self.index['no_semantic_occurrence_discarded']
 def __getitem__(self,name):return self._data[self.aliases[name]]
 def __contains__(self,name):return name in self.aliases
 def __enter__(self):return self
 def __exit__(self,*args):self._data.close()
 def close(self):self._data.close()
