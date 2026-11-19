"""Complete reversible bit-plane NPY payloads and bit-exact duplicate occurrences.

The manifest never substitutes for tensor values: every distinct complete tensor
has a full ZIP-LZMA NPY payload, every original name/shape/dtype/raw bit is bound.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib,json,zipfile
import numpy as np
META='__complete_original_tensor_occurrences_v1__.json'
SCHEMA='complete_original_raw_bits_bitplanes_exact_occurrences_v1'

def raw_metadata(a):
 original=np.asarray(a);a=np.ascontiguousarray(original)
 assert not original.dtype.hasobject and original.dtype.fields is None and original.dtype.subdtype is None
 return dict(dtype=original.dtype.str,shape=list(original.shape),elements=int(original.size),itemsize=original.dtype.itemsize,raw_C_SHA256=hashlib.sha256(a.view(np.uint8).reshape(-1).tobytes()).hexdigest())

def write_complete_bitplane_npz(path,values):
 path=Path(path);assert not path.exists() and values
 physical={};logical={};unique={};seq=0
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_LZMA) as z:
  for name,original in values.items():
   assert isinstance(name,str) and name and name!=META and isinstance(original,np.ndarray)
   a=np.ascontiguousarray(original);m=raw_metadata(original);key=(m['dtype'],tuple(m['shape']),m['raw_C_SHA256'])
   if key in unique:
    first,payload=unique[key]
    assert np.array_equal(np.ascontiguousarray(values[first]).view(np.uint8).reshape(-1),a.view(np.uint8).reshape(-1))
   else:
    payload='complete_value_%06d.npy'%seq;seq+=1
    raw=a.view(np.uint8).reshape(a.size,a.dtype.itemsize)
    bits=np.unpackbits(raw,axis=1,bitorder='little')
    encoded=np.packbits(bits.T,axis=1,bitorder='little')
    assert encoded.dtype==np.uint8 and encoded.shape==(8*a.dtype.itemsize,(a.size+7)//8)
    with z.open(payload,'w',force_zip64=True) as f:np.lib.format.write_array(f,np.ascontiguousarray(encoded),allow_pickle=False)
    unique[key]=(name,payload);physical[payload]=dict(m,encoded_shape=list(encoded.shape))
   logical[name]=dict(m,payload=payload)
  manifest=dict(schema=SCHEMA,bitorder='little',original_occurrences=logical,complete_unique_payloads=physical,every_duplicate_actual_raw_bit_compared=True,no_tensor_math_dtype_budget_or_occurrence_removed=True)
  z.writestr(META,json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')))
 return dict(logical_arrays=len(logical),physical_complete_arrays=len(physical),archive_bytes=path.stat().st_size)

class CompleteBitplaneNPZ:
 """NPZ-compatible logical tensors, independently decoded by shifts, not unpackbits."""
 def __init__(self,path,allow_pickle=False,**kwargs):
  assert allow_pickle is False and not kwargs
  self.path=Path(path);self._z=zipfile.ZipFile(self.path);self._cache=OrderedDict();self._cache_bytes=0
  assert len(self._z.namelist())==len(set(self._z.namelist()))
  self.meta=json.loads(self._z.read(META));assert self.meta['schema']==SCHEMA and self.meta['bitorder']=='little'
  self.logical=self.meta['original_occurrences'];self.files=list(self.logical)
  assert set(self._z.namelist())==set(self.meta['complete_unique_payloads'])|{META}
  assert set(m['payload'] for m in self.logical.values())==set(self.meta['complete_unique_payloads'])
  assert self.meta['every_duplicate_actual_raw_bit_compared'] and self.meta['no_tensor_math_dtype_budget_or_occurrence_removed']
 def __getitem__(self,name):
  m=self.logical[name];p=m['payload'];base=self.meta['complete_unique_payloads'][p]
  assert all(base[k]==m[k] for k in ['dtype','shape','elements','itemsize','raw_C_SHA256'])
  if p in self._cache:
   a=self._cache.pop(p);self._cache[p]=a;return a
  with self._z.open(p) as f:encoded=np.lib.format.read_array(f,allow_pickle=False)
  n=m['elements'];size=m['itemsize'];assert encoded.dtype==np.uint8 and encoded.shape==(size*8,(n+7)//8)
  # Independent inverse: read each original bit by integer masks and shifts.
  out=np.zeros((n,size),dtype=np.uint8);j=np.arange(n,dtype=np.int64)
  for bit in range(size*8):out[:,bit//8]|=((encoded[bit,j//8]>>(j%8))&1).astype(np.uint8)<<(bit%8)
  dt=np.dtype(m['dtype']);assert dt.itemsize==size and not dt.hasobject and dt.fields is None
  a=out.reshape(-1).view(dt).reshape(m['shape']);assert raw_metadata(a)=={k:m[k] for k in ['dtype','shape','elements','itemsize','raw_C_SHA256']}
  if a.nbytes<=128*1024**2:
   self._cache[p]=a;self._cache_bytes+=a.nbytes
   while self._cache_bytes>128*1024**2:
    _,old=self._cache.popitem(last=False);self._cache_bytes-=old.nbytes
  return a
 def __contains__(self,name):return name in self.logical
 def __iter__(self):return iter(self.files)
 def keys(self):return self.files
 def close(self):self._cache.clear();self._cache_bytes=0;self._z.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()

def copy_and_check_complete_values(source,destination):
 from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
 source=Path(source);destination=Path(destination);assert source.resolve()!=destination.resolve()
 with ReadOnlyNPZ(source) as z:values={n:np.array(z[n],copy=True) for n in z.files}
 before={n:raw_metadata(a) for n,a in values.items()};rec=write_complete_bitplane_npz(destination,values)
 with CompleteBitplaneNPZ(destination) as decoded:
  assert set(decoded.files)==set(values)
  for name,a in values.items():
   b=decoded[name];assert raw_metadata(b)==before[name] and np.array_equal(np.ascontiguousarray(a).view(np.uint8).reshape(-1),np.ascontiguousarray(b).view(np.uint8).reshape(-1)),name
 return dict(rec,all_logical_original_arrays=before,all_names_shapes_dtypes_every_original_raw_bit_equal=True,source_bytes=source.stat().st_size,no_original_file_modified_or_deleted=True)
