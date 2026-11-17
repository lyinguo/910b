"""Lossless sharing of identical complete NPY values; never drops an occurrence."""
from pathlib import Path
import hashlib,json
import numpy as np

def array_key(a):
 a=np.ascontiguousarray(a)
 return (a.dtype.str,tuple(a.shape),hashlib.sha256(a.view(np.uint8).reshape(-1).tobytes()).hexdigest())

class AliasArchive:
 def __init__(self,archive,aliases,loader):
  self._z=loader(archive,allow_pickle=False);self.aliases=dict(aliases);self.files=list(aliases)
  assert set(self.aliases.values())==set(self._z.files)
 def __getitem__(self,key):return self._z[self.aliases[key]]
 def __contains__(self,key):return key in self.aliases
 def __iter__(self):return iter(self.files)
 def keys(self):return self.files
 def close(self):self._z.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()

def copy_with_exact_aliases(source,destination):
 source=Path(source);destination=Path(destination);assert source!=destination and not destination.exists()
 unique={};lookup={};aliases={};records={};original_loader=np.load
 with original_loader(source,allow_pickle=False) as z:
  for name in z.files:
   a=z[name];key=array_key(a)
   if key in lookup:
    target=lookup[key];assert np.array_equal(unique[target].view(np.uint8).reshape(-1),np.ascontiguousarray(a).view(np.uint8).reshape(-1))
   else:target=name;lookup[key]=name;unique[name]=a.copy()
   aliases[name]=target;records[name]=dict(dtype=a.dtype.str,shape=list(a.shape),raw_value_SHA256=key[2],nbytes=a.nbytes)
 np.savez_compressed(destination,**unique)
 with original_loader(source,allow_pickle=False) as original,AliasArchive(destination,aliases,original_loader) as decoded:
  assert set(original.files)==set(decoded.files)
  for name in original.files:
   a=original[name];b=decoded[name];assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a.view(np.uint8).reshape(-1),b.view(np.uint8).reshape(-1))
 return dict(all_logical_arrays_bitwise_reconstructed=True,original_occurrences=len(aliases),physical_unique_arrays=len(unique),aliases=aliases,all_original_values=records,logical_nbytes=sum(v['nbytes'] for v in records.values()),unique_nbytes=sum(a.nbytes for a in unique.values()),no_original_file_modified_or_deleted=True)
