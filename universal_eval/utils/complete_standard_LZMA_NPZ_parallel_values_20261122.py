"""Bounded CPU compression of complete standard NPY members; no value or model change.

Each worker uses the original standard ZipFile/NPY writer. The main thread copies
the complete local member records in original order and lets the same standard
ZipFile implementation write the central directory and ZIP64 end records.
No member is sampled, aliased, quantized, discarded, or deleted.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,zipfile
import numpy as np

def raw_key(a):
 c=np.ascontiguousarray(a)
 return dict(dtype=c.dtype.str,shape=list(c.shape),raw_C_order_SHA256=hashlib.sha256(c.view(np.uint8).reshape(-1).tobytes()).hexdigest())

def _complete_member(item):
 name,array=item
 stored=np.array(array,copy=True,order='F' if array.ndim>=2 else 'C')
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_LZMA,allowZip64=True) as archive:
  with archive.open(name+'.npy','w',force_zip64=True) as target:
   np.lib.format.write_array(target,stored,allow_pickle=False)
  assert len(archive.filelist)==1
  info=archive.filelist[0]
  assert info.filename==name+'.npy' and info.header_offset==0
  boundary=archive.start_dir
 payload=buffer.getvalue()[:boundary]
 assert payload[:4]==b'PK\x03\x04' and info.compress_type==zipfile.ZIP_LZMA and len(payload)>info.compress_size
 return info,payload

def write_full_standard_npz(path,values,max_workers=8):
 path=Path(path)
 assert not path.exists() and isinstance(max_workers,int) and 1<=max_workers<=8
 items=list(values.items())
 assert all(isinstance(n,str) and n and '/' not in n and '\\' not in n and '\0' not in n and isinstance(a,np.ndarray) and not a.dtype.hasobject for n,a in items)
 assert len({n for n,a in items})==len(items)
 # Only a bounded number of complete members can be compressed ahead of writing.
 workers=min(max_workers,len(items))
 with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_LZMA,allowZip64=True) as archive:
  if not items:return path
  with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='complete_NPY_LZMA') as pool:
   pending={i:pool.submit(_complete_member,items[i]) for i in range(workers)}
   submitted=workers
   for index in range(len(items)):
    info,payload=pending.pop(index).result()
    assert info.filename==items[index][0]+'.npy' and info.filename not in archive.NameToInfo and not archive._writing
    archive.fp.seek(archive.start_dir)
    info.header_offset=archive.start_dir
    archive._writecheck(info)
    archive._didModify=True
    assert archive.fp.write(payload)==len(payload)
    archive.filelist.append(info)
    archive.NameToInfo[info.filename]=info
    archive.start_dir=archive.fp.tell()
    if submitted<len(items):
     pending[submitted]=pool.submit(_complete_member,items[submitted]);submitted+=1
   assert not pending and len(archive.filelist)==len(items)
 return path

def copy_and_check_all_values(source,destination,max_workers=8):
 from universal_eval.utils.readonly_npz_indexed import ReadOnlyNPZ
 source=Path(source);destination=Path(destination)
 assert source.resolve()!=destination.resolve()
 with ReadOnlyNPZ(source) as old:values={n:old[n].copy() for n in old.files}
 original={n:raw_key(a) for n,a in values.items()}
 write_full_standard_npz(destination,values,max_workers=max_workers)
 with ReadOnlyNPZ(destination) as new:
  assert set(new.files)==set(values)
  for n,a in values.items():
   actual=new[n]
   assert raw_key(actual)==original[n] and np.array_equal(np.ascontiguousarray(a).view(np.uint8).reshape(-1),np.ascontiguousarray(actual).view(np.uint8).reshape(-1)),n
 return dict(all_original_semantic_array_names=list(values),all_original_complete_values_SHA256=original,all_names_shapes_dtypes_and_every_raw_bit_preserved=True,bounded_CPU_compression_workers=max_workers,no_alias_approximation_sampling_math_change_or_evidence_deletion=True,original_bytes=source.stat().st_size,new_bytes=destination.stat().st_size)
