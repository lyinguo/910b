"""Complete native tensor bytes in standard shaped uint8 NPY/ZIP-LZMA.

The ordinary independent decoder already restores C-contiguous raw bytes before
viewing the original dtype/shape. Every key and every byte are retained here.
"""
from pathlib import Path
import hashlib,zipfile
import numpy as np
import torch
from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_full_standard_npz
def native_raw_view(tensor):
 t=tensor.detach().cpu().contiguous()
 return t.reshape(-1).view(torch.uint8).numpy().reshape([*t.shape,t.element_size()])
def write_complete_native_raw_npz(path,values):
 assert all(isinstance(n,str) and n and isinstance(t,torch.Tensor) and t.device.type=='cpu' for n,t in values.items())
 views={n:native_raw_view(t) for n,t in values.items()}
 return write_full_standard_npz(Path(path),views)
def verify_complete_native_archive(path,values):
 result={}
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(values)
  for n,t in values.items():
   actual=np.ascontiguousarray(z[n]);expected=native_raw_view(t)
   assert actual.dtype==expected.dtype==np.uint8 and actual.shape==expected.shape
   assert np.array_equal(actual,expected),n
   original=t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
   assert actual.tobytes()==original,n
   result[n]=dict(original_tensor_shape=list(t.shape),original_tensor_dtype=str(t.dtype),stored_complete_uint8_shape=list(actual.shape),complete_raw_SHA256=hashlib.sha256(original).hexdigest())
 return result
