"""Only the observer storage changes; every original model method stays inherited."""
from pathlib import Path
import numpy as np
import torch
from universal_eval.model.merve_pali_full_native_value_20261117.worker import OwnWorkerCapture,inspect_own_sources,meta,jwrite,sha

class LosslessSharedOwnWorkerCapture(OwnWorkerCapture):
 def begin(self,group):
  self._same_values={};self._capture_occurrences=0;self._logical_nbytes=0;self._bitwise_reuses=0
  return super().begin(group)
 def save(self,t,label):
  assert torch.is_tensor(t);a=t.detach().cpu().contiguous();m=meta(a);data=a.view(torch.uint16).numpy() if a.dtype==torch.bfloat16 else a.numpy();key=(m['dtype'],tuple(m['shape']),m['SHA256']);self._capture_occurrences+=1;self._logical_nbytes+=data.nbytes
  if key in self._same_values:
   name=self._same_values[key];old=self.arrays[name];assert old.dtype==data.dtype and old.shape==data.shape and np.array_equal(old.view(np.uint8).reshape(-1),data.view(np.uint8).reshape(-1));self._bitwise_reuses+=1
  else:
   name='a'+str(self.seq).zfill(8)+'_'+label.replace('.','_');self.seq+=1;self.arrays[name]=data.copy();self._same_values[key]=name
  return dict(array=name,**m)
 def finish(self):
  storage=dict(schema='complete_tensor_occurrences_exact_value_sharing_v1',all_actual_tensor_occurrences=self._capture_occurrences,physical_unique_arrays=len(self.arrays),logical_unshared_raw_nbytes=self._logical_nbytes,physical_unique_raw_nbytes=sum(a.nbytes for a in self.arrays.values()),every_duplicate_compared_bitwise_to_actual_current_NPU_value=True,bitwise_duplicate_occurrences=self._bitwise_reuses,no_tensor_or_occurrence_or_numeric_check_dropped=True,no_precision_or_model_math_or_sampling_change=True,storage_source=__file__,storage_source_SHA256=sha(__file__))
  result=super().finish();p=Path(result['path']);import json;g=json.loads(p.read_bytes());g['lossless_shared_tensor_storage']=storage;jwrite(p,g);result['SHA256']=sha(p);result['lossless_shared_tensor_storage']=storage;self._same_values={};return result

def install_capture(worker,root,headers,home):
 assert not hasattr(worker,'_own_pali10pt_full_capture');assert len(headers)==603
 inspection=inspect_own_sources(worker,root,headers,home);worker._own_pali10pt_full_capture=LosslessSharedOwnWorkerCapture(worker,root,headers,home)
 return dict(installed=True,own_source_inspection=inspection,actual_model_implementation_unchanged=True,actual_model_forward_not_executed_by_install=True)

def begin_capture(worker,group):
 worker._own_pali10pt_full_capture.begin(group);return dict(started=True,group_index=group)
def finish_capture(worker):return worker._own_pali10pt_full_capture.finish()
