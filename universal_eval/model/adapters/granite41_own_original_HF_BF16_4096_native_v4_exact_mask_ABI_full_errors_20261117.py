"""Exact old mask-call keyword compatibility and complete truthful errors."""
import inspect,json,traceback
import torch
from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v3_private_clean_imports_20261117 import Granite41OwnOriginalBF16PrivateCleanImports
from universal_eval.model.coder_gemma4_e2b_own_20261116.observer import write,sha,save_npz
from universal_eval.model.granite41_own_original_source_20261117 import modeling as own_modeling
class Granite41OwnOriginalBF16ExactMaskABI(Granite41OwnOriginalBF16PrivateCleanImports):
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  original=own_modeling.create_causal_mask;assert 'cache_position' not in inspect.signature(original).parameters
  def original_mask_with_legacy_alias_guard(*args,**kw):
   legacy=kw.pop('cache_position');positions=kw['position_ids'];assert torch.equal(legacy,positions[0]),'Old cache_position must be the identical explicit position_ids alias'
   return original(*args,**kw)
  own_modeling.create_causal_mask=original_mask_with_legacy_alias_guard
  self.lineage.update(only_removed_redundant_deprecated_cache_position_keyword=True,removed_keyword_bitwise_equal_to_preserved_explicit_position_ids=True,original_mask_function_source_file=inspect.getfile(original),original_mask_function_source_SHA256=sha(inspect.getfile(original)),no_causal_mask_mathematics_changed=True)
 def predict(self,prompt,media_path=None):
  before=self.index
  try:return self._predict(prompt,media_path)
  except BaseException:
   assert self.index==before;self.active=False;self.failed=True;rp=self.home/f'sample_{before:05d}_failure.json'
   if rp.exists():failure=json.loads(rp.read_bytes())
   else:
    failure=dict(index=before,error=traceback.format_exc(),actual_native_forwards=0,raw_output=None,actual_model_prompt=prompt,reference=self.front[before]['reference'],failure_before_original_generate=True);write(rp,failure)
   if getattr(self,'arrays',{}):
    partial=self.home/f'sample_{before:05d}_unaccepted_partial_values.npz';save_npz(partial,self.arrays);write(self.home/f'sample_{before:05d}_unaccepted_partial_values.json',dict(archive=str(partial),archive_SHA256=sha(partial),failed_request_record=str(rp),failed_request_SHA256=sha(rp),complete_model_forward_not_claimed=True,not_an_accepted_sample=True))
   detail=dict(index=before,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),raw_output=None,error=failure['error'],actual_native_forwards=failure['actual_native_forwards'],generated_tokens=None)
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\n')
   self.details.append(detail);self.index+=1;self.arrays={};self.cache_previous=self.cache_cpu=None
   # None is the original GreedyInference error value. Returning it avoids
   # re-executing this failed request through the batch retry fallback.
   return None
