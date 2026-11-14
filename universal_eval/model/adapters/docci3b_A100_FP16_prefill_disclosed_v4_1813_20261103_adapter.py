"""Only disclose measured numerical differences; inherit original FP16/SDPA/1024 inference and NPU hardware profiling."""
import json,pathlib
from universal_eval.model.adapters.docci3b_A100_FP16_prefill_profile_v4_1813_20261103_adapter import Docci3bA100FP16PrefillProfileV4_1813Adapter,sha,record
class Docci3bA100FP16PrefillDisclosedV4_1813Adapter(Docci3bA100FP16PrefillProfileV4_1813Adapter):
 def load(self,model_path,**kwargs):
  p=self.protocol;assert sha(p['numerical_review'])==p['numerical_review_sha256']
  audit=json.loads(pathlib.Path(p['numerical_review']).read_text());assert audit['passed'] and audit['CPU_diagnostics_only'] and audit['official_precision_unchanged']=='FP16' and audit['all727_actual_A100_GPU_NPU_and_CPU_cast_SHA_equal']
  for path,digest in audit['evidence_sha256'].items():assert sha(path)==digest,path
  super().load(model_path,**kwargs)
  assert {n:record(v) for n,v in self.model.named_buffers()}==p['actual_A100_analytic_buffers']
  self.lineage.update(numerical_review=p['numerical_review'],numerical_review_sha256=p['numerical_review_sha256'],numerical_disclosure=audit['disclosure'],static_numeric_results=audit['results'],numerical_evidence_sha256=audit['evidence_sha256'],no_parent_native_FP16_SDPA_math_changed=True)
