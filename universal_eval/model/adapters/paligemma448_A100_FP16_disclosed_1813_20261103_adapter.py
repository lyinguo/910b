"""Numerical disclosure only; all official load/predict/native FP16 SDPA mathematics inherited unchanged."""
import json,pathlib
from universal_eval.model.adapters.paligemma448_A100_FP16_alignment_1813_20261103_adapter import PaliGemma448A100FP16Alignment1813Adapter,sha,record

class PaliGemma448A100FP16Disclosed1813Adapter(PaliGemma448A100FP16Alignment1813Adapter):
 def load(self,model_path,**kwargs):
  p=self.protocol;assert sha(p['numerical_review'])==p['numerical_review_sha256']
  audit=json.loads(pathlib.Path(p['numerical_review']).read_text());assert audit['passed'] and audit['CPU_diagnostics_only'] and audit['official_precision_unchanged']=='FP16' and audit['all603_actual_A100_GPU_NPU_and_CPU_cast_SHA_equal']
  for path,digest in audit['evidence_sha256'].items():assert sha(path)==digest,path
  super().load(model_path,**kwargs)
  assert {n:record(v) for n,v in self.model.named_buffers()}==p['actual_A100_analytic_buffers']
  self.lineage.update(numerical_review=p['numerical_review'],numerical_review_sha256=p['numerical_review_sha256'],numerical_disclosure=audit['disclosure'],static_numeric_results=audit['results'],numerical_evidence_sha256=audit['evidence_sha256'],no_parent_native_FP16_SDPA_math_changed=True)
