"""Restore original nonlearned FP16 scale representation and disclose measured native differences; inherit original inference."""
import json,pathlib,torch
from universal_eval.model.adapters.paligemma2_A100_FP16_alignment_v2_1513_20261103_adapter import PaliGemma2A100FP16AlignmentV2_1513Adapter,sha,record,fp

class PaliGemma2A100FP16Disclosed1513Adapter(PaliGemma2A100FP16AlignmentV2_1513Adapter):
 def load(self,model_path,**kwargs):
  p=self.protocol
  assert sha(p['numerical_review'])==p['numerical_review_sha256']
  audit=json.loads(pathlib.Path(p['numerical_review']).read_text())
  assert audit['passed'] and audit['CPU_diagnostics_only'] and audit['official_precision_unchanged']=='FP16' and audit['all727_actual_A100_GPU_NPU_and_CPU_cast_SHA_equal']
  for path,digest in audit['evidence_sha256'].items():assert sha(path)==digest,path
  super().load(model_path,**kwargs)
  embedding=self.model.model.language_model.embed_tokens
  before=record(embedding.embed_scale)
  assert float(embedding.embed_scale.cpu())==48.0 and embedding.embed_scale.dtype==torch.bfloat16
  # Meta construction retains source config BF16 for this analytic scalar. Actual A100 from_pretrained casts it FP16.
  # Native forward already casts the exactly representable value48 to the learned FP16 dtype; arithmetic stays identical.
  original=embedding.embed_scale.to(torch.float16)
  assert record(original)==p['actual_A100_analytic_buffers']['model.language_model.embed_tokens.embed_scale']
  embedding.embed_scale=original
  buffers={n:record(v) for n,v in self.model.named_buffers()}
  assert buffers==p['actual_A100_analytic_buffers']
  self.lineage.update(analytic_buffers=buffers,original_A100_analytic_buffers_restored=True,analytic_scale_representation_restoration={'before':before,'after':record(original),'exact_value':48.0,'native_forward_already_cast_same_value_to_learned_FP16':True,'no_learned_weight_or_inference_formula_changed':True},numerical_review=p['numerical_review'],numerical_review_sha256=p['numerical_review_sha256'],numerical_disclosure=audit['disclosure'],static_numeric_results=audit['results'],numerical_evidence_sha256=audit['evidence_sha256'])
