"""Original native parent unchanged; append reviewed static numerical disclosures only."""
import pathlib,json
from universal_eval.model.adapters.cosmos_reason2_original_v2_910b_adapter import CosmosReason2Original910BAdapter
from scripts.cosmos_reason2_original_cpu_v2_910b import sha

class CosmosReason2OriginalDisclosed910BAdapter(CosmosReason2Original910BAdapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
  self.review_path=pathlib.Path(native_numerical_review);self.review_sha=native_numerical_review_sha256;assert sha(self.review_path)==self.review_sha;self.review=json.loads(self.review_path.read_text());assert self.review['official_inference_remains_original_BF16_grouped_eager'] and self.review['CPU_FP32_controls_diagnostic_only'] and self.review['all315_source_BF16_and_FP32_control_SHA_verified']
  for q,digest in self.review['evidence_sha256'].items():assert pathlib.Path(q).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(q)==digest
  super().__init__(**kwargs)
 def load(self,*args,**kwargs):
  super().load(*args,**kwargs);self.lineage['native_numerical_review']=self.review;self.lineage['static_numerical_review_sha256']=self.review_sha;self.lineage['precision_disclosure']+=' Numerical review contains original NPU BF16 vs CPU-erf and CPU-only module-tanh differences, plus separate CPU/NPU2 FP32 controls of identical BF16-rounded source weights/pixels. These native differences remain; static controls come from earlier true batch3 and are referenced by SHA, not recomputed for every formal report. Official all28 language layers stay native NPU BF16 and never use CPU/FP32 answers.'
