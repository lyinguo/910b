"""Disclosure-only subclass. Parent original BF16/eager/no-cache/CE math is unchanged."""
import pathlib,json
from universal_eval.model.adapters.ferret_original_option_v4_0106_adapter import FerretOriginalOptionV40106Adapter
from universal_eval.model.adapters.ferret_original_io_v3_0106 import sha
class FerretOriginalDisclosed0106Adapter(FerretOriginalOptionV40106Adapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,max_new_tokens=2048,scoring='option',**kwargs):
  assert int(max_new_tokens)==2048 and scoring=='option' and sha(native_numerical_review)==native_numerical_review_sha256
  self.numerics_file=native_numerical_review;self.numerics_sha=native_numerical_review_sha256;self.numerics=json.loads(pathlib.Path(native_numerical_review).read_text());assert self.numerics['official_BF16_original_eager_option_math_unchanged'] and self.numerics['CPU_FP32_controls_diagnostic_only'];super().__init__(**kwargs)
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  for path,digest in self.numerics['evidence_sha256'].items():assert sha(path)==digest
  self.lineage.update(native_numerical_review=self.numerics,native_numerical_review_file=self.numerics_file,native_numerical_review_sha256=self.numerics_sha,static_CPU_NPU_controls_from_prior_true3_not_recomputed_per_report=True)
  self.lineage['precision_disclosure']+=' Measured original BF16 CPU/NPU selected CLIP max[44.84375,30.9375,67.5], projector max[7.3359375,8.98828125,10.1640625]. Independent NPU1 BF16 exactly reproduced these three real NPU0 visual batches. Same BF16-rounded395 vision/projector tensors and input lifted only in independent FP32 control still give encoder max[0.00476837158203125,0.004273414611816406,0.008138895034790039]; CPU-only tanh projector comparison max[0.004443168640136719,0.001847982406616211,0.004832342267036438]. Differences are retained, not erased, not all<0.001, and no18-layer CPU/FP32 language answers replace official BF16 CE.'
