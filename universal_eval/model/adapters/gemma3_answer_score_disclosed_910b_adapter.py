"""Numerical disclosure only; original Gemma3 BF16 scoring calculation is inherited."""
import pathlib,json
from universal_eval.model.adapters.gemma3_answer_score_original_v2_910b_adapter import Gemma3AnswerScoreOriginalV2_910BAdapter,sha

class Gemma3AnswerScoreDisclosed910BAdapter(Gemma3AnswerScoreOriginalV2_910BAdapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
  assert sha(native_numerical_review)==native_numerical_review_sha256;self.nr=json.loads(pathlib.Path(native_numerical_review).read_text());self.nrpath=str(native_numerical_review);self.nrsha=native_numerical_review_sha256
  assert self.nr['official_inference_remains_original_BF16_SDPA_full_forward_FP32_log_softmax'] and self.nr['CPU_FP32_controls_diagnostic_only'] and self.nr['all_control_CPU_NPU_parameter_SHA_equal'] and self.nr['native_operator_formulas_unchanged']
  for path,digest in self.nr['evidence_sha256'].items():assert sha(path)==digest
  super().__init__(**kwargs);assert self.nr['model_id']==self.model_name
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  self.lineage.update(native_numerical_review=self.nr,native_numerical_review_file=self.nrpath,native_numerical_review_sha256=self.nrsha,native_BF16_computation_unchanged=True,static_CPU_diagnostic_from_prior_real3_not_recomputed_per_report=True)
