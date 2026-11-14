"""Only attach own measured numerical/fallback disclosure to unchanged native parent."""
import pathlib,json
from universal_eval.model.adapters.unsloth_mllama11b_original_fallback_910b_adapter import UnslothMllama11BOriginalFallback910BAdapter
from scripts.mllama11b_original_independent_cpu_910b import sha
class UnslothMllama11BOriginalFallbackDisclosed910BAdapter(UnslothMllama11BOriginalFallback910BAdapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
  assert sha(native_numerical_review)==native_numerical_review_sha256
  self.numerical_path=str(pathlib.Path(native_numerical_review).resolve());self.numerical_sha=native_numerical_review_sha256;self.numerical_review=json.loads(pathlib.Path(native_numerical_review).read_text())
  assert self.numerical_review['original_recorded_successful126_groups_max7_preserved'] and self.numerical_review['official_inference_remains_original_BF16_eager_true_groups'] and self.numerical_review['CPU_FP32_controls_only_no_language_answers'] and self.numerical_review['native_full_original_successful_batch7_vision_memory_screen_passed']
  for path,digest in self.numerical_review['evidence_sha256'].items():assert sha(path)==digest
  super().__init__(**kwargs)
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  self.lineage.update(native_numerical_review=self.numerical_review,native_numerical_review_file=self.numerical_path,native_numerical_review_sha256=self.numerical_sha,static_CPU_diagnostics_from_prior_real3_not_rerun_per_report=True)
  self.lineage['precision_disclosure']+=' Own original BF16 vs CPU-erf and CPU-only module-tanh, same BF16-rounded weights/pixels widened only in isolated NPU13 FP32 controls all retain measured differences. Independent NPU13 BF16 vision/projector replay and original maximum successful batch7 native full eager memory screen are diagnostic; no language CPU/FP32 answer or replacement precision. Numerical evidence is from prior true3 and referenced by SHA, not recomputed for each report. All original measured differences are preserved.'
