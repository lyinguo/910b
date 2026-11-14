"""Only append immutable numerical disclosure; parent native BF16 scoring is untouched."""
import pathlib,json
from universal_eval.model.adapters.idefics2_base_original_v3_910b_adapter import Idefics2BaseOriginalV3_910BAdapter,sha
class Idefics2BaseDisclosedV2_910BAdapter(Idefics2BaseOriginalV3_910BAdapter):
 def __init__(self,*args,native_numerical_review,native_numerical_review_sha256,log_softmax_reference_review,log_softmax_reference_review_sha256,**kwargs):
  self.numerical_path=native_numerical_review;self.numerical_sha=native_numerical_review_sha256;self.log_review_path=log_softmax_reference_review;self.log_review_sha=log_softmax_reference_review_sha256
  assert sha(self.numerical_path)==self.numerical_sha and sha(self.log_review_path)==self.log_review_sha
  super().__init__(*args,**kwargs)
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  assert sha(self.numerical_path)==self.numerical_sha and sha(self.log_review_path)==self.log_review_sha
  numerical=json.loads(pathlib.Path(self.numerical_path).read_text());reference=json.loads(pathlib.Path(self.log_review_path).read_text())
  assert numerical['passed_diagnostic_integrity'] and numerical['CPU_and_NPU6_FP32_diagnostic_only'] and numerical['official_inference_remains_original_BF16_eager_cacheTrue'] and numerical['no_activation_replacement'] and numerical['control_parameter_count']==472 and numerical['all_control_CPU_NPU_parameter_SHA_equal'] and numerical['same_original_BF16_rounded_weights_pixels_and_original_BF16_vision_position_ids']
  assert reference['passed'] and reference['original_real3_report']==numerical['original_real3_report'] and reference['original_extra_real3_NPU_inference_unchanged']
  for path,digest in numerical['evidence_sha256'].items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
  for name,digest in numerical['original_real3_file_sha256'].items():assert sha(pathlib.Path(numerical['original_real3_report'])/name)==digest==reference['original_real3_file_sha256'][name]
  self.lineage.update(native_numerical_review=numerical,native_numerical_review_path=self.numerical_path,native_numerical_review_sha256=self.numerical_sha,log_softmax_reference_review=reference,log_softmax_reference_review_path=self.log_review_path,log_softmax_reference_review_sha256=self.log_review_sha,prior_real3_CPU_diagnostic_reused_not_recomputed_per_report=True,native_BF16_computation_unchanged=True)
  self.lineage['precision_disclosure']+=' Original BF16 CPU/NPU visual/perceiver differences and separate same-rounded FP32 CPU/NPU6 controls are fully preserved in the referenced static prior-real3 diagnostic, not rerun for every report and never used for scores. Original vision activation is '+numerical['original_vision_activation']+'. Native BF16 full log_softmax agrees exactly with independent FP64 max-shift/math.fsum then BF16 rounding on the reviewed3 vectors; CPU torch BF16 reduction differences are separately disclosed. The native NPU BF16 scoring calculation remains untouched.'
