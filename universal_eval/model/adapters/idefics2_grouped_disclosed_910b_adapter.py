"""Read-only original BF16 numerical disclosure, using prior true3 diagnostics."""
import pathlib,json
from universal_eval.model.adapters.idefics2_grouped_generation_v2_910b_adapter import Idefics2GroupedGenerationV2_910BAdapter,sha
class Idefics2GroupedDisclosed910BAdapter(Idefics2GroupedGenerationV2_910BAdapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
  assert sha(native_numerical_review)==native_numerical_review_sha256
  self.nr=json.loads(pathlib.Path(native_numerical_review).read_text());self.nr_path=str(native_numerical_review);self.nr_sha=native_numerical_review_sha256
  assert self.nr['passed_diagnostic_integrity'] and self.nr['CPU_and_NPU8_FP32_diagnostic_only'] and self.nr['no_activation_replacement'] and self.nr['official_inference_remains_original_BF16_eager_cacheTrue'] and self.nr['control_parameter_count']==472
  for path,digest in self.nr['evidence_sha256'].items():assert sha(path)==digest
  for comparison in self.nr['comparisons']:assert sha(comparison['official_real3_external_NPU_evidence'])==comparison['official_real3_external_NPU_evidence_sha256']
  for name,digest in self.nr['original_real3_file_sha256'].items():assert sha(pathlib.Path(self.nr['original_real3_report'])/name)==digest
  super().__init__(**kwargs);assert self.nr['model_id']==self.model_name
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  self.lineage.update(native_numerical_review=self.nr,native_numerical_review_path=self.nr_path,native_numerical_review_sha256=self.nr_sha,native_BF16_computation_unchanged=True,prior_real3_CPU_diagnostic_reused_not_recomputed_per_report=True)
  self.lineage['precision_disclosure']+=' Static numerical diagnostics are from the identified earlier real3 report, with the actual original true generation groups and all five images per sample. Original native BF16 CPU/NPU visual/perceiver differences are retained and disclosed. Separate same-BF16-rounded472-weight CPU/NPU8 FP32 controls pin original BF16 vision positions; they diagnose arithmetic accumulation and never supply official model outputs or scores. No claim that differences vanished, all differences fall below0.001 or CPU/NPU BF16 arithmetic is bitwise equal.'
