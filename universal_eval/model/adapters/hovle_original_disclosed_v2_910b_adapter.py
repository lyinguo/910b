"""Disclosure only. Original native BF16 parent load/predict/math unchanged."""
import pathlib,json
from universal_eval.model.adapters.hovle_original_native_v3_910b_adapter import HoVLEOriginalNativeV3_910BAdapter,sha

class HoVLEOriginalDisclosedV2_910BAdapter(HoVLEOriginalNativeV3_910BAdapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
  self.nr_path=native_numerical_review;self.nr_sha=native_numerical_review_sha256;super().__init__(**kwargs)
 def load(self,*args,**kwargs):
  assert pathlib.Path(self.nr_path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(self.nr_path)==self.nr_sha;nr=json.loads(pathlib.Path(self.nr_path).read_text());assert nr['model_id']=='OpenGVLab/HoVLE-HD' and nr['official_original_BF16_dual_stream_generation_unchanged'] and nr['CPU_FP32_controls_diagnostic_only'] and nr['all_CPU_NPU14_control_parameter_SHA_equal'] and nr['full_HD_inputs_no_truncation'] and nr['native_operator_formulas_unchanged']
  for p,d in nr['evidence_sha256'].items():assert sha(p)==d
  super().load(*args,**kwargs);self.lineage.update(native_numerical_review=nr)
