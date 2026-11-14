"""Adds immutable diagnostics to unchanged original Oct4 BF16 native inference."""
import pathlib,json,hashlib
from universal_eval.model.adapters.florence_oct4_caption_v3_910b_adapter import FlorenceOct4CaptionV3_910BAdapter
class FlorenceOct4Disclosed910BAdapter(FlorenceOct4CaptionV3_910BAdapter):
 def load(self,model_path,**kwargs):
  nr=self.p['native_numerical_review']
  for source,digest in nr['evidence_sha256'].items():assert pathlib.Path(source).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()==digest
  cpu=json.loads(pathlib.Path(nr['CPU_original_reference']).read_text());control=json.loads(pathlib.Path(nr['FP32_operator_control']).read_text())
  assert cpu['CPU_only_not_official_scores'] and cpu['source_runtime_BF16_cast_preserved'] and cpu['source_F16_to_original_BF16_roundtrip_changed']==202212655 and cpu['native_NPU_activation_attention_cache_math_unchanged_by_diagnostic']
  assert control['CPU_and_FP32_NPU5_diagnostic_only'] and control['official_inference_remains_original_BF16'] and control['same_original_BF16_rounded_parameters_and_pixels'] and control['all_control_CPU_NPU_tensor_SHA_equal'] and control['control_tensor_count']==506
  assert cpu['CPU_original_vision_module_erf_vs_NPU_and_CPU_only_tanh']==nr['CPU_original_and_tanh_diagnostic_comparisons'] and control['comparisons']==nr['same_original_BF16_rounded_FP32_control_comparisons']
  super().load(model_path,**kwargs);self.lineage['native_numerical_review']=nr
  self.lineage['precision_disclosure']+=' Prior own real3 originalBF16 CPU-erf maxima: vision0.6953125, no-temporal projector1.95703125, same-actual-prefix BART4.5625; CPU-only tanh does not consistently reduce differences, with projector maximum1.873046875 and BART5.25. These actual BF16 differences are retained, not eliminated or claimed bitwise/smaller than0.001. Independent NPU5 FP32 control promotes only the same alreadyBF16-rounded weights/pixels/prefix, hashes all506 control-part tensors CPU/NPU identical, native activation unchanged; against CPU-only tanh maximum0.0004558563232421875 across vision/projector/BART. This supports precision-related native arithmetic accumulation alongside separately measured GELU semantics. Official inference remains original BF16, no ERF replacement, no FP32 or CPU output/score substitution. Static diagnostic references are prior own real3 evidence, not claimed rerun for each later report.'
