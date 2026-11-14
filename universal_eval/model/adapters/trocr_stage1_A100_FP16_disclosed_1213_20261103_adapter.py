"""Disclosure-only child. The native FP16 stage1 load/predict/attention/cache/positions/activation are inherited unchanged."""
import json,pathlib
from universal_eval.model.adapters.trocr_stage1_A100_FP16_alignment_v2_1213_20261103_adapter import TrOCRStage1A100FP16AlignmentV2_1213Adapter,sha
class TrOCRStage1A100FP16Disclosed1213Adapter(TrOCRStage1A100FP16AlignmentV2_1213Adapter):
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  p=self.protocol;nr=pathlib.Path(p['A100_numerical_review']);assert sha(nr)==p['A100_numerical_review_sha256'];n=json.loads(nr.read_text())
  assert n['passed'] and n['official_A100_aligned_execution_remains_FP16_SDPA_vision_eager_decoder'] and n['CPU_visual_only_no_LM_forward_no_answer_or_score_substitution'] and n['native_NPU_differences_retained_not_claimed_eliminated']
  pr=pathlib.Path(p['CPU_position_numerical_review']);assert sha(pr)==p['CPU_position_numerical_review_sha256'];pos=json.loads(pr.read_text());assert pos['same_exact_native_CPU_formula_source'] and pos['same_original_FP32_formula_then_FP16_cast']
  self.lineage['static_numerical_disclosure']={'review':str(nr),'sha256':sha(nr),'CPU_visual_only_from_prior_real3_not_recomputed_for_this_report':True,'original_erf_and_CPU_only_tanh_per_image':n['per_image'],'official_native_FP16_math_precision_preserved':True}
  self.lineage['original_CPU_position_rounding_disclosure']={'review':str(pr),'sha256':sha(pr),'same_original_formula_and_FP16_precision':True,'cross_CPU_all4104_positions_not_bitwise_equal':True,'first32_rows_bitwise_equal':pos['first32_position_rows_difference_count']==0,'max_abs':pos['max_abs'],'different_FP16_elements':pos['different_FP16_elements'],'sign_disagreements':pos['sign_disagreements'],'native_host_formula_unchanged_no_A100_constants_or_CPU_answers_substituted':True}
  self.lineage['precision']+=' Original CPU analytic positions use the identical FP32 native exp/sin/cos formula then FP16 on this host. Cross-host analytic rounding has4607/4202496 differing FP16 values, max0.00048828125 and one sign disagreement; first32 rows bitwise equal. Full differences remain disclosed; no CPU table or model-answer substitution, ERF replacement, or numerical-error-elimination claim.'
