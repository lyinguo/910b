"""Only validate and append measured numerical disclosure; original native FP16 inference is inherited."""
import json,pathlib
from universal_eval.model.adapters.nougat_A100_FP16_native_2116_20261103_adapter import NougatA100FP16Native2116Adapter,sha
class NougatA100FP16Disclosed2116Adapter(NougatA100FP16Native2116Adapter):
 def load(self,model_path,**kwargs):
  p=self.p;path=pathlib.Path(p['numerical_review']);assert sha(path)==p['numerical_review_sha256'];review=json.loads(path.read_text());assert review['passed'] and review['CPU_diagnostics_only'] and review['original_NPU_FP16_SDPA_math_unchanged'] and review['no_CPU_decoder_answers_or_official_scores'] and review['actual_A100_and_NPU_encoder_control_input_identical']
  for filename,h in review['evidence_sha256'].items():assert sha(filename)==h,filename
  super().load(model_path,**kwargs)
  self.lineage['native_numerical_review']={'path':str(path),'sha256':sha(path),'review':review,'scope':'Static real3 same-FP16 weights/inputs actual-A100 and CPU-erf/tanh diagnostics; each new report retains fresh native probe and all24 actual per-sample activation calls. No CPU/FP32 official outputs or changes to native original FP16 calculation.'}
