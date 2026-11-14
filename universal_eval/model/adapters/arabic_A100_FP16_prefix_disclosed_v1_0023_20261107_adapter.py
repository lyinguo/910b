"""Disclosure only: unchanged selected A100 Arabic prefix10 native model computation."""
import json,pathlib
from universal_eval.model.adapters.arabic_A100_FP16_prefix_native_v1_0023_20261107_adapter import ArabicA100FP16PrefixNative0023Adapter,sha
class ArabicA100FP16PrefixDisclosed0023Adapter(ArabicA100FP16PrefixNative0023Adapter):
 def load(self,model_path,**kwargs):
  p=self.p;assert sha(p['numerical_disclosure'])==p['numerical_disclosure_sha256'];self._numeric=json.loads(pathlib.Path(p['numerical_disclosure']).read_text());assert self._numeric['diagnostic_evidence_verified'] and self._numeric['no_NPU_model_math_precision_mask_budget_change']
  for name,h in self._numeric['all_static_diagnostic_evidence_sha256'].items():assert sha(name)==h,name
  super().load(model_path,**kwargs)
 def get_execution_lineage(self):
  line=super().get_execution_lineage();line['Arabic_A100_original_numerical_disclosure']=self._numeric;line['Arabic_A100_original_numerical_disclosure_sha256']=self.p['numerical_disclosure_sha256'];line['precision_disclosure']+=' Same 373 visual tensors and original half-rounded inputs were independently controlled in FP16 CPU/actual-A100/NPU and CPU/NPU5 FP32 diagnostic clones. Encoder maxima and raw-output differences remain nonzero, including FP32 maximum 2.4279708862304688. CPU-only tanh is a separate diagnostic, never substituted official NPU math or score. No full-model bitwise equivalence or eliminated-error claim.';return line
