"""Disclosure only; own reviewed original-A100 HF parent computes all official BF16 outputs."""
import pathlib,json
from universal_eval.model.adapters.zelk12_A100_HighNine_HF_BF16_serial1_v3_identity_guard_0913_20261106_adapter import Zelk12A100HighNineHF0913Adapter,sha
class Zelk12A100HighNineHFDisclosed0913Adapter(Zelk12A100HighNineHF0913Adapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
  self.nr_path=pathlib.Path(native_numerical_review).resolve();assert self.nr_path.is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(self.nr_path)==native_numerical_review_sha256
  self.nr=json.loads(self.nr_path.read_text());assert self.nr['model_id']=='zelk12/MT-Gen1-gemma-3-12B' and self.nr['official_inference_remains_original_A100_HF_BF16_serial1_generation'] and self.nr['CPU_FP32_controls_diagnostic_only'] and self.nr['all_control_CPU_NPU_parameter_SHA_equal'] and self.nr['actual_original_serial_batch_size']==1
  for path,digest in self.nr['evidence_sha256'].items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
  self.nr_SHA=native_numerical_review_sha256;super().__init__(**kwargs)
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  for source in [pathlib.Path(__file__).resolve(),self.nr_path]:
   dest=self.evidence/'source_snapshot'/source.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());assert sha(dest)==sha(source)
  self.lineage.update(native_numerical_review=self.nr,native_numerical_review_path=str(self.nr_path),native_numerical_review_sha256=self.nr_SHA,CPU_and_FP32_controls_static_from_prior_true_serial1_real3_not_recomputed_per_report=True)
  self.lineage['precision_disclosure']+=' Own actual A100 HF textSDPA/visionEager/serial1/max2048 remains official BF16. Same own439 rounded-parameter CPU/NPU4 FP32 vision controls use each original serial1 input; all genuine AA/CPU/NPU nonzero differences are retained and disclosed. These are visual diagnostics only, no CPU or FP32 LM answers, no claim of full-model bitwise equality or differences eliminated. New extra3/numerical controls never replace the later controlled NEWfresh3 and NEW200. Historical MLU grouped/attention evidence and scores remain preserved as prior records.'
