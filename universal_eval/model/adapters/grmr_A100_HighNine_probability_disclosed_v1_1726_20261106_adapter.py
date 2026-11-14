"""Disclosure only; the own original-A100 probability parent computes all official BF16 outputs."""
import pathlib,json
from universal_eval.model.adapters.grmr_A100_HighNine_probability_native_v1_1726_20261106_adapter import GRMRActualA100HighNineProbability1726Adapter,sha
class GRMRActualA100HighNineProbabilityDisclosed1726Adapter(GRMRActualA100HighNineProbability1726Adapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
  self.nr_path=pathlib.Path(native_numerical_review).resolve();assert self.nr_path.is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(self.nr_path)==native_numerical_review_sha256
  self.nr=json.loads(self.nr_path.read_text());assert self.nr['model_id']=='qingy2024/GRMR-V3-G4B' and self.nr['official_inference_remains_actual_A100_BF16_textSDPA_visionEager_full_forward_FP32_log_softmax'] and self.nr['CPU_FP32_controls_diagnostic_only'] and self.nr['all_control_CPU_NPU_parameter_SHA_equal'] and self.nr['actual_original_serial_batch_size']==1 and self.nr['native_operator_formulas_unchanged']
  for path,digest in self.nr['evidence_sha256'].items():assert pathlib.Path(path).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(path)==digest
  self.nr_SHA=native_numerical_review_sha256;super().__init__(**kwargs)
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  sources=[pathlib.Path(__file__).resolve(),self.nr_path,pathlib.Path('/root/projects/comperror/scripts/verify_GRMR_A100_HighNine_probability_v3_1726_20261106.py'),pathlib.Path('/root/projects/comperror/scripts/audit_GRMR_A100_HighNine_raw_sources_v2_1726_20261106.py')]
  for source in sources:
   dest=self.evidence/'source_snapshot'/source.relative_to('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());assert sha(dest)==sha(source)
  self.lineage.update(native_numerical_review=self.nr,native_numerical_review_path=str(self.nr_path),native_numerical_review_sha256=self.nr_SHA,CPU_and_FP32_controls_static_from_prior_true_serial1_real3_not_recomputed_per_report=True)
  self.lineage['precision_disclosure']+=' Same own439 rounded-parameter CPU/NPU2 FP32 visual controls use each original serial1 input. Full AA/NPU true3 vocabulary and all34 cache-value differences plus nonzero visual CANN differences are retained and disclosed. CPU and FP32 visual controls supply no LM answer or official score. The official BF16 scoring parent and original three FIRST-token probability variants remain unchanged; no decoding, EOS, shortened budget, token replacement or AA RoPE-table copying. Original scoring raw None is retained as metadata; delivered raw label is explicitly a selected probability candidate.'
