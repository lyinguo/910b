"""Huihui numerical disclosure only; native serial parent math is unchanged."""
import pathlib,json
from universal_eval.model.adapters.huihui_mllama11b_original_serial_min0_v2_910b_adapter import HuihuiMllama11BOriginalSerialMin0V2_910BAdapter
from scripts.huihui_mllama11b_original_independent_cpu_910b import sha
class HuihuiMllama11BOriginalMin0DisclosedV3_910BAdapter(HuihuiMllama11BOriginalSerialMin0V2_910BAdapter):
    def __init__(self,native_numerical_review,native_numerical_review_sha256,**kwargs):
        assert sha(native_numerical_review)==native_numerical_review_sha256
        self.numerical_path=str(pathlib.Path(native_numerical_review).resolve());self.numerical_sha=native_numerical_review_sha256;self.numerical_review=json.loads(pathlib.Path(native_numerical_review).read_text())
        assert self.numerical_review['own_original_serial1_single_BOS_explicit_BF16_eager_preserved'] and self.numerical_review['CPU_FP32_controls_only_no_language_answers'] and self.numerical_review['all_original_numerical_differences_retained']
        for path,digest in self.numerical_review['evidence_sha256'].items():assert sha(path)==digest
        super().__init__(**kwargs)
    def load(self,model_path,**kwargs):
        super().load(model_path,**kwargs)
        self.lineage.update(native_numerical_review=self.numerical_review,native_numerical_review_file=self.numerical_path,native_numerical_review_sha256=self.numerical_sha,static_CPU_diagnostics_from_prior_real3_not_rerun_per_report=True)
        self.lineage['precision_disclosure']+=' Own original serial1 native BF16 encoder/projector vs CPU-erf and CPU-only module-tanh retain measured differences. Isolated NPU10 FP32 controls use exactly the same own BF16-rounded weights/pixels widened only for diagnosis. Independent NPU10 BF16 full visual replay and all511 full parameter SHA are retained. No40-layer CPU/FP32 language answer or replacement official precision; exact arrays/differences are disclosed by immutable static SHA.'
