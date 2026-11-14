"""Disclosure-only subclass; original native learned calculation and generation inherited unchanged."""
import pathlib,json
from universal_eval.model.adapters.mineru_original_native_910b_adapter import MinerUOriginalNative910BAdapter
from scripts.mineru_original_independent_cpu_v2_910b import sha
class MinerUOriginalDisclosed910BAdapter(MinerUOriginalNative910BAdapter):
 def __init__(self,numerical_review_file,numerical_review_sha256,**kwargs):
  self.numeric_file=str(pathlib.Path(numerical_review_file).resolve());self.numeric_sha=numerical_review_sha256;assert sha(self.numeric_file)==self.numeric_sha
  self.numeric=json.loads(pathlib.Path(self.numeric_file).read_text())
  for path,digest in self.numeric['evidence_sha256'].items():assert sha(path)==digest
  super().__init__(**kwargs)
 def load(self,*args,**kwargs):
  super().load(*args,**kwargs)
  self.lineage['native_numerical_review']=self.numeric
  self.lineage['numerical_review_file']=self.numeric_file;self.lineage['numerical_review_sha256']=self.numeric_sha
  self.lineage['precision_disclosure']+=' Static independent original BF16 CPU references, same-input32-layer/SDPA/activation controls and independent NPU6 FP32 diagnostic retain all observed original BF16 differences. BF16 encoder max1536/1560/1520 and merger196/160/216 are not hidden; same-BF16-rounded-weight FP32 CPU-tanh/NPU control encoder max1.1148681640625/0.4912109375/0.24951171875 and merger0.20556640625/0.0211181640625/0.0194091796875 remain nonzero. No all-less-than0.001 or bitwise equivalence claim. Static controls come from earlier real3 only; each current run independently records its actual57 activations/probe/weights/vision/logits/cache. CPU and FP32 controls never replace official BF16 or write model scores.'
