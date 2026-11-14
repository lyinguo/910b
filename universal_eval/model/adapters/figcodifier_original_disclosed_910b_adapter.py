"""Disclosure/progress only; original parent learned load and manual predict unchanged."""
import pathlib,json,datetime
from universal_eval.model.adapters.figcodifier_original_manual_910b_adapter import FigCodifierOriginalManual910BAdapter,sha

class FigCodifierOriginalDisclosed910BAdapter(FigCodifierOriginalManual910BAdapter):
 def __init__(self,native_numerical_review,native_numerical_review_sha256,metric_contract_file,metric_contract_sha256,frozen_extraction_audit_file,frozen_extraction_audit_sha256,**kwargs):
  self.nr_path=native_numerical_review;self.nr_sha=native_numerical_review_sha256;self.mc_path=metric_contract_file;self.mc_sha=metric_contract_sha256;self.old_extract_path=frozen_extraction_audit_file;self.old_extract_sha=frozen_extraction_audit_sha256;super().__init__(**kwargs);self.progress=self.evidence/'generation_progress.jsonl';self._active_index=None
 def load(self,*args,**kwargs):
  for q,d in [(self.nr_path,self.nr_sha),(self.mc_path,self.mc_sha),(self.old_extract_path,self.old_extract_sha)]:assert pathlib.Path(q).resolve().is_relative_to('/root/projects/comperror/workspace/diagnostics') and sha(q)==d
  nr=json.loads(pathlib.Path(self.nr_path).read_text());assert nr['model_id']=='MathLLMs/FigCodifier' and nr['official_inference_remains_original_BF16_eager_manual_serial'] and nr['CPU_FP32_controls_diagnostic_only'] and nr['all_control_CPU_NPU_parameter_SHA_equal'] and nr['native_operator_formulas_unchanged']
  for q,d in nr['evidence_sha256'].items():assert sha(q)==d
  super().load(*args,**kwargs);self.lineage.update(native_numerical_review=nr,generation_progress=str(self.progress),original_metric_contract=json.loads(pathlib.Path(self.mc_path).read_text()),original_frozen_extraction_audit=json.loads(pathlib.Path(self.old_extract_path).read_text()))
 def _observers(self):
  super()._observers()
  def progress(module,args,out):
   n=len(self._steps);token=self._steps[-1]['next_token_ID']
   if n==1 or n%128==0 or token==92542 or n==2048:
    with self.progress.open('a') as f:f.write(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'native_decoder_forward_completed','index':self._active_index,'decoder_forward_completed':n,'tokens_already_appended_before_this_return':n-1,'observed_native_argmax_ID':token,'actual_language_batch_size':1,'original_budget':2048,'actual_NPU_device':str(self.device)})+'\n')
  self.model.language_model.register_forward_hook(progress)
 def predict(self,prompt,media_path=None):
  self._active_index=self.by_media[str(pathlib.Path(media_path).resolve())]['index']
  out=super().predict(prompt,media_path)
  with self.progress.open('a') as f:f.write(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'sample_complete','index':self._active_index,'generation_token_count':len(self._steps),'truncated':len(self._steps)>=2048,'error':None})+'\n')
  return out
