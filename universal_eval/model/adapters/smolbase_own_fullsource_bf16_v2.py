"""Own original option-text likelihood protocol with complete source identity evidence.
The existing 200 result used sparse weight slices. This qualification rerun checks
every byte of every own source/actual NPU tensor; no scoring or model math changes.
"""
from pathlib import Path
import json,hashlib,shutil,re,copy
import numpy as np
import torch
from safetensors import safe_open
from universal_eval.model.adapters.idefics3_option_ranking_910b_adapter import Idefics3OptionRanking910BAdapter
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
class SmolBaseOwnFullSource(Idefics3OptionRanking910BAdapter):
 def __init__(self,protocol_file,protocol_sha256,**kwargs):
  self.full_protocol_file=Path(protocol_file);assert sha(self.full_protocol_file)==protocol_sha256;self.full_protocol=json.loads(self.full_protocol_file.read_bytes());p=self.full_protocol
  super().__init__(npu_devices=[4],max_new_tokens=2048,processor_protocol='smoldocling_option_ranking',resource_sha256=p['resource_sha256'],runtime_sha256=p['runtime_sha256'],weight_headers=p['weight_headers'],original_config=p['original_config'],original_config_sha256=p['original_config_sha256'],original_adapter_sha256=p['original_adapter_sha256'])
 def load(self,model_path,**kwargs):
  p=self.full_protocol;assert str(Path(model_path))==p['weight_path']
  for n,h in p['implementation_source_SHA256'].items():assert sha(Path.cwd()/n)==h,n
  for n,h in p['weight_file_SHA256'].items():assert sha(Path(model_path)/n)==h,n
  super().load(model_path,**kwargs)
  source={};actual=self.model.state_dict()
  for name in p['weight_file_SHA256']:
   with safe_open(str(Path(model_path)/name),framework='pt',device='cpu') as sf:
    for key in sf.keys():
     t=sf.get_tensor(key);a=actual[key];assert key not in source and t.dtype==a.dtype==torch.bfloat16 and a.device==self.device and t.shape==a.shape
     h=tensor_sha(t);assert h==tensor_sha(a);source[key]=dict(shape=list(t.shape),source_dtype=str(t.dtype),source_SHA256=h,actual_NPU_SHA256=tensor_sha(a),actual_NPU_device=str(a.device))
  assert all(t.device==self.device for t in self.model.parameters())
  evidence=self.evidence/'all_own_source_actual_NPU_SHA256.json';evidence.write_text(json.dumps(dict(model_id=self.model_name,source_count=len(source),weights=source,all_source_tensors_checked_in_full=True,no_other_checkpoint_or_head_or_tokenizer_used=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  self.lineage.update(protocol_file=str(self.full_protocol_file),protocol_SHA256=sha(self.full_protocol_file),full_source_evidence=str(evidence),full_source_evidence_SHA256=sha(evidence),current_A100_D_values=p['current_A100_D_values'],new_qualification_scope='Full own source/NPU tensor identity replaces prior sparse-slice-only proof; original option likelihood equations and all output records unchanged',no_autoregressive_or_probability_protocol_switch=True,original_cache_policy='use_cache=False for every original candidate forward',actual_NPU_card=4)
  self.full_trace=self.evidence/'full_source_and_visual_sample_trace.jsonl';self.full_trace.touch();self.lineage['original_parent_sample_trace']=str(self.trace_path);self.lineage['per_sample_evidence']=str(self.full_trace);self.full_visual=[]
  def visual(module,args,out):
   assert out.device==self.device and out.dtype==torch.bfloat16
   path=self.evidence/f'own_visual_sample_{self.calls:05d}_candidate_{len(self.full_visual):02d}.npz';assert not path.exists();a=out.detach().cpu().contiguous().view(torch.uint16).numpy();np.savez_compressed(path,own_projected_visual=a)
   self.full_visual.append(dict(path=str(path),SHA256=sha(path),shape=list(out.shape),dtype=str(out.dtype),actual_NPU_value_SHA256=tensor_sha(out)))
  self.model.model.connector.register_forward_hook(visual)
 def predict(self,prompt,media_path=None):
  assert shutil.disk_usage(self.evidence).free>self.full_protocol['hard_reserve_bytes']+self.full_protocol['log_reserve_bytes']+self.full_protocol['sample_guard_bytes']
  self.full_visual=[]
  try:return super().predict(prompt,media_path)
  finally:
   key=(re.sub(r'\n+Answer the question by providing only the option letter.*$','',prompt or '',flags=re.S).strip(),media_path)
   entry=copy.deepcopy(self.details[key]);entry['full_own_visual_values']=list(self.full_visual)
   self.details[key]=entry
   with self.full_trace.open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
