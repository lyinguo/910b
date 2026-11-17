import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,json,hashlib
P=Path(__file__).parent
adapter='''"""Exact old mask-call keyword compatibility and complete truthful errors."""
import inspect,json,traceback
import torch
from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v3_private_clean_imports_20261117 import Granite41OwnOriginalBF16PrivateCleanImports
from universal_eval.model.coder_gemma4_e2b_own_20261116.observer import write,sha,save_npz
from universal_eval.model.granite41_own_original_source_20261117 import modeling as own_modeling
class Granite41OwnOriginalBF16ExactMaskABI(Granite41OwnOriginalBF16PrivateCleanImports):
 def load(self,model_path,**kwargs):
  super().load(model_path,**kwargs)
  original=own_modeling.create_causal_mask;assert 'cache_position' not in inspect.signature(original).parameters
  def original_mask_with_legacy_alias_guard(*args,**kw):
   legacy=kw.pop('cache_position');positions=kw['position_ids'];assert torch.equal(legacy,positions[0]),'Old cache_position must be the identical explicit position_ids alias'
   return original(*args,**kw)
  own_modeling.create_causal_mask=original_mask_with_legacy_alias_guard
  self.lineage.update(only_removed_redundant_deprecated_cache_position_keyword=True,removed_keyword_bitwise_equal_to_preserved_explicit_position_ids=True,original_mask_function_source_file=inspect.getfile(original),original_mask_function_source_SHA256=sha(inspect.getfile(original)),no_causal_mask_mathematics_changed=True)
 def predict(self,prompt,media_path=None):
  before=self.index
  try:return self._predict(prompt,media_path)
  except BaseException:
   assert self.index==before;self.active=False;self.failed=True;rp=self.home/f'sample_{before:05d}_failure.json'
   if rp.exists():failure=json.loads(rp.read_bytes())
   else:
    failure=dict(index=before,error=traceback.format_exc(),actual_native_forwards=0,raw_output=None,actual_model_prompt=prompt,reference=self.front[before]['reference'],failure_before_original_generate=True);write(rp,failure)
   if getattr(self,'arrays',{}):
    partial=self.home/f'sample_{before:05d}_unaccepted_partial_values.npz';save_npz(partial,self.arrays);write(self.home/f'sample_{before:05d}_unaccepted_partial_values.json',dict(archive=str(partial),archive_SHA256=sha(partial),failed_request_record=str(rp),failed_request_SHA256=sha(rp),complete_model_forward_not_claimed=True,not_an_accepted_sample=True))
   detail=dict(index=before,actual_execution_file=str(rp),actual_execution_SHA256=sha(rp),raw_output=None,error=failure['error'],actual_native_forwards=failure['actual_native_forwards'],generated_tokens=None)
   with self.trace.open('a',encoding='utf8') as f:f.write(json.dumps(detail,ensure_ascii=False)+'\\n')
   self.details.append(detail);self.index+=1;self.arrays={};self.cache_previous=self.cache_cpu=None
   # None is the original GreedyInference error value. Returning it avoids
   # re-executing this failed request through the batch retry fallback.
   return None
'''
an='universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v4_exact_mask_ABI_full_errors_20261117.py'
nf=(P/'verify_granite41_own_original_HF_BF16_4096_native_v1_20261117.py').read_text()
nf=nf.replace("assert group['target_layer'] in [0,3,6,9]","assert group['target_layer'] in {v for _,v in config.deepstack_layer_map}|set(config.spatial_target_layers)")
nf=nf.replace("assert raw==e['raw_output']==compact['raw_output']==outs[i].get('raw_output',outs[i].get('extra',{}).get('raw_output'))==details[i].get('raw_output',details[i].get('extra',{}).get('raw_output'))","assert raw==e['raw_output']==compact['raw_output']==(outs[i].get('raw_output') if outs[i].get('raw_output') is not None else outs[i].get('extra',{}).get('raw_output'))==(details[i].get('raw_output') if details[i].get('raw_output') is not None else details[i].get('extra',{}).get('raw_output'))")
line="    for n,h in line['runtime_library_source_SHA256'].items():check(n,h)"
nf=nf.replace(line,"    assert line['legacy_unrelated_startup_hook_disabled'] and line['original_HF_no_vllm_spawn_environment'] and line['original_default_text_and_vision_SDPA_confirmed'] and line['no_causal_mask_mathematics_changed'] and line['only_removed_redundant_deprecated_cache_position_keyword']\n    check(line['original_mask_function_source_file'],line['original_mask_function_source_SHA256'])\n"+line)
nn='scripts/verify_granite41_own_original_HF_BF16_4096_native_v3_exact_mask_ABI_20261117.py'
oldcn='scripts/continue_granite41_after_Pali_actual_release_true3_capacity_fresh3_200_v3_private_clean_imports_20261117.py';cn='scripts/continue_granite41_true3_capacity_fresh3_200_v4_exact_mask_ABI_20261117.py'
controller=(P/Path(oldcn).name).read_text().replace('Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v3_private_clean_imports_continuation_20261117','Granite41_true3_shape_capacity_fresh3_fixed200_v4_exact_mask_ABI_continuation_20261117').replace('Granite41_own_original_HF_BF16_4096_true3_v2_private_clean_imports_real_FLOCK_20261117','Granite41_own_original_HF_BF16_4096_true3_v3_exact_mask_ABI_real_FLOCK_20261117').replace('Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v2_private_clean_imports_real_FLOCK_20261117','Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v3_exact_mask_ABI_real_FLOCK_20261117').replace('approved_first200_v2_private_clean_imports_20261117.json','approved_first200_v3_exact_mask_ABI_20261117.json').replace('4096-approved-first200-v2-private-clean-imports-20261117.yaml','4096-approved-first200-v3-exact-mask-ABI-20261117.yaml')
ready=(P/'prepare_granite41_own_original_HF_BF16_4096_native_v4_private_clean_imports_readiness_20261117.py').read_text().replace('Granite41_own_original_HF_BF16_4096_native_v4_private_clean_imports_readiness_20261117','Granite41_own_original_HF_BF16_4096_native_v5_exact_mask_ABI_readiness_20261117').replace('Granite41_own_original_HF_BF16_4096_true3_v2_private_clean_imports_real_FLOCK_20261117','Granite41_own_original_HF_BF16_4096_true3_v3_exact_mask_ABI_real_FLOCK_20261117').replace(oldcn,cn)
ready=ready.replace('from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v3_private_clean_imports_20261117 import Granite41OwnOriginalBF16PrivateCleanImports as Granite41OwnOriginalBF16','from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v4_exact_mask_ABI_full_errors_20261117 import Granite41OwnOriginalBF16ExactMaskABI as Granite41OwnOriginalBF16').replace('import verify_granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117','import verify_granite41_own_original_HF_BF16_4096_native_v3_exact_mask_ABI_20261117').replace('universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v3_private_clean_imports_20261117.py:Granite41OwnOriginalBF16PrivateCleanImports',an+':Granite41OwnOriginalBF16ExactMaskABI').replace('scripts/verify_granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117.py',nn).replace('native_v4_private_clean_imports_20261117.json','native_v5_exact_mask_ABI_20261117.json').replace('4096-native-v4-private-clean-imports-20261117.yaml','4096-native-v5-exact-mask-ABI-20261117.yaml')
rn='scripts/prepare_granite41_own_original_HF_BF16_4096_native_v5_exact_mask_ABI_readiness_20261117.py'
files={an:adapter,nn:nf,cn:controller,rn:ready}
for n,s in files.items():ast.parse(s);(P/Path(n).name).write_bytes(s.encode())
payload={n:dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest()) for n,s in files.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,base64,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;RN=__RN__;CN=__CN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'Granite41_native_v5_exact_mask_ABI_full_errors_readiness_actual_launch_20261117';I.mkdir(exist_ok=False);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');env.pop('VLLM_WORKER_MULTIPROC_METHOD',None)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',RN],cwd=R,stdout=out,stderr=err,env=env)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_readiness_returncode=p.returncode,stdout=(I/'readiness_stdout.json').read_text(),stderr=(I/'readiness_stderr.log').read_text(),sources={n:x['SHA256'] for n,x in FILES.items()});(I/'readiness_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());assert p.returncode==0,v
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',CN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=env)
v.update(controller_PID=p.pid,old_failed_jobs_reports_source_logs_kept=True,no_model_math_budget_precision_change=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload)).replace('__RN__',repr(rn)).replace('__CN__',repr(cn));ast.parse(remote);(P/'deploy_Granite41_v5_exact_mask_ABI_full_errors_ready_controller_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
