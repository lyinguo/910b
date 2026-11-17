import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
adapter='''"""Own HF math with inert legacy import placeholders removed in this process."""
import os,sys
from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v1_20261117 import Granite41OwnOriginalBF16
class Granite41OwnOriginalBF16PrivateCleanImports(Granite41OwnOriginalBF16):
 def load(self,model_path,**kwargs):
  assert os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')!='spawn'
  removed=[]
  for root in ['flash_attn','transformer_engine']:
   module=sys.modules.get(root)
   if getattr(module,'__file__',None)=='/dev/null':
    for name in list(sys.modules):
     if name==root or name.startswith(root+'.'):removed.append(name);del sys.modules[name]
  assert all(getattr(sys.modules.get(n),'__file__',None)!='/dev/null' for n in ['flash_attn','flash_attn.ops.triton.rotary','transformer_engine'])
  super().load(model_path,**kwargs)
  assert self.model.config._attn_implementation==self.model.config.text_config._attn_implementation==self.model.config.vision_config._attn_implementation=='sdpa'
  assert all('flash_attn' not in type(module).__module__ and 'transformer_engine' not in type(module).__module__ for module in self.model.modules())
  self.lineage.update(legacy_unrelated_startup_hook_disabled=True,original_HF_no_vllm_spawn_environment=True,inert_legacy_import_registry_removed_only_in_this_process=removed,original_default_text_and_vision_SDPA_confirmed=True,no_flash_or_transformer_engine_placeholder_model_module=True)
'''
an='universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v3_private_clean_imports_20261117.py'
oldcn='scripts/continue_granite41_after_Pali_actual_release_true3_capacity_fresh3_200_v2_no_legacy_startup_20261117.py';cn='scripts/continue_granite41_after_Pali_actual_release_true3_capacity_fresh3_200_v3_private_clean_imports_20261117.py'
controller=(P/Path(oldcn).name).read_text().replace('Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v2_no_legacy_startup_continuation_20261117','Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v3_private_clean_imports_continuation_20261117').replace('Granite41_own_original_HF_BF16_4096_true3_v1_real_FLOCK_20261117','Granite41_own_original_HF_BF16_4096_true3_v2_private_clean_imports_real_FLOCK_20261117').replace('Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v1_real_FLOCK_20261117','Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v2_private_clean_imports_real_FLOCK_20261117')
controller=controller.replace('granite41_own_original_HF_BF16_4096_approved_first200_v1_20261117.json','granite41_own_original_HF_BF16_4096_approved_first200_v2_private_clean_imports_20261117.json').replace('4096-approved-first200-v1-20261117.yaml','4096-approved-first200-v2-private-clean-imports-20261117.yaml')
ready=(P/'prepare_granite41_own_original_HF_BF16_4096_native_v3_source_scope_readiness_20261117.py').read_text().replace('Granite41_own_original_HF_BF16_4096_native_v3_source_scope_readiness_20261117','Granite41_own_original_HF_BF16_4096_native_v4_private_clean_imports_readiness_20261117').replace('Granite41_own_original_HF_BF16_4096_true3_v1_real_FLOCK_20261117','Granite41_own_original_HF_BF16_4096_true3_v2_private_clean_imports_real_FLOCK_20261117').replace(oldcn,cn)
ready=ready.replace('from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117 import Granite41OwnOriginalBF16NoLegacyStartup as Granite41OwnOriginalBF16','from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v3_private_clean_imports_20261117 import Granite41OwnOriginalBF16PrivateCleanImports as Granite41OwnOriginalBF16')
ready=ready.replace('universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117.py:Granite41OwnOriginalBF16NoLegacyStartup',an+':Granite41OwnOriginalBF16PrivateCleanImports')
ready=ready.replace('native_v3_source_scope_20261117.json','native_v4_private_clean_imports_20261117.json').replace('4096-native-v3-source-scope-20261117.yaml','4096-native-v4-private-clean-imports-20261117.yaml')
rn='scripts/prepare_granite41_own_original_HF_BF16_4096_native_v4_private_clean_imports_readiness_20261117.py';files={an:adapter,cn:controller,rn:ready}
for n,s in files.items():ast.parse(s);(P/Path(n).name).write_bytes(s.encode())
payload={n:dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest()) for n,s in files.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,base64,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;RN=__RN__;CN=__CN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'Granite41_native_numeric_v4_private_clean_imports_readiness_actual_launch_20261117';I.mkdir(exist_ok=False);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');env.pop('VLLM_WORKER_MULTIPROC_METHOD',None)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',RN],cwd=R,stdout=out,stderr=err,env=env)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_readiness_returncode=p.returncode,stdout=(I/'readiness_stdout.json').read_text(),stderr=(I/'readiness_stderr.log').read_text(),sources={n:x['SHA256'] for n,x in FILES.items()});(I/'readiness_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());assert p.returncode==0,v
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',CN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=env)
v.update(controller_PID=p.pid,no_global_package_or_sitecustomize_change=True,old_failures_preserved=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload)).replace('__RN__',repr(rn)).replace('__CN__',repr(cn));ast.parse(remote);(P/'deploy_Granite41_v4_private_clean_imports_ready_and_controller_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
