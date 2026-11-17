import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).parent
adapter='''"""Own original Granite math; forbid unrelated legacy startup mock branch."""
import os,sys
from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v1_20261117 import Granite41OwnOriginalBF16
class Granite41OwnOriginalBF16NoLegacyStartup(Granite41OwnOriginalBF16):
 def load(self,model_path,**kwargs):
  assert os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')!='spawn','The original HF deployment does not need the old unrelated vLLM startup branch'
  assert all(getattr(sys.modules.get(n),'__file__',None)!='/dev/null' for n in ['flash_attn','flash_attn.ops.triton.rotary','transformer_engine']), 'Legacy mock may not participate in this HF task'
  super().load(model_path,**kwargs)
  self.lineage.update(legacy_unrelated_startup_hook_disabled=True,original_HF_no_vllm_spawn_environment=True)
'''
nf='''"""Original full numeric verifier plus explicit inactive legacy startup guard."""
import sys,json,contextlib
sys.dont_write_bytecode=True
from pathlib import Path
from verify_granite41_own_original_HF_BF16_4096_native_v1_20261117 import verify as original_verify,js,sha
def verify(folder,count):
 line=js(folder/'operator_lineage.json')['actual_execution'];assert line['legacy_unrelated_startup_hook_disabled'] and line['original_HF_no_vllm_spawn_environment']
 value=original_verify(folder,count);value.update(independent_verifier_SHA256=sha(__file__),legacy_startup_mock_branch_not_used=True);return value
if __name__=='__main__':
 with contextlib.redirect_stdout(sys.stderr):proof=verify(Path(sys.argv[1]).resolve(),int(sys.argv[2]))
 print(json.dumps(proof,ensure_ascii=False,allow_nan=False))
'''
controller=(P/'continue_granite41_after_Pali_full_numeric_actual_release_then_true3_capacity_fresh3_200_v1_20261117.py').read_text()
controller=controller.replace("H=B/'Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v1_continuation_20261117'","H=B/'Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v2_no_legacy_startup_continuation_20261117'")
pos=controller.index('\ndef js(p):');controller=controller[:pos]+"\nENV.pop('VLLM_WORKER_MULTIPROC_METHOD',None)\n"+controller[pos:]
cn='scripts/continue_granite41_after_Pali_actual_release_true3_capacity_fresh3_200_v2_no_legacy_startup_20261117.py'
ready=(P/'prepare_granite41_own_original_HF_BF16_4096_native_v2_real_source_files_readiness_20261117.py').read_text()
ready=ready.replace('Granite41_own_original_HF_BF16_4096_native_v2_real_source_files_readiness_20261117','Granite41_own_original_HF_BF16_4096_native_v3_source_scope_readiness_20261117')
ready=ready.replace("if f.is_relative_to(R) and f.suffix=='.py' and f.is_file():source[str(f.relative_to(R))]=sha(f)","if f.is_relative_to(R) and f.suffix=='.py' and f.is_file() and (f==R/'run.py' or any(f.is_relative_to(R/x) for x in ['scripts','universal_eval','configs'])):source[str(f.relative_to(R))]=sha(f)")
ready=ready.replace('scripts/continue_granite41_after_Pali_full_numeric_actual_release_then_true3_capacity_fresh3_200_v1_20261117.py',cn)
ready=ready.replace("from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v1_20261117 import Granite41OwnOriginalBF16","from universal_eval.model.adapters.granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117 import Granite41OwnOriginalBF16NoLegacyStartup as Granite41OwnOriginalBF16")
ready=ready.replace('import verify_granite41_own_original_HF_BF16_4096_native_v1_20261117','import verify_granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117')
ready=ready.replace('universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v1_20261117.py:Granite41OwnOriginalBF16','universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117.py:Granite41OwnOriginalBF16NoLegacyStartup')
ready=ready.replace('scripts/verify_granite41_own_original_HF_BF16_4096_native_v1_20261117.py','scripts/verify_granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117.py')
ready=ready.replace('configs/protocols/granite41_own_original_HF_BF16_4096_native_v1_20261117.json','configs/protocols/granite41_own_original_HF_BF16_4096_native_v3_source_scope_20261117.json').replace('4096-native-v1-20261117.yaml','4096-native-v3-source-scope-20261117.yaml')
line=" for n in ['configuration.py','modeling.py','processing.py','downsampling.py']:"
pos=ready.index(line);ready=ready[:pos]+" preload=R/'sitecustomize.py';assert sha(preload)=='e68ec327754e0020c670f3f771fc03310420fb9b55846f6d946acf63e65d1f09';static[str(preload)]=sha(preload)\n"+ready[pos:]
rn='scripts/prepare_granite41_own_original_HF_BF16_4096_native_v3_source_scope_readiness_20261117.py'
files={
'universal_eval/model/adapters/granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117.py':adapter,
'scripts/verify_granite41_own_original_HF_BF16_4096_native_v2_no_legacy_startup_20261117.py':nf,cn:controller,rn:ready}
for n,s in files.items():ast.parse(s);(P/Path(n).name).write_bytes(s.encode())
payload={n:dict(data=base64.b64encode(s.encode()).decode(),SHA256=hashlib.sha256(s.encode()).hexdigest()) for n,s in files.items()}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';FILES=__FILES__;RN=__RN__;CN=__CN__
for n,x in FILES.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];ast.parse(raw);p=R/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
I=B/'Granite41_native_numeric_v3_source_scope_disabled_legacy_startup_readiness_actual_launch_20261117';I.mkdir(exist_ok=False);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');env.pop('VLLM_WORKER_MULTIPROC_METHOD',None)
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',RN],cwd=R,stdout=out,stderr=err,env=env)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_readiness_returncode=p.returncode,stdout=(I/'readiness_stdout.json').read_text(),stderr=(I/'readiness_stderr.log').read_text(),sources={n:x['SHA256'] for n,x in FILES.items()});(I/'readiness_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());assert p.returncode==0,v
with (I/'controller_stdout.log').open('xb') as out,(I/'controller_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',CN],cwd=R,stdout=out,stderr=err,start_new_session=True,env=env)
v.update(controller_PID=p.pid,original_HF_environment_no_unrelated_vllm_spawn_branch=True);(I/'actual_launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False))
'''.replace('__FILES__',repr(payload)).replace('__RN__',repr(rn)).replace('__CN__',repr(cn));ast.parse(remote);(P/'deploy_Granite41_v3_source_scope_disabled_legacy_startup_ready_and_controller_20261117_remote.py').write_bytes(remote.encode());print(json.dumps({n:x['SHA256'] for n,x in payload.items()}))
