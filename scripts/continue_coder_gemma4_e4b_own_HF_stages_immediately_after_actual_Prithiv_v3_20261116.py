import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,fcntl,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'CoderGemma4E4B_wait_Prithiv_v3_actual_completion_then_own_HF_native_stages_v1_20261116';PREV=B/'PrithivGemma4E4B_own_original_BF16_true3_capacity_fresh3_fixed200_continuation_v3_materialized_v2_readiness_20261116';FRONT=B/'CoderGemma4E4B_own_saved_HF_default_template_F32_pixel_patch_frontend200_v1_20261116';BOUND={'universal_eval/model/coder_gemma4_e4b_own_20261116/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'universal_eval/model/coder_gemma4_e4b_own_20261116/observer.py': '92a483014da19a91553751e4f9d80a1ae4137f5f924aa2ce52ff2b42b8c7fdad', 'universal_eval/model/adapters/coder_gemma4_e4b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': 'ffaaf6865d79b40b5ee801ed55d157934c43ce82f530f606e5b1a3527441d7cc', 'scripts/verify_coder_gemma4_e4b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': '975416f3690783c95e31e0f08a4c21f9d9de6e2561c80b60fc73e2d030fb37ec', 'scripts/prepare_coder_gemma4_e4b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py': '31ed129985e6c66cef4b540b2a35e3d7b9a750fcdb0af326cdf656ad367cb016', 'scripts/continue_coder_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py': 'cb24d34fdd29c35c2882937ed78eb50b5ae32c97f5056ba7e90cb329d72c60bb', 'scripts/measure_coder_gemma4_e4b_own_original_BF16_actual_capacity_v1_saved_HF_F32_20261116.py': '0cc6487ec47b14792c2c279b127f8b372c9e6a7eea0660e4c7280cf3080fad1e'}
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 while True:
  procs=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners();done=(PREV/'actual_completion.json').exists()
  if done and not procs and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_Prithiv_v3_NN_numeric_or_capacity_exit_and_original_lock_release',previous_actual_completion_exists=done,actual_processes=procs,runner_lock=locks,no_new_NPU=True));time.sleep(10)
 assert js(FRONT/'actual_completion.json')['actual_returncode']==0 and js(FRONT/'review.json')['passed']
 for n,h in BOUND.items():assert sha(R/n)==h,n
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 for stage,source in [('preparing_own_HF_F32_source_YAML_and_true3','scripts/prepare_coder_gemma4_e4b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py'),('running_own_true3_numeric_capacity_then_approved_fresh3_fixed200','scripts/continue_coder_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py')]:
  with (H/(stage+'_stdout.log')).open('xb') as out,(H/(stage+'_stderr.log')).open('xb') as err:
   child=subprocess.Popen([sys.executable,'-B',str(R/source)],cwd=R,stdout=out,stderr=err,env=env,start_new_session=True);put('state.json',dict(UTC=now(),PID=os.getpid(),phase=stage,child_PID=child.pid));rc=child.wait()
  assert rc==0,(stage,rc)
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():
   e=dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True);put('actual_completion.json',e);put('NOTIFY_error.json',dict(e,chat_notification_not_claimed=True))
  raise
