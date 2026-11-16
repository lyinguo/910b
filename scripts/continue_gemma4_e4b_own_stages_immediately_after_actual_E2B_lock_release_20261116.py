"""Wait for the actual Gemma4E2B runner, then prepare own E4B and run its stages.
No duplicate runner or live source mutation; no waiting for a heartbeat or A100.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,time,json,hashlib,fcntl,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Gemma4E4B_wait_actual_E2B_completion_then_own_readiness_true3_fresh3_fixed200_v1_20261116'
PREVIOUS=B/'Gemma4E2B_own_original_BF16_capacity_recovered_approved_fresh3_fixed200_v5_real_FLOCK_20261116';FRONT=B/'Gemma4E4B_own_original_BF16_adapter_template_false_pixel_patch_frontend200_v1_20261116'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
BOUND={'universal_eval/model/gemma4_e4b_own_20261116/observer.py': 'cb905a7d2be6e4193fbcd08a64318931c8801385a9c14c14baa459138063867c', 'universal_eval/model/gemma4_e4b_own_20261116/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'universal_eval/model/adapters/gemma4_e4b_own_original_BF16_native_v1_20261116.py': '99ef86d239ad6bef730eb247b4d272206d714067ab95fe86cb255bc2f5dbb11e', 'scripts/verify_gemma4_e4b_own_original_BF16_native_v1_20261116.py': 'bf22e26de992a1c16b473a32ebd7885e29e56b3588336655b24532fdb258dd90', 'scripts/prepare_gemma4_e4b_own_original_BF16_native_readiness_v1_own_config_20261116.py': '007c50ba0f13a6f85b17f95129c8bb5203888b1d661d31139c33da690ec6493b', 'scripts/measure_gemma4_e4b_own_original_BF16_actual_capacity_v1_own_config_20261116.py': '5545d75044a7780d5ca1d76aa8d32ded9902a8e0b3adafa1bebbdc1ed136dec4', 'scripts/continue_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_own_config_20261116.py': 'ff5284672e6dacf9c9b94ad7239249ec1796e378707c25d34698c02d12150b2e'}
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(t,p)
def run(source,stage):
 with (H/(stage+'_stdout.log')).open('xb') as out,(H/(stage+'_stderr.log')).open('xb') as err:
  p=subprocess.Popen(['python3','-B',str(R/source)],cwd=R,env=ENV,stdout=out,stderr=err,start_new_session=True);put('state.json',dict(UTC=now(),PID=os.getpid(),phase=stage,child_PID=p.pid,source=source));rc=p.wait()
 assert rc==0,(stage,rc)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 while True:
  done=PREVIOUS/'completion.json';cpu=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier','verifier']]
  if done.exists() and not cpu and not runner_lock_owners():break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_same_real_E2B_NN_and_independent_verifier_and_original_FLOCK_release',previous_completion_exists=done.exists(),real_processes=cpu,runner_lock=runner_lock_owners(),no_new_NPU=True));time.sleep(15)
 assert sha(PREVIOUS/'job.json')=='87a3ea8309cb25fe5d8740fedcc4706a099ad30409a5633dbbf8842c50968684';previous=js(PREVIOUS/'completion.json');put('previous_actual_completion.json',previous)
 assert (FRONT/'actual_completion.json').exists() and js(FRONT/'actual_completion.json')['actual_returncode']==0 and js(FRONT/'review.json')['passed']
 for n,h in BOUND.items():assert sha(R/n)==h,n
 # New own bindings are created only after the previous real lock is released.
 run('scripts/prepare_gemma4_e4b_own_original_BF16_native_readiness_v1_own_config_20261116.py','preparing_own_E4B_native_source_YAML_true3_binding')
 run('scripts/continue_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_own_config_20261116.py','running_own_E4B_true3_then_capacity_approved_NEWfresh3_fixed200')
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():
   message=dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True);put('actual_completion.json',message);put('NOTIFY_error.json',dict(message,chat_notification_not_claimed=True))
  raise
