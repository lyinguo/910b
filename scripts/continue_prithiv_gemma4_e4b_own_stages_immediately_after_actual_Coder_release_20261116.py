import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,subprocess,fcntl,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'PrithivGemma4E4B_wait_actual_Coder_release_then_own_four_shard_true3_fresh3_fixed200_v1_20261116';FRONT=B/'PrithivGemma4E4B_own_original_BF16_adapter_template_false_pixel_patch_frontend200_v1_20261116';PREV=B/'CoderGemma4E2B_wait_E4B_actual_completion_and_E2B_dual_sync_then_own_HF_native_stages_v1_20261116';BOUND={'universal_eval/model/prithiv_gemma4_e4b_own_20261116/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'universal_eval/model/prithiv_gemma4_e4b_own_20261116/observer.py': '8217200308e81516156bdce81062541771c6e9f4de6161855124901a25de7008', 'universal_eval/model/adapters/prithiv_gemma4_e4b_own_original_BF16_native_v1_20261116.py': 'c82e1a09896d632628bac14134964571504b3c5da3f2deb5bad54ebfbbc6ed54', 'scripts/verify_prithiv_gemma4_e4b_own_original_BF16_native_v1_20261116.py': 'b99050b3c77bbf0e4a300333d333533087c7274d8170fb0288f8570891364138', 'scripts/prepare_prithiv_gemma4_e4b_own_original_BF16_native_readiness_v1_own_config_20261116.py': '2bd3923cefd88dfe7482a412798a3e65498460225f7b625d2a2f5e6d374b952b', 'scripts/continue_prithiv_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_own_config_20261116.py': '828753d9d1c5d9f522bccd7015aac8e8ca55d7e06b75375f4c8cd20ca5aa17b4', 'scripts/measure_prithiv_gemma4_e4b_own_original_BF16_actual_capacity_v1_own_config_20261116.py': '9bd916b292c2b814846e88215323c70916b9c570784616d9cd382119249a2f8d'}
def now():return datetime.now(timezone.utc).isoformat()
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 while True:
  procs=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners();done=(PREV/'actual_completion.json').exists()
  if done and not procs and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_Coder_NN_numeric_and_original_lock_release',previous_actual_completion_exists=done,actual_processes=procs,runner_lock=locks,no_new_NPU=True));time.sleep(15)
 assert js(FRONT/'actual_completion.json')['actual_returncode']==0 and js(FRONT/'review.json')['passed']
 for n,h in BOUND.items():assert sha(R/n)==h,n
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 for stage,source in [('preparing_own_four_shard_source_yaml_true3','scripts/prepare_prithiv_gemma4_e4b_own_original_BF16_native_readiness_v1_own_config_20261116.py'),('running_own_true3_numeric_capacity_then_approved_fresh3_fixed200','scripts/continue_prithiv_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_own_config_20261116.py')]:
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
