import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,fcntl,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'RedHatGemma4E4B_wait_RedHatE2B_actual_completion_then_own_HF_native_stages_v1_20261116';PREV=B/'RedHatGemma4E2B_own_original_BF16_true3_capacity_fresh3_fixed200_continuation_v1_saved_HF_F32_20261116';FRONT=B/'RedHatGemma4E4B_own_saved_HF_default_template_F32_pixel_patch_frontend200_v1_20261116';BOUND={'scripts/prepare_redhat_gemma4_e4b_own_saved_HF_default_template_F32_frontend200_v1_20261116.py': '197f279d9017529fec21529ca4a608c7c2bd6674bfe56f5d1bd6ae56ef2d60d2', 'universal_eval/model/redhat_gemma4_e4b_own_20261116/observer.py': '6a31251f0b0bd18facf2069bff7aca499ceb74420b4444b06daea8fdc6a2c401', 'universal_eval/model/redhat_gemma4_e4b_own_20261116/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'universal_eval/model/adapters/redhat_gemma4_e4b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': 'bd9a3e113d1e6e1dbe4c4e34fd575b7aa0727733eb5c1be51190ba0855cbf3d6', 'scripts/verify_redhat_gemma4_e4b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': '000be880f6665b2ed513e6b97b7ee1f04e2d53e65b89d3d680bbcd7c9a2f9f4d', 'scripts/prepare_redhat_gemma4_e4b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py': 'fb64d3f1652732f5ab6fa5a3334372361c50a55dc50c2f541b49e5739bbf1941', 'scripts/measure_redhat_gemma4_e4b_own_original_BF16_actual_capacity_v1_saved_HF_F32_20261116.py': 'f6d040b481f6efe13e741aef1c708540d46d854476301f447669f931e0888c67', 'scripts/continue_redhat_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py': 'de01ef2a2d30d81fc89f12d66feb70d13e1db62a0cd2deaf6431ccaffcf280f6'}
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
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_RedHatE2B_NN_numeric_or_capacity_exit_and_original_lock_release',previous_actual_completion_exists=done,actual_processes=procs,runner_lock=locks,no_new_NPU=True));time.sleep(10)
 assert js(FRONT/'actual_completion.json')['actual_returncode']==0 and js(FRONT/'review.json')['passed']
 for n,h in BOUND.items():assert sha(R/n)==h,n
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 for stage,source in [('preparing_own_HF_F32_source_YAML_and_true3','scripts/prepare_redhat_gemma4_e4b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py'),('running_own_true3_numeric_capacity_then_approved_fresh3_fixed200','scripts/continue_redhat_gemma4_e4b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py')]:
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
