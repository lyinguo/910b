import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'RedHatGemma4E2B_own_HF_frontend_then_true3_capacity_fresh3_fixed200_immediate_v1_20261116';F=B/'RedHatGemma4E2B_own_saved_HF_default_template_F32_pixel_patch_frontend200_v1_20261116';BOUND={'universal_eval/model/redhat_gemma4_e2b_own_20261116/observer.py': 'd21110b7abef3209d9b44dcbc259b73422bfc267dde606482c60c9cc2694834f', 'universal_eval/model/redhat_gemma4_e2b_own_20261116/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'universal_eval/model/adapters/redhat_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': 'cca531a65ab3a44277622bcccef2babdd13f5784ea79c3929ed5fdc9312af704', 'scripts/verify_redhat_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': '36355a0e1f5a190b363f447dae7e09153f5d8a23b841bb9eb5985fb489d3ba78', 'scripts/prepare_redhat_gemma4_e2b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py': 'b0966829ce573b831e625912bd776a13185416c4667537de352f036a4ed0fcab', 'scripts/measure_redhat_gemma4_e2b_own_original_BF16_actual_capacity_v1_saved_HF_F32_20261116.py': '2f6705f193f5fa9db619ca91559e89b5f86963a7335737f611303d18fc415b7c', 'scripts/continue_redhat_gemma4_e2b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py': 'd48f7ea0a1f4264c0aac7d9f82bf564dacd9870de56e75f54c13b07f4457c8af'}
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 while not (F/'actual_completion.json').exists():put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_only_own_CPU_input_and_source_preflight',no_NN=True));time.sleep(5)
 assert js(F/'actual_completion.json')['actual_returncode']==0 and js(F/'review.json')['passed']
 for n,h in BOUND.items():assert sha(R/n)==h,n
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 for name,stage in [('scripts/prepare_redhat_gemma4_e2b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py','own_source_YAML_real_readiness'),('scripts/continue_redhat_gemma4_e2b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py','actual_own_true3_numeric_capacity_then_approved_fresh3_fixed200')]:
  with (H/(stage+'_stdout.log')).open('xb') as out,(H/(stage+'_stderr.log')).open('xb') as err:
   p=subprocess.Popen(['python3','-B',str(R/name)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=now(),PID=os.getpid(),phase=stage,child_PID=p.pid));rc=p.wait()
  assert rc==0,(stage,rc)
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
