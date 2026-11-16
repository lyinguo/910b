"""After the real E4B job and completed E2B dual publication, run own HF Coder.
Never touch an active model/verifier, old failure, original PAUSE owner or source.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,time,fcntl,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'CoderGemma4E2B_wait_E4B_actual_completion_and_E2B_dual_sync_then_own_HF_native_stages_v1_20261116';FRONT=B/'CoderGemma4E2B_own_saved_HF_default_template_F32_pixel_patch_frontend200_v1_20261116';PREVIOUS=B/'Gemma4E4B_own_original_BF16_approved_fresh3_fixed200_v1_own_config_real_FLOCK_20261116'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');BOUND={'universal_eval/model/coder_gemma4_e2b_own_20261116/observer.py': '5feeeaf2ae9a936bce0315eca94a5b3c7f659b420ca51cabd4a130e744b2564a', 'universal_eval/model/coder_gemma4_e2b_own_20261116/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'universal_eval/model/adapters/coder_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': '727d98ccc3c132d54b0faf3cad2bc46cf404aae817b01f4366bd901fb4416176', 'scripts/verify_coder_gemma4_e2b_own_original_BF16_native_v1_saved_HF_F32_20261116.py': '3709945b346a334c3788ea6d7a8f648903e57557e2085b86aede7a4dbe2cf9e3', 'scripts/prepare_coder_gemma4_e2b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py': '140af4d4651e02055d3e75697a9852445d3fe0cad2ed54523d31d16a849d109b', 'scripts/measure_coder_gemma4_e2b_own_original_BF16_actual_capacity_v1_saved_HF_F32_20261116.py': 'd1d8599fe4f64521ad7b4e0fe9a485e9cdb12656a7e28af37d9f2e1683887485', 'scripts/continue_coder_gemma4_e2b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py': 'a4b7771bbf2fdb977c8edd4a4cd66fa5494f86e1fb93ea9deb4f0862cf740799'}
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(t,p)
def dual_complete():
 e=js(D/'state.json')['models']['google/gemma-4-E2B-it'];reg=js(D/'registry.json').get('google/gemma-4-E2B-it',{});b=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json')
 return e['status']=='done' and e['score']==.805 and reg.get('accepted_formal200_proof_sha256')=='702305f6386c60002a416dad5d487eb92183db6f22fde9b1c4cb8600a73b2a47' and b['local_workbook_sha256']==sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx') and b['real_record_count']>=171
def run(source,stage):
 with (H/(stage+'_stdout.log')).open('xb') as out,(H/(stage+'_stderr.log')).open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(R/source)],cwd=R,env=ENV,stdout=out,stderr=err,start_new_session=True);put('state.json',dict(UTC=now(),PID=os.getpid(),phase=stage,child_PID=child.pid));rc=child.wait()
 assert rc==0,(stage,rc)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 while True:
  procs=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier','verifier']];locks=runner_lock_owners();done=(PREVIOUS/'completion.json').exists();dual=dual_complete()
  if done and dual and not procs and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_E4B_NN_verifier_and_original_lock_release_and_completed_E2B_dual_publication',previous_completion_exists=done,E2B_dual_publication_complete=dual,actual_processes=procs,runner_lock=locks,no_new_NPU=True));time.sleep(15)
 assert js(FRONT/'actual_completion.json')['actual_returncode']==0 and js(FRONT/'review.json')['passed']
 for n,h in BOUND.items():assert sha(R/n)==h,n
 run('scripts/prepare_coder_gemma4_e2b_own_original_BF16_native_readiness_v1_saved_HF_F32_20261116.py','preparing_own_HF_default_template_F32_input_source_YAML_true3')
 run('scripts/continue_coder_gemma4_e2b_own_original_BF16_true3_fresh3_fixed200_v1_saved_HF_F32_20261116.py','running_own_HF_true3_capacity_then_approved_fresh3_fixed200')
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():
   e=dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True);put('actual_completion.json',e);put('NOTIFY_error.json',dict(e,chat_notification_not_claimed=True))
  raise
