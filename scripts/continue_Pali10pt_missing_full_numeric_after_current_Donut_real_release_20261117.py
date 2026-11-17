"""Wait for the actual current Donut runner/NF and continue one private stage.

No NPU use or runner lock while waiting. Keep all old stages and failures.
No workbook publication, unsafe signal, fake lock or hot modification.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,time,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Pali10Pt448_after_current_Donut_real_lock_then_missing_full_native_value_true3_continuation_20261117';OLD=B/'DonutInvoice_own_original_FP16_poscap2048_approved_fresh3_fixed200_v4_real_FLOCK_20261117';J=B/'Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v3_real_FLOCK_20261117';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(OLD/'job.json')=='9fb844c8f05543fb081c911ae458c3bc02edd98b071688f40802ef16eae4a75e' and sha(J/'job.json')=='1ac3e1549c22e4d364c22288f9b85e613fbc7523725ff9a4684fafa4b91d65fe'
 assert not (J/'state.json').exists() and not (J/'completion.json').exists();job=js(J/'job.json');batch.check_sources(job['spec'])
 while True:
  live=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners();complete=(OLD/'completion.json').exists()
  if complete and not live and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_Donut_NN_NF_runner_completion_and_real_FLOCK_release',old_job=str(OLD),old_actual_completion_exists=complete,actual_project_NN_or_NF=live,actual_FLOCK=locks,no_NPU_model_started=True));time.sleep(30)
 before=js(OLD/'completion.json');assert before['actual_returncode'] in [0,1]
 if before['actual_returncode']!=0:put('NOTIFY_old_Donut_error_preserved_continue_other_ready_model.json',dict(UTC=now(),chat_notification_not_claimed=True,first_line='严重阻塞｜正式评测未运行',old_completion=before,no_old_stage_restarted=True))
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert batch.device_idle(job['spec'])
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_new_missing_full_native_numeric_three_samples',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=child.pid,original_passed_compatibility3_and_B10_not_repeated_as_compatibility_tests=True));rc=child.wait()
 final=js(J/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_job_completion=final,no_G_or_canonical_write=True));assert rc==final['actual_returncode']==0
 put('NOTIFY_next_capacity_and_approval_review_required.json',dict(UTC=now(),chat_notification_not_claimed=True,new_full_value3_completed=True,stage_completed_not_formal200_not_G=True,next_requires_complete_numeric_and_ZIP_all200_shape_capacity_review_before_first200_measurement_approval=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True));put('NOTIFY_error.json',dict(UTC=now(),chat_notification_not_claimed=True,first_line='严重阻塞｜正式评测未运行',error=traceback.format_exc()))
  raise
