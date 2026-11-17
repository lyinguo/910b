"""Continue another own-weight model after the existing upstream actual stage.

The queue is a real configuration consumer. Waiting uses no model/NPU or
original runner lock. Old successful stages and all original errors remain.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,os,fcntl,subprocess,traceback,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
CONFIG=R/'configs/queues/available_own_after_Donut_Pali_Deadbydawn_20261117.json'
H=B/'DeadbydawnGemma4E2B_after_Pali_full_numeric_actual_completion_immediate_continuation_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(name,value):
 f=H/name;t=f.with_name(f.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(value,ensure_ascii=False,indent=2).encode());os.replace(t,f)
def main():
 import batch_910b as batch
 from A100_alignment_evidence_runner_20261103 import unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=js(CONFIG);assert cfg['model_id']=='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
 for name,h in cfg['source_SHA256'].items():assert sha(R/name)==h,name
 assert sha(cfg['readiness'])==cfg['readiness_SHA256'];ready=js(cfg['readiness']);J=Path(ready['true3_job']);assert ready['passed'] and ready['ready_for_original_runner_true3'] and sha(J/'job.json')==cfg['new_true3_job_SHA256']==ready['job_SHA256']
 assert not (J/'state.json').exists() and not (J/'completion.json').exists();job=js(J/'job.json');batch.check_sources(job['spec'])
 donut=Path(cfg['upstream_Donut_job']);pali=Path(cfg['upstream_Pali_stage']);pali_controller=Path(cfg['upstream_Pali_controller_state_directory'])
 assert sha(donut/'job.json')==cfg['upstream_Donut_job_SHA256'] and sha(pali/'job.json')==cfg['upstream_Pali_job_SHA256']
 while True:
  live=[x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners()
  a=(donut/'completion.json').exists();b=(pali_controller/'actual_completion.json').exists()
  if a and b and not live and not locks:break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_original_Donut_and_existing_Pali_new_numeric_stage_actual_completions_and_original_FLOCK_release',upstream_Donut_actual_completion_exists=a,upstream_Pali_actual_completion_exists=b,actual_project_NN_or_NF=live,actual_FLOCK=locks,no_NPU_model_started=True));time.sleep(30)
 upstream=js(pali_controller/'actual_completion.json');put('upstream_Pali_actual_completion_preserved_for_independent_next200_review.json',dict(UTC=now(),completion=upstream,source_stage_not_repeated=True,not_an_approval_of_upstream_formal200=True,no_G_write=True))
 if upstream.get('actual_returncode')!=0:put('NOTIFY_upstream_failure_preserved_continue_other_ready_model.json',dict(UTC=now(),chat_notification_not_claimed=True,error=upstream,continue_other_own_weight_ready_model=True))
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert batch.device_idle(job['spec']) and shutil.disk_usage(D).free>=12*1024**3
 for name,h in cfg['source_SHA256'].items():assert sha(R/name)==h,name
 controller=R/cfg['model_stage_controller'];assert controller.is_relative_to(R/'scripts')
 with (H/'model_controller_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(controller)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_own_Deadbydawn_true3_then_actual_capacity_then_if_approved_NEWfresh3_fixed200',child_PID=child.pid,controller=str(controller),controller_SHA256=sha(controller),no_G_or_canonical_write=True));rc=child.wait()
 own=Path(cfg['model_stage_controller_state_directory']);actual=js(own/'actual_completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,model_controller_actual_completion=actual,no_G_or_canonical_write=True))
 if rc!=0:put('NOTIFY_error.json',dict(UTC=now(),first_line='严重阻塞｜正式评测未运行',chat_notification_not_claimed=True,actual_returncode=rc,actual_completion=actual))
 else:put('NOTIFY_completed_result_requires_original_qualification_threeD_and_dual_CAS_publication_review.json',dict(UTC=now(),chat_notification_not_claimed=True,stage_complete_not_G_publication=True,next_model_must_continue_when_own_prerequisites_ready=True))
 return rc
if __name__=='__main__':
 try:raise SystemExit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True));put('NOTIFY_error.json',dict(UTC=now(),first_line='严重阻塞｜正式评测未运行',chat_notification_not_claimed=True,error=traceback.format_exc()))
  raise
