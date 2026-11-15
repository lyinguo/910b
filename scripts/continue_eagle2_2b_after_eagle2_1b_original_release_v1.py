"""Consume the frozen two-model queue; immediately continue after actual release."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,fcntl,subprocess,time,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'Eagle2_2B_wait_Eagle2_1B_actual_complete_then_original_lock_v1_1930_20261115'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(name,obj):
 p=H/name;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged
 import batch_910b as batch
 cfg=Path(sys.argv[1]).resolve();assert cfg.is_relative_to(R/'configs/queues') and sha(cfg)==sys.argv[2];q=js(cfg);assert q['next_model_id']=='nvidia/Eagle2-2B' and q['prior_model_id']=='nvidia/Eagle2-1B'
 H.mkdir(exist_ok=True);lock=(H/'continuation.guard').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 source=Path(q['next_controller']);assert sha(source)==q['next_controller_SHA256'];assert sha(q['next_readiness'])==q['next_readiness_SHA256'];ready=js(q['next_readiness']);assert ready['passed'] and ready['model_id']==q['next_model_id'];job=js(Path(ready['job'])/'job.json');roles={'model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier'}
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_Eagle2_1B_actual_completion_and_original_lock_release',parent=q['prior_controller_home'],next_model=q['next_model_id'],no_NPU_inference_yet=True,no_G_or_canonical_write=True))
 prior=Path(q['prior_controller_home'])/'actual_completion.json'
 while not prior.exists() or runner_lock_owners() or [p for p in project_processes() if p['role'] in roles] or not batch.device_idle(job['spec']):time.sleep(2)
 parent=js(prior);put('prior_model_actual_completion.json',parent);assert sha(source)==q['next_controller_SHA256'] and sha(q['next_readiness'])==q['next_readiness_SHA256'];unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec'])
 with (H/'next_controller_stdout.log').open('xb') as out,(H/'next_controller_stderr.log').open('xb') as err:
  p=subprocess.Popen([sys.executable,'-B',str(source)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'))
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_Eagle2_2B_own_true3_fresh3_fixed200',child_PID=p.pid,model_id=q['next_model_id'],prior_actual_returncode=parent['actual_returncode'],continues_without_periodic_wait=True,no_G_or_canonical_write=True));rc=p.wait()
 put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,next_controller_completion=js(Path(q['next_controller_home'])/'actual_completion.json'),more_inventory_pending=True,no_claim_all_models_completed=True,no_G_or_canonical_write=True));put('NOTIFY_queue_exhausted.json',dict(UTC=now(),severity='reviewed_stage_queue_exhausted_more_inventory_pending',requires_actual_next_available_model_preparation=True,disk_journal_is_not_a_chat_notification=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True));put('NOTIFY_error.json',dict(UTC=now(),error=traceback.format_exc(),disk_journal_is_not_a_chat_notification=True))
  raise
