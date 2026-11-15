"""Consume a reviewed private-stage queue immediately when the original lock releases.

Every model remains independently runnable by its own YAML. This is scheduling
only: no math, output filtering, score publication, pruning or lock bypass.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,time,fcntl,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 cfg=Path(sys.argv[1]);assert cfg.resolve().is_relative_to(R/'configs/queues') and sha(cfg)==sys.argv[2];q=js(cfg);H=Path(q['evidence_home']);H.mkdir(exist_ok=False);guard=(H/'queue.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);events=[]
 def put(n,v):
  p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
 try:
  for i,entry in enumerate(q['stages']):
   while runner_lock_owners() or [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier']]:
    put('state.json',dict(UTC=now(),phase='waiting_for_actual_original_lock_and_model_verifier_release',next_model=entry['model_id'],stage_index=i,no_second_runner=True));time.sleep(10)
   source=R/entry['source'];assert sha(source)==entry['source_SHA256']
   # An unavailable own checkpoint skips only this stage, never blocks the queue.
   if not Path(entry['own_weight_path']).is_dir():
    events.append(dict(UTC=now(),model_id=entry['model_id'],status='temporarily_unreadable_original_weight_skip',permanent_missing_count_changed=False));put('events.json',events);continue
   if (Path(entry['completion'])).exists():
    events.append(dict(UTC=now(),model_id=entry['model_id'],status='already_has_actual_completion_not_restarted',completion=js(entry['completion'])));put('events.json',events);continue
   with (H/(str(i)+'_stdout.log')).open('xb') as out,(H/(str(i)+'_stderr.log')).open('xb') as err:
    child=subprocess.Popen([sys.executable,'-B',str(source)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_private_model_stage',model_id=entry['model_id'],stage_index=i,child_PID=child.pid,controller_completion=entry['completion'],immediate_next_model_on_actual_exit=True,no_G_or_canonical_write=True));rc=child.wait()
   completion=js(entry['completion']) if Path(entry['completion']).exists() else None;event=dict(UTC=now(),model_id=entry['model_id'],actual_returncode=rc,completion=completion,continue_next_without_periodic_wait=True);events.append(event);put('events.json',events)
   if rc!=0:put('NOTIFY_error.json',dict(UTC=now(),severity='model_stage_error_next_eligible_stage_continues',event=event,original_failure_evidence_preserved=True))
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='reviewed_stage_queue_exhausted',all_events=events,more_inventory_pending=True,new_ready_models_must_be_prepared_before_queue_drains=True,no_claim_all_models_done=True));put('actual_completion.json',dict(UTC=now(),actual_returncode=0,events=events,more_inventory_pending=True));put('NOTIFY_queue_exhausted.json',dict(UTC=now(),severity='严重阻塞｜正式评测未运行',reason='All currently reviewed private stages completed or skipped; original inventory still pending. Inspect actual NN/verifier/lock and prepare next own available model immediately.'))
 except BaseException:
  put('NOTIFY_error.json',dict(UTC=now(),severity='严重阻塞｜正式评测未运行',error=traceback.format_exc()));raise
if __name__=='__main__':main()
