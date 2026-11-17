"""One original global FLOCK, disjoint card workers, shared-disk stage reservations."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,fcntl,subprocess,time,traceback,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R/'scripts'),str(R)]
from A100_alignment_parallel_control_20261117 import sha,can_admit,validate_cards,proc_start,real_exclusive_owner
from A100_alignment_evidence_runner_20261103 import scope,unchanged
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(tmp,p)
def main(home):
 home=Path(home).resolve();assert home.is_relative_to(D/'A100_relative5pct_alignment_20261103');q=js(home/'batch.json');rows=q['jobs'];validate_cards(rows);assert q['max_parallel_stages']==2 and len(rows)<=2
 assert sha(R/'scripts/A100_alignment_evidence_runner_20261103.py')=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 decision=js(q['latest_user_parallel_request']);assert sha(q['latest_user_parallel_request'])==q['latest_user_parallel_request_SHA256'] and decision['origin']=='actual_user_chat_instruction' and '多卡' in decision['exact_user_request'] and '并行' in decision['exact_user_request']
 unchanged(q['protected_canonical_hashes']);assert not project_processes() and not runner_lock_owners();assert js(D/'PAUSE')['owner']=='A100_relative5pct_alignment_20261103_explicit_user_resume'
 import batch_910b as batch
 for row in rows:
  job=js(Path(row['home'])/'job.json');assert sha(Path(row['home'])/'job.json')==row['job_SHA256'] and job['spec']['npu_devices']==[row['card']];assert job['protected_canonical_hashes']==q['protected_canonical_hashes'];scope(job['spec'],job['stages'],False);batch.check_sources(job['spec']);assert batch.device_idle(job['spec'])
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);real_exclusive_owner(os.getpid(),D/'runner.lock');lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush();os.fsync(lock.fileno());children={};active={};completed={};grants={};handled=set()
 manifest=dict(coordinator_PID=os.getpid(),coordinator_start_ticks=proc_start(os.getpid()),coordinator_source=__file__,coordinator_source_SHA256=sha(__file__),worker_source_SHA256=sha(q['worker_source']),jobs={row['home']:row for row in rows},coordinator_failure_pending=str(home/'coordinator_failure_pending_natural_children.json'),one_original_global_exclusive_FLOCK=True,publishers_must_wait_for_all_workers_and_global_lock_release=True);mf=home/'coordinator_manifest.json';assert not mf.exists();put(mf,manifest)
 try:
  unchanged(q['protected_canonical_hashes'])
  for row in rows:
   path=Path(row['home']);out=(path/'parallel_worker_stdout.log').open('xb');err=(path/'parallel_worker_stderr.log').open('xb');p=subprocess.Popen(['python3','-B',q['worker_source'],row['home'],str(mf)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'));children[row['home']]=(p,out,err);put(path/'parallel_actual_launch.json',dict(UTC=now(),PID=p.pid,PPid=os.getpid(),card=row['card'],job_SHA256=row['job_SHA256'],coordinator_manifest=str(mf),coordinator_manifest_SHA256=sha(mf)))
  while len(completed)<len(rows):
   unchanged(q['protected_canonical_hashes']);real_exclusive_owner(os.getpid(),D/'runner.lock')
   for row in rows:
    name=row['home'];p,out,err=children[name];path=Path(name)
    if p.poll() is not None and name not in completed:
     completed[name]=dict(actual_returncode=p.wait(),UTC=now(),card=row['card']);active.pop(name,None);out.close();err.close()
    if name in active:
     count=grants[name]['count']
     if (path/('parallel_stage_'+str(count)+'_released.json')).exists():active.pop(name,None)
     elif (path/('parallel_stage_'+str(count)+'_NN_finished.json')).exists():active[name]=min(active[name],1024**3)
   for row in rows:
    name=row['home'];path=Path(name)
    if name in completed or name in active:continue
    for count in js(path/'job.json')['stages']:
     request=path/('parallel_stage_'+str(count)+'_request.json');key=(name,count)
     if not request.exists() or key in handled:continue
     req=js(request);assert req['job_SHA256']==row['job_SHA256'] and req['card']==row['card'] and req['worker_PID']==children[name][0].pid;free=shutil.disk_usage(D).free;need=req['budget_bytes']
     if can_admit(free,active,need,q['max_parallel_stages']):
      grant=dict(UTC=now(),coordinator_PID=os.getpid(),request_SHA256=sha(request),budget_bytes=need,count=count,card=row['card'],free_at_admission=free,other_active_reservations=dict(active));put(path/('parallel_stage_'+str(count)+'_grant.json'),grant);active[name]=need;grants[name]=grant;handled.add(key)
     elif not active and need>free:
      refusal=dict(UTC=now(),budget_bytes=need,actual_free_bytes=free,count=count,other_jobs_may_continue=True,no_model_inference_started_for_refused_stage=True);put(path/('parallel_stage_'+str(count)+'_refused.json'),refusal);put(home/'NOTIFY_capacity_error.json',dict(refusal,model_id=js(path/'job.json')['model_id'],chat_notification_not_claimed=True));handled.add(key)
     break
   current=dict(UTC=now(),PID=os.getpid(),phase='coordinating_real_disjoint_card_stages',active_reservations=active,grants=grants,completed=completed,jobs=rows,one_original_global_FLOCK_held=True,no_G_state_registry_or_workbook_write=True);put(home/'state.json',current);put(D/'A100_ALIGNMENT_PARALLEL_STATUS.json',dict(current,batch=str(home),manifest=str(mf)));time.sleep(.5)
  unchanged(q['protected_canonical_hashes']);rc=0 if all(x['actual_returncode']==0 for x in completed.values()) else 1;put(home/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,jobs=completed,no_G_or_canonical_write=True,all_actually_owned_workers_naturally_exited=True));return rc
 except BaseException:
  put(home/'coordinator_failure_pending_natural_children.json',dict(UTC=now(),error=traceback.format_exc(),no_signals_or_restarts=True))
  # No cancellation/signals: keep the real original lock while any owned stage
  # worker is still alive. A source/canonical failure remains an actual failure.
  for p,out,err in children.values():
   p.wait();out.close();err.close()
  put(home/'actual_failure.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),owned_children_all_naturally_exited=True));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
