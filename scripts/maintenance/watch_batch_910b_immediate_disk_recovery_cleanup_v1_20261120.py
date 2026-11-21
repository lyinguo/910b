"""Standing project disk recovery: react now, never interrupt formal NN/NF."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
CONFIG=R/'configs/maintenance/batch_910b_immediate_disk_recovery_cleanup_v3_20261120.json';H=D/'immediate_disk_recovery_cleanup_watch_20261120'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,x):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def pending_budgets(fresh_seconds):
 result=[];clock=datetime.now(timezone.utc)
 for p in B.glob('*/state.json'):
  try:
   # Ignore historical/dead controller states; actual live PID and fresh atomic state required.
   if p.stat().st_size>128*1024:continue
   x=js(p);phase=x.get('phase','');pid=x.get('PID');n=next((x[k] for k in ['needed_bytes','required_bytes'] if type(x.get(k)) is int and x[k]>0),None)
   if not n or 'wait' not in phase.lower() or type(pid) is not int:continue
   stamp=datetime.fromisoformat(x.get('UTC',''));age=(clock-stamp).total_seconds()
   if not 0<=age<=fresh_seconds:continue
   proc=Path('/proc')/str(pid);argv=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
   if not argv or not Path(argv[0]).name.startswith('python') or not any(s.startswith(str(R)+'/scripts/') for s in argv[1:]):continue
   result.append(dict(state=str(p),PID=pid,required_bytes=n,phase=phase,actual_argv=argv,state_age_seconds=age,actual_startticks=(proc/'stat').read_text().split(') ',1)[1].split()[19]))
  except (FileNotFoundError,ProcessLookupError,ValueError,KeyError,TypeError):continue
 return result
def signature():
 rows=[]
 for p in (R/'workspace/diagnostics/model_runs').iterdir():
  if p.is_dir() and not p.is_symlink():rows.append((p.name,p.stat().st_mtime_ns))
 return hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()
def main():
 from continue_meta_mllama_own_SDPA2048_true3_full_capacity_fresh3_fixed200_serial_v1_20261120 import quiet
 cfg=js(CONFIG);assert cfg['max_concurrent_models']==1 and cfg['physical_deduplication_keeps_all_original_paths_complete_bytes_and_SHA']
 for n,h in cfg['exact_source_SHA256'].items():assert sha(R/n)==h
 H.mkdir(exist_ok=True);guard=(H/'watch.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);poll=0;last_signature=None;last_attempt=0;last_result=None
 while True:
  poll+=1;free=shutil.disk_usage(R).free;idle=quiet();budgets=pending_budgets(cfg['active_state_max_age_seconds']) if idle else [];blocked=[x for x in budgets if free<x['required_bytes']];ready=[x for x in budgets if free>=x['required_bytes']]
  state=dict(UTC=now(),PID=os.getpid(),poll=poll,phase='observing',actual_free_bytes=free,original_NN_NF_and_global_EX_released=idle,active_resource_waiters=budgets,actual_NEW_model_forwards=0,no_model_signal_source_job_config_or_canonical_write=True,original_app_task_300min_unchanged=True,last_cleanup=last_result)
  if not idle:state['phase']='formal_NN_NF_or_original_EX_active_no_cleanup'
  elif ready:state['phase']='capacity_ready_prioritize_next_serial_model_no_cleanup_delay'
  elif not blocked and free>=cfg['proactive_free_target_bytes']:state['phase']='sufficient_actual_disk_no_cleanup_needed'
  else:
   required=min((x['required_bytes'] for x in blocked),default=cfg['proactive_free_target_bytes']);sig=signature()
   if sig==last_signature and time.monotonic()-last_attempt<cfg['no_new_values_retry_seconds']:state['phase']='no_new_values_since_last_attempt_backoff_not_first_block_delay'
   else:
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');run=H/('cleanup_'+stamp);run.mkdir();argv=[sys.executable,'-B',str(R/cfg['urgent_policy_consumer']),'--apply','--required-free-bytes',str(required)];put(run/'request.json',dict(UTC=now(),required_bytes=required,capacity_blocked_waiters=blocked,argv=argv,trigger='immediate_actual_capacity_block' if blocked else 'proactive_disk_low_watermark',no_new_unique_array_pruning=True))
    state.update(phase='immediate_safe_cleanup_running',cleanup_request=str(run/'request.json'));put(H/'state.json',state)
    with (run/'stdout.log').open('xb') as out,(run/'stderr.log').open('xb') as err:child=subprocess.run(argv,cwd=R,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
    last_result=dict(UTC=now(),actual_returncode=child.returncode,request=str(run/'request.json'),stdout=str(run/'stdout.log'),stderr=str(run/'stderr.log'),actual_free_after_bytes=shutil.disk_usage(R).free,chat_notification_not_claimed=True)
    put(run/'actual_completion.json',last_result);last_signature=signature();last_attempt=time.monotonic();state.update(phase='cleanup_finished_recheck_original_queue_immediately' if child.returncode==0 else 'cleanup_error_evidence_preserved_requires_context_review',last_cleanup=last_result,actual_free_bytes=shutil.disk_usage(R).free)
  put(H/'state.json',state)
  if '--once' in sys.argv:print(json.dumps(state,ensure_ascii=False));return
  time.sleep(cfg['poll_seconds'])
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put(H/'actual_failure.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),chat_notification_not_claimed=True,no_model_or_canonical_change=True))
  raise
