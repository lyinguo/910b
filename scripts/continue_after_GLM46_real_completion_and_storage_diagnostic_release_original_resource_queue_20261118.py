"""Wait without runner lock, then dispatch a newly eligible original queue stage.

No NN replay, G write, timeout changes, signals, disk substitution or fake run.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,time,fcntl,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 cfg=js(R/'configs/queues/existing_original_resources_immediately_after_GLM46_completed_own_true3_20261118.json');J=Path(cfg['parent_private_job']);H=Path(cfg['followup_home']);H.mkdir(exist_ok=True);guard=(H/'followup.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 def put(name,value):
  p=H/name;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(value,ensure_ascii=False,indent=2).encode());os.replace(t,p)
 try:
  while True:
   assert sha(J/'job.json')==cfg['parent_job_SHA256'];active=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];locks=runner_lock_owners()
   put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='waiting_for_actual_parent_completion_and_all_NN_NF_original_lock_release',parent_completion_exists=(J/'completion.json').exists(),actual_runner_FLOCK=locks,active_NN_NF=active,followup_holds_original_runner_lock=False))
   if (J/'completion.json').exists() and not active and not locks:break
   time.sleep(20)
  parent=js(J/'completion.json');put('parent_actual_completion.original.json',parent);selector=R/cfg['selector_source'];assert sha(selector)==cfg['selector_SHA256'];result=subprocess.run(['python3','-B',str(selector)],cwd='/root',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True);(H/'next_original_selector_stdout.json').write_text(result.stdout);(H/'next_original_selector_stderr.log').write_text(result.stderr);assert result.returncode==0
  selected=json.loads(result.stdout);put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,parent_actual_returncode=parent['actual_returncode'],selector_actual_returncode=0,actual_dispatch=selected.get('dispatch'),next_original_queue_result=selected,no_completed_phase_replayed=True,no_G_or_canonical_write=True))
  if not selected.get('dispatch'):put('NOTIFY_no_next_original_queue_stage.json',dict(reason='Actual existing original queue currently has no eligible dispatch; inspect exact resource/source rows and continue adapting other own-weight candidates',actual=selected,notification_file_only_not_chat_delivery=True))
 except BaseException:
  put('NOTIFY_followup_error.json',dict(error=traceback.format_exc(),notification_file_only_not_chat_delivery=True));raise
if __name__=='__main__':main()
