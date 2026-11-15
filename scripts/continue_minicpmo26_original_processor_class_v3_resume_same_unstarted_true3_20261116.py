"""Own MiniCPMO image task preflight then unique real true3 after prior job/publication."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,subprocess,time,traceback,hashlib,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMO26_own_image_original_processor_class_v3_resume_same_unstarted_true3_20261116'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 for script in ['prepare_minicpmo26_own_native_fullsource_frontend200_v3_original_processor_class_20261116.py']:
  with (H/(script+'.stdout.log')).open('xb') as out,(H/(script+'.stderr.log')).open('xb') as err:
   c=subprocess.Popen(['python3','-B',str(R/'scripts'/script)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='own_original_source_or_frontend_preflight_not_NPU',stage=script,child_PID=c.pid));rc=c.wait();assert rc==0,(script,rc)
 marker=B/'AgentCPMGUI_verified_outcome_publication_or_outside_review_20261116/completed.json';until=time.monotonic()+86400
 while not marker.is_file() or runner_lock_owners() or [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]:
  assert time.monotonic()<until,'Prior task actual release/publication timed out'
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_prior_true200_numeric_and_qualified_publication_before_next_true3',no_NPU_inference_by_waiter=True));time.sleep(5)
 assert json.loads(marker.read_bytes())['completed']
 import shutil
 assert shutil.disk_usage(D).free>=12*1024**3,'Insufficient original project disk; preserve evidence and release without dispatch'
 script='prepare_minicpmo26_own_native_readiness_true3_v3_original_processor_class_20261116.py'
 with (H/(script+'.stdout.log')).open('xb') as out,(H/(script+'.stderr.log')).open('xb') as err:
  c=subprocess.Popen(['python3','-B',str(R/'scripts'/script)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
 assert c.wait()==0
 J=B/'MiniCPMO26_own_original_BF16_4096_true3_v2_real_FLOCK_user_continue_20261116';assert not (J/'state.json').exists()
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner_stdout.log').open('xb') as out,(H/'runner_stderr.log').open('xb') as err:
  c=subprocess.Popen(['python3','-B',str(runner),str(J)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_own_image_task_original_BF16_4096_true3',child_PID=c.pid,job=str(J),job_SHA256=sha(J/'job.json')))
 rc=c.wait();assert rc==0;put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
