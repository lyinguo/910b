"""Wait same own input/source preflight; start sole original private true3 after real guards."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,time,hashlib,subprocess,os,traceback,runpy,contextlib,io,shutil,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'AgentCPMGUI_own_type_import_recovery_same_unstarted_true3_v3_user_continue_20261116'
def now():return datetime.now(timezone.utc).isoformat()
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 import batch_910b as batch
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged
 H.mkdir(exist_ok=True);g=(H/'continuation.guard').open('a+');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists()
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_same_original_fullsource_frontend200_v2',no_NPU_inference_by_waiter=True))
 for script in ['prepare_agentcpmgui_own_native_fullsource_frontend200_v3_type_import_20261116.py']:
  with (H/(script+'.stdout.log')).open('xb') as out,(H/(script+'.stderr.log')).open('xb') as err:
   rc=subprocess.run(['python3','-B',str(R/'scripts'/script)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')).returncode
  assert rc==0,(script,'Original source/frontend preflight failed before NPU')

 pre=B/'AgentCPMGUI_own_original_fullsource_frontend200_v3_type_import_user_continue_20261116';until=time.monotonic()+14400
 while not (pre/'actual_completion.json').exists():
  assert time.monotonic()<until;time.sleep(10)
 assert js(pre/'actual_completion.json')['actual_returncode']==0
 publication=B/'MiniCPMV45_original_BF16_verified164_blank_G_publication_20261116/dual_publication_completed.json'
 while not publication.is_file() or runner_lock_owners() or [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]:
  assert time.monotonic()<until,'Wait for verified164 publication and original lock release timed out'
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_verified164_publication_then_original_lock_release',no_NPU_inference_by_waiter=True))
  time.sleep(5)
 assert js(publication)['completed']

 with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(R/'scripts/prepare_agentcpmgui_own_native_readiness_true3_v3_type_import_20261116.py'),run_name='__main__')
 ready=js(B/'AgentCPMGUI_own_original_native_readiness_v2_unpadded_vllm_user_continue_20261116/readiness.json');J=Path(ready['job']);job=js(J/'job.json')
 assert sha(J/'job.json')==ready['job_SHA256'] and not (J/'state.json').exists()
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert batch.device_idle(job['spec']) and shutil.disk_usage(D).free>12*1024**3
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner_stdout.log').open('xb') as out,(H/'runner_stderr.log').open('xb') as err:c=subprocess.Popen(['python3','-B',str(runner),str(J)],cwd=R,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_original_private_true3',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=c.pid,no_G_or_canonical_write=True))
 rc=c.wait();done=js(J/'completion.json');assert rc==done['actual_returncode']==0
 put('actual_completion.json',dict(UTC=now(),actual_returncode=0,true3_completion=done,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),original_evidence_and_G_preserved=True,disk_flag_is_not_chat_delivery=True))
  raise
