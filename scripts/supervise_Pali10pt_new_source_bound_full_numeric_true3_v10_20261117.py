"""Run one NEW source-bound full-numeric3 phase after exact readiness checks."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,traceback,fcntl
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Pali10Pt448_new_full_native_numeric_true3_v10_actual_supervision_20261117';J=B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117'
def put(n,v):(H/n).write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import batch_910b as batch
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 job=json.loads((J/'job.json').read_bytes());assert not (J/'state.json').exists() and not (J/'completion.json').exists();batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes'])
 live=[x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']];assert not live and not runner_lock_owners(),live;assert batch.device_idle(job['spec'])
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(runner),str(J)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=now(),phase='running_NEW_source_bound_full_native_numeric_true3',PID=os.getpid(),runner_PID=child.pid,job=str(J),job_SHA256=sha(J/'job.json'),no_G_write=True));rc=child.wait()
 final=json.loads((J/'completion.json').read_bytes());put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_job_completion=final,no_G_write=True));assert rc==final['actual_returncode']==0
 put('next_capacity_measurement_approval_required.json',dict(UTC=now(),no_formal200_approval_claimed=True,full200_requires_complete_all_shape_capacity_and_count200_numeric_implementation=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_write=True))
  raise
