import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
o=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_procs=project_processes(),FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(D).free,files={},native_progress=[])
for name in ['DeadbydawnGemma4E2B_completed_true3_419_exact_retention_capacity_recovered_approved_fresh3_fixed200_v3_continuation_20261117','DeadbydawnGemma4E2B_completed_true3_419_capacity_recovered_approved_fresh3_fixed200_v3_real_FLOCK_20261117','DeadbydawnGemma4E2B_completed419_exact170_retention_recovery_controller_v3_actual_launch_20261117','authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117']:
 h=B/name
 if not h.exists():continue
 for sub in ['', 'fresh3','formal200']:
  f=h/sub
  if not f.is_dir():continue
  for p in f.iterdir():
   if not p.is_file() or p.suffix not in ['.json','.log'] or p.stat().st_size>8*1024*1024 or p.name=='plan.json':continue
   raw=p.read_bytes()
   try:v=json.loads(raw)
   except Exception:v=raw[-12000:].decode(errors='replace')
   o['files'][str(p)]=dict(SHA256=sha(p),bytes=len(raw),value=v)
for h in (R/'workspace/diagnostics/model_runs').glob('DeadbydawnGemma4E2B_own_original_BF16_*'):
 t=h/'actual_sample_trace.jsonl'
 if t.exists():
  r=[json.loads(s) for s in t.read_text().splitlines()];o['native_progress'].append(dict(home=str(h),completed_samples=len(r),actual_submitted_forwards=sum(x.get('actual_native_forwards',0) for x in r),error_indices=[x['index'] for x in r if x.get('error')]))
for x in o['actual_procs']:
 if x['role'] not in ['model','alignment_runner','CPU_verifier_or_audit']:continue
 p=Path('/proc')/str(x['PID'])
 try:o.setdefault('actual_process_IO_FD',[]).append(dict(PID=x['PID'],io=(p/'io').read_text(),project_FD_targets=[str(f.resolve()) for f in (p/'fd').iterdir() if str(f.resolve()).startswith(str(R))][-20:]))
 except OSError as e:o.setdefault('natural_process_races',[]).append(str(e))
s=subprocess.run(['npu-smi','info'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);o['npu_smi']=dict(actual_returncode=s.returncode,stdout=s.stdout,stderr=s.stderr)
print(json.dumps(o,ensure_ascii=False))
