import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def read(p):
 if not p.is_file():return None
 b=p.read_bytes()
 try:return json.loads(b)
 except Exception:return b.decode('utf-8',errors='replace')
H=B/'XtunerLlava3_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v2_IModel_20261122';READY=B/'XtunerLlava3_own_original_Transformers_FP16_2048_source_readiness_v2_IModel_20261122';J=B/'XtunerLlava3_own_original_Transformers_FP16_2048_true3_v2_IModel_real_FLOCK_20261122';I=B/'XtunerLlava3_own_true3_private_IModel_v2_controller_launcher_20261122'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),controller={str(p):read(p) for p in [H/'state.json',H/'actual_completion.json',H/'readiness_stderr.log',I/'stderr.log',I/'launch.json']},readiness={str(p):read(p) for p in [READY/'readiness.json',READY/'actual_completion.json',READY/'actual_failure.json']},job={str(p):read(p) for p in [J/'state.json',J/'completion.json']},captures=[])
if J.is_dir():
 out['attempt_files']={str(p):read(p) for p in J.rglob('*') if p.is_file() and p.name!='job.json' and p.stat().st_size<=8*1024**2 and p.suffix in ['.json','.log','.jsonl']}
for home in sorted((R/'workspace/diagnostics/model_runs').glob('XtunerLlava3_own_original_FP16_2048_*')):
 progress=home/'generation_progress.jsonl';records=[];partial=False
 if progress.is_file():
  lines=progress.read_bytes().splitlines(keepends=True)
  for i,line in enumerate(lines):
   if not line.endswith(b'\n') and i==len(lines)-1:partial=True;continue
   records.append(json.loads(line))
 counts={}
 for r in records:counts[str(r['index'])]=max(counts.get(str(r['index']),0),r['step'])
 out['captures'].append(dict(home=str(home),committed_step_counts=counts,total_committed_native_forwards=sum(counts.values()),last_committed_step=records[-1] if records else None,pending_partial_line=partial,committed_sample_execution=read(home/'sample_execution.jsonl'),NPZ=[dict(path=str(p),bytes=p.stat().st_size) for p in home.rglob('*.npz')]))
procs=project_processes();details=[]
for p in procs:
 v=dict(p);proc=Path('/proc')/str(p['PID'])
 try:v.update(stat=(proc/'stat').read_text(),io=(proc/'io').read_text(),fd={f.name:os.readlink(f) for f in (proc/'fd').iterdir()})
 except OSError as e:v['inspection_race']=str(e)
 details.append(v)
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);out.update(actual_processes=details,original_FLOCK=runner_lock_owners(),npu_smi=dict(returncode=smi.returncode,stdout=smi.stdout,stderr=smi.stderr),free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize)
print(json.dumps(out,ensure_ascii=False))
