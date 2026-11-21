import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120';READY=B/'GLM41Thinking_own_original_VLLM_BF16_8192_native_source_readiness_v1_20261120';J3=B/'GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120';J200=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
def read(p):
 if not p.is_file():return None
 raw=p.read_bytes();return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),complete_text=raw.decode(errors='replace'))
files={}
for home in [H,READY,J3,J200]:
 for name in ['state.json','actual_completion.json','actual_failure.json','readiness.json','completion.json','job.json','own_passed3_complete_shape_capacity_review.json']:
  files[str(home/name)]=read(home/name)
 for p in home.glob('*.log'):files[str(p)]=read(p)
 for stage in ['fresh3','fixed200']:
  for name in ['console.log','model_result.json','verifier_stdout.log','verifier_stderr.log','run_result.json']:
   p=home/stage/name
   if p.is_file():files[str(p)]=read(p)
procs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if not argv or not Path(argv[0]).name.startswith('python'):continue
  cwd=str((p/'cwd').resolve())
  if not (any(str(R) in x for x in argv) or Path(cwd).is_relative_to(R)):continue
  status=(p/'status').read_text();start=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
  procs.append(dict(PID=int(p.name),PPid=next(x.split(':',1)[1].strip() for x in status.splitlines() if x.startswith('PPid:')),argv=argv,cwd=cwd,startticks=start,io=(p/'io').read_text()))
 except FileNotFoundError:continue
captures=[]
for p in (R/'workspace/diagnostics/model_runs').glob('GLM41Thinking_*'):
 if not p.is_dir():continue
 rows=[];trace=p/'actual_sample_trace.jsonl'
 if trace.is_file():rows=[json.loads(s) for s in trace.read_text().splitlines()]
 failures={str(x):read(x) for x in p.glob('*failed.json')}
 captures.append(dict(directory=str(p),trace_metadata=read(trace),completed_samples=len(rows),committed_native_forwards=sum(r['actual_native_forwards'] for r in rows),last_sample=rows[-1] if rows else None,failures=failures))
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,actual_proc_including_rootcwd=procs,frozen_monitor_process_snapshot=project_processes(),actual_original_lock_owners=runner_lock_owners(),actual_capture_traces=captures,npu_smi=dict(returncode=info.returncode,stdout=info.stdout,stderr=info.stderr),actual_original_disk_free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))
