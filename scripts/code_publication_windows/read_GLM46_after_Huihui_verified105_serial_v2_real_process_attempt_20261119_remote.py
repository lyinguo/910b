import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'GLM46_own_passed3_after_Huihui_full_NF_baselinegap_verified_correct_nonexample_cleanup_serial_v2_20261119';J=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119';I=B/'GLM46_after_Huihui_gap_verified105_cleanup_serial_v2_actual_launch_20261119'
def maybe(p):
 if not p.is_file():return None
 try:return json.loads(p.read_bytes())
 except ValueError:return dict(partial_JSON=True,bytes=p.stat().st_size)
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
procs=project_processes();locks=runner_lock_owners();details={}
for row in procs+locks:
 p=Path('/proc')/str(row['PID'])
 if not p.exists():continue
 try:details[row['PID']]=dict(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text())
 except OSError:pass
progress=[]
for row in procs:
 if row['role']!='model':continue
 # Use this model's actual unchanged native capture prefix, proved by its
 # live sample NPZ FD; the earlier reader's guessed prefix matched no home.
 for home in (R/'workspace/diagnostics/model_runs').glob('GLM46Flash_own_BF16_thinking_*_'+str(row['PID'])):
  trace=home/'actual_sample_trace.jsonl';records=[json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else [];progress.append(dict(home=str(home),PID=row['PID'],sample_count=len(records),committed_actual_forwards=sum(x['actual_native_forwards'] for x in records),failures=[dict(path=str(f),value=maybe(f)) for f in home.glob('*_failed.json')]))
states={n:maybe(p) for n,p in {'launch':I/'actual_launch.json','controller_state':H/'state.json','controller_actual_completion':H/'actual_completion.json','capacity':H/'fresh_complete_component_capacity_review.json','approval':H/'first200_measurement_approval.json','job_state':J/'state.json','job_completion':J/'completion.json'}.items()}
if (J/'job.json').exists():states['job_SHA256']=hashlib.sha256((J/'job.json').read_bytes()).hexdigest();states['job_source_SHA256']=maybe(J/'job.json')['spec']['source_sha256']
logs={str(f):f.read_text(errors='replace')[-7000:] for root in [I,H,J/'fresh3',J/'formal200'] if root.exists() for f in root.glob('*.log') if f.is_file()}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),states=states,actual_procs=procs,actual_FLOCK=locks,actual_proc_details=details,real_progress=progress,logs=logs,free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))
