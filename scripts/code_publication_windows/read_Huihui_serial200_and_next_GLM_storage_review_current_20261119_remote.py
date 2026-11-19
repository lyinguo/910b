import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
H=B/'GLM46Flash_completed3_complete_bitplane_exact_sharing_storage_for_immediate_serial_next_20261119'
I=B/'GLM46Flash_complete_bitplane_storage_diagnostic_actual_launch_for_next_serial_20261119'
def maybe(p):
 if not p.is_file():return None
 try:return json.loads(p.read_bytes())
 except ValueError:return dict(live_partial=True,bytes=p.stat().st_size)
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
procs=project_processes();locks=runner_lock_owners();samples=[];identities={}
for p in procs:
 if p['role']=='model':
  for home in (R/'workspace/diagnostics/model_runs').glob('HuihuiGLM46Flash_own_original_BF16_2048_*_'+str(p['PID'])):
   trace=home/'actual_sample_trace.jsonl';rows=[json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else []
   samples.append(dict(home=str(home),PID=p['PID'],samples=len(rows),forwards=sum(x['actual_native_forwards'] for x in rows),failure_records=[dict(path=str(f),value=maybe(f)) for f in home.glob('*_failed.json')]))
for row in locks:
 p=Path('/proc')/str(row['PID']);identities[str(row['PID'])]=dict(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text())
diag={n:maybe(H/n) for n in ['state.json','initial_private_copy_budget.json','actual_failure.json','review.json','actual_completion.json']}
diag['supervisor_completion']=maybe(I/'actual_completion.json')
stderr=I/'diagnostic_stderr.log'
if stderr.is_file():diag['stderr_tail']=stderr.read_text(errors='replace')[-5000:]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_NN_NF=procs,original_FLOCK=locks,lock_actual_identity=identities,current_samples=samples,Huihui_job_state=maybe(J/'state.json'),Huihui_completion=maybe(J/'completion.json'),Huihui_formal_accepted=maybe(J/'formal200/accepted.json'),next_GLM_storage_only=diag,free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))
