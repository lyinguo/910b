import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119';H=B/'GLM46_own_passed3_after_Huihui_full_NF_baselinegap_verified_correct_nonexample_cleanup_serial_v2_20261119';sys.path[:0]=[str(R),str(R/'scripts')]
def maybe(p):
 if not p.is_file():return None
 try:return json.loads(p.read_bytes())
 except ValueError:return dict(partial_json=True,bytes=p.stat().st_size)
state=maybe(J/'state.json');completed=[];logs={};phases={}
for phase in ['fresh3','formal200']:
 home=J/phase
 if not home.exists():continue
 phases[phase]={n:maybe(home/n) for n in ['accepted.json','verification_stdout.json']}
 for f in home.glob('*.log'):
  size=f.stat().st_size
  assert size<=8*1024**2,('Read with complete incremental scanner instead',str(f),size)
  data=f.read_bytes();logs[str(f)]=dict(bytes=len(data),SHA256=hashlib.sha256(data).hexdigest(),complete_text=data.decode(errors='replace'))
folder=state.get('report_dir')
if folder and (Path(folder)/'operator_lineage.json').exists():
 p=Path(folder);r=maybe(p/'eval_report.json');line=maybe(p/'operator_lineage.json')['actual_execution'];trace=Path(line['sample_trace']);rows=[json.loads(x) for x in trace.read_text().splitlines()]
 completed.append(dict(report=str(p),five=list(x.name for x in p.iterdir()),reported_metrics=r['metrics'],statistics=r['statistics'],evidence_root=line['evidence_root'],actual_samples=len(rows),committed_actual_forwards=sum(x['actual_native_forwards'] for x in rows),model_rc=state.get('model_returncode'),reported_run_is_not_claimed_fullNF_until_accepted=True))
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
pid=state.get('child_PID');proc=Path('/proc')/str(pid);actual=dict(PID=pid,exists=proc.exists())
if proc.exists():
 actual.update(argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(proc/'cwd'),stat=(proc/'stat').read_text(),io=(proc/'io').read_text())
 fds=[]
 for f in (proc/'fd').iterdir():
  try:
   name=os.readlink(f)
   if any(x in name for x in ['sample_','safetensors','/mnt/afs']):fds.append(name)
  except OSError:pass
 actual['relevant_actual_FDs']=fds
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),state=state,controller_completion=maybe(H/'actual_completion.json'),job_completion=maybe(J/'completion.json'),phases=phases,current_report_completed_inference=completed,complete_new_attempt_logs=logs,actual_child=actual,actual_project_processes=project_processes(),actual_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))
