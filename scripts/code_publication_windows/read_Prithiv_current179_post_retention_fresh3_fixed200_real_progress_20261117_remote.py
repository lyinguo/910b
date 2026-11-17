import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os,json,subprocess,hashlib,shutil,re
from datetime import datetime,timezone
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117';H=B/'Prithiv_completed_card2_true3_user_verified179_retention_capacity_recovered_continuation_20261117'
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def read(p):return json.loads(p.read_bytes()) if p.exists() else None
state=read(J/'state.json');rows=[]
for p in project_processes():
 if p['role']!='other_project_python' or 'prithiv' in str(p).lower():
  z=dict(p);pid=p.get('PID',p.get('pid'));root=Path('/proc')/str(pid)
  if root.exists():
   try:z.update(actual_argv=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),actual_cwd=os.readlink(root/'cwd'),PPid=next(l for l in (root/'status').read_text().splitlines() if l.startswith('PPid:')))
   except OSError:pass
  rows.append(z)
reports=[];trace_progress=[];console=[];seen=set()
for stage in ['fresh3','formal200']:
 cp=J/stage/'console.log'
 if cp.exists():
  content=cp.read_text(errors='replace');console.append(dict(stage=stage,tail=content[-3500:]));folders=[Path(x) for x in re.findall(r'运行日志目录: ([^\r\n]+)',content)]
  for folder in folders:
   line=read(folder/'operator_lineage.json')
   if line:
    e=line.get('actual_execution',{});name=e.get('sample_trace');t=Path(name) if isinstance(name,str) else None
    if t and t.exists() and str(t) not in seen:
     seen.add(str(t));x=[json.loads(s) for s in t.read_text().splitlines()];trace_progress.append(dict(stage=stage,report_dir=str(folder),sample_trace=str(t),completed_samples=len(x),submitted_native_forwards=sum(i.get('actual_native_forwards',0) for i in x),devices=e.get('devices'),evidence_root=e.get('evidence_root')))
for row in rows:
 if row['role']!='model':continue
 for home in (R/'workspace/diagnostics/model_runs').glob('PrithivGemma4E4B_own_original_BF16_*_'+str(row['PID'])):
  t=home/'actual_sample_trace.jsonl'
  if t.exists() and str(t) not in seen:
   seen.add(str(t));x=[json.loads(s) for s in t.read_text().splitlines()];trace_progress.append(dict(stage=state['phase'],sample_trace=str(t),completed_samples=len(x),submitted_native_forwards=sum(i.get('actual_native_forwards',0) for i in x),devices=[2],evidence_root=str(home)))
result=dict(UTC=datetime.now(timezone.utc).isoformat(),state=state,completion=read(J/'completion.json'),controller_state=read(H/'state.json'),controller_completion=read(H/'actual_completion.json'),failure=read(H/'preparation_or_continuation_failure.json'),actual_project_processes=rows,kernel_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(D).free,trace_progress=trace_progress,console=console,npu_smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30).stdout)
print(json.dumps(result,ensure_ascii=False))
