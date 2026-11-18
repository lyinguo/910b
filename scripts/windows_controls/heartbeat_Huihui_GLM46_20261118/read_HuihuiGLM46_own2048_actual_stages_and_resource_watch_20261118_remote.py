import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
names=['HuihuiGLM46Flash_own_original2048_frontend_readiness_true3_automatic_continuation_20261118','HuihuiGLM46Flash_own_original_BF16_2048_frontend200_meta_v1_20261118','HuihuiGLM46Flash_own_original_BF16_2048_native_source_readiness_v1_20261118','HuihuiGLM46Flash_own_original_BF16_2048_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_20261118','HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118','HuihuiGLM46Flash_own_original_BF16_2048_approved_fresh3_fixed200_v1_real_FLOCK_20261118','GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118']
homes={};pids=set()
for name in names:
 h=B/name;items={}
 for fn in ['state.json','progress.json','review.json','readiness.json','actual_completion.json','actual_failure.json','completion.json','full_shape_complete_numeric_storage_capacity_review.json']:
  p=h/fn
  if p.is_file():
   b=p.read_bytes();v=json.loads(b);items[fn]=dict(SHA256=hashlib.sha256(b).hexdigest(),value=v)
   for k in ['PID','child_PID','runner_PID','model_PID','verifier_PID']:
    if isinstance(v.get(k),int):pids.add(v[k])
 if h.is_dir():
  items['recent_logs']={}
  for p in sorted(h.glob('*.log')):
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-24000));b=f.read()
   items['recent_logs'][p.name]=dict(bytes=p.stat().st_size,tail=b.decode('utf8',errors='replace'))
 homes[name]=items
procs=project_processes();actual=[]
for pr in procs:
 if pr['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']:actual.append(pr)
 if 'huihui_glm46' in str(pr).lower() or 'HuihuiGLM46' in str(pr):
  for k in ['pid','PID']:
   if isinstance(pr.get(k),int):pids.add(pr[k])
details={}
for pid in sorted(pids):
 p=Path('/proc')/str(pid)
 if p.is_dir():
  try:details[pid]=dict(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),stat=(p/'stat').read_text(),cwd=os.readlink(p/'cwd'),io=(p/'io').read_text())
  except OSError as e:details[pid]=dict(read_error=str(e))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),homes=homes,actual_project_NN_NF=actual,actual_original_FLOCK=runner_lock_owners(),actual_proc_details=details,free_bytes=shutil.disk_usage(R).free),ensure_ascii=False))
