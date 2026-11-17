import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
o={'UTC':datetime.now(timezone.utc).isoformat(),'procs':project_processes(),'FLOCK':runner_lock_owners(),'files':{}}
dirs=[B/'MervePali_own603_wait_Granite_delivery_release_then_new_numeric3_capacity_fresh3_fixed200_continuation_20261117',B/'MervePali_own603_original_native_2048_missing_full_numeric_true3_v1_real_FLOCK_20261117',B/'MervePali_own603_original_native_2048_approved_fresh3_fixed200_v1_real_FLOCK_20261117']
for d in dirs:
 for n in ['state.json','completion.json','actual_completion.json','NOTIFY_error.json','complete_true3_capacity_review.json','first200_measurement_approval.json']:
  p=d/n
  if p.exists():o['files'][str(p)]=json.loads(p.read_bytes())
 for name in ['smoke3','fresh3','formal200']:
  a=d/name
  if not a.exists():continue
  for p in a.iterdir():
   if p.is_file() and p.name in ['state.json','attempt.json','returncode.json','actual_completion.json','console.log','stderr.log','model_stdout.log','model_stderr.log','verification_stderr.log','verification_stdout.json']:
    raw=p.read_bytes()
    try:v=json.loads(raw)
    except Exception:v=raw[-12000:].decode(errors='replace')
    o['files'][str(p)]=v
  o['files'][str(a)+'/directory']=[p.name for p in a.iterdir()]
for p in o['procs']:
 pid=p.get('pid',p.get('PID'))
 if pid:
  x=Path('/proc')/str(pid)
  try:p.update(real_argv=(x/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),real_cwd=str((x/'cwd').resolve()),real_status=[v for v in (x/'status').read_text().splitlines() if v.startswith(('Name:','State:','Pid:','PPid:'))])
  except FileNotFoundError:p['exited_during_read']=True
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True);o['npu_smi']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(o,ensure_ascii=False))
