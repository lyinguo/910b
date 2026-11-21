import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
def rec(p):
 raw=p.read_bytes();v=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 try:v['value']=json.loads(raw)
 except ValueError:v['complete_text']=raw.decode(errors='replace')
 return v
ptr=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';p=json.loads(ptr.read_bytes());H=Path(p['handoff_directory']);waiter=B/'GLM41Thinking_completed_true3_verified183_exact271_wait_actual_original_capacity_first200_serial_v2_20261120';out=dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,actual_original_FLOCK=runner_lock_owners(),pointer=rec(ptr),closing=rec(H/'closing_addendum.json'),files={})
for home in [waiter,B/'GLM41Thinking_completed3_verified183_exact271_real_capacity_serial_v2_launch_20261120']:
 for n in ['state.json','actual_completion.json','exact_authorized_ledger_only_continuation_review.json','stderr.log','launch.json']:
  if (home/n).is_file():out['files'][str(home/n)]=rec(home/n)
print(json.dumps(out,ensure_ascii=False))
