import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, hashlib, os
R = Path('/root/projects/comperror')
sys.path[:0] = [str(R), str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes, runner_lock_owners
D=R/'workspace/diagnostics/batch_910b'
B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121'
out={'H_exists':H.exists(),'free':os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,'processes':project_processes(),'lock_owners':runner_lock_owners()}
out['files']=[{'name':p.name,'bytes':p.stat().st_size} for p in H.iterdir() if p.is_file()] if H.exists() else []
for name in ['progress.json','plan.json','actual_completion.json']:
 p=H/name
 if p.exists():
  out[name]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'value':json.loads(p.read_bytes())}
print(json.dumps(out,ensure_ascii=False))
