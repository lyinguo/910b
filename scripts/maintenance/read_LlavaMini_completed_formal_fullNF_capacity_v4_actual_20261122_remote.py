import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
H=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/LlavaMini_completed_formal_single_JSON_NF_acceptance_only_capacity_resume_v4_20261122'
x=dict(UTC=datetime.now(timezone.utc).isoformat(),home=str(H),original_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free,files={})
for n in ['state.json','formal_acceptance.json','actual_completion.json','immediate_next_original_resource_queue_actual.json','immediate_next_original_resource_queue_stdout.json','own_passed3_complete_all200_shape_capacity.json']:
 p=H/n
 if p.is_file():
  z=json.loads(p.read_bytes());x['files'][n]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),value=z)
for n in ['_launcher_stdout.log','_launcher_stderr.log']:
 p=Path(str(H)+n)
 if p.is_file():x[n]=p.read_text()
x['processes']=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
  if any('continue_LlavaMini_completed_formal_fullNF_only_capacity_v4' in a for a in args):x['processes'].append(dict(PID=int(p.name),argv=args,cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text(),io=(p/'io').read_text()))
 except (OSError,UnicodeError):pass
print(json.dumps(x,ensure_ascii=False))
