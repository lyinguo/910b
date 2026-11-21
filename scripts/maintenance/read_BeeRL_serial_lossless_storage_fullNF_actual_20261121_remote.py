import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
H=B/'BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),files=[{'name':p.name,'bytes':p.stat().st_size} for p in H.iterdir() if p.is_file()] if H.exists() else [],original_FLOCK=runner_lock_owners(),project_processes=project_processes(),actual_free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize)
for n in ['state.json','actual_completion.json','complete_storage_all_values_review.json','complete_storage_fullNF_proof.json','capacity_review.json']:
 p=H/n
 if p.exists():out[n]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'value':json.loads(p.read_bytes())}
for p in [Path(str(H)+'_launcher_stderr.log'),H/'storage_fullNF_stderr.log']:
 if p.exists():out[str(p)]={'bytes':p.stat().st_size,'text':p.read_text()}
procs=[]
for p in out['project_processes']:
 if 'continue_BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121.py' not in ' '.join(p['argv']):continue
 q=Path('/proc')/str(p['PID']);p.update(stat=(q/'stat').read_text(),io=(q/'io').read_text(),FD=[{'fd':x.name,'target':os.readlink(x)} for x in (q/'fd').iterdir() if x.exists()]);procs.append(p)
out['actual_bound_RL']=procs
out['formal_fullNF_active']=bool(procs and (H/'storage_fullNF_stderr.log').exists() and not (H/'complete_storage_fullNF_proof.json').exists() and not (H/'actual_completion.json').exists() and any(x['PID']==procs[0]['PID'] for x in out['original_FLOCK']))
print(json.dumps(out,ensure_ascii=False))
