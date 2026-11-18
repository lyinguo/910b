import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
H=B/'GLM46Flash_completed_true3_wait_actual_original_disk_capacity_then_approved_fresh3_fixed200_v3_1723_20261118';s=js(H/'state.json');pid=s['PID'];proc={}
for name in ['stat','status','cmdline','io']:
 p=Path(f'/proc/{pid}/{name}')
 if p.exists():proc[name]=p.read_bytes().decode(errors='replace').replace('\x00',' ')
if proc:proc['cwd']=os.readlink(f'/proc/{pid}/cwd')
source=R/'scripts/continue_glm46flash_completed_true3_wait_actual_capacity_resume_fresh3_fixed200_v3_1723_20261118.py';assert sha(source)=='96dfdd09b7b14a4845e142cbcc6dcc4384130a362ff1bea777f9fcf2a059637b'
selector=R/'scripts/select_existing_original_resource_queue_1245_20261117.py';assert sha(selector)=='27daea622ea40c92d1182895e6dcd1a9d5803fac9cc0b7a7d224a4f80dd52494'
r=subprocess.run(['python3','-B',str(selector)],cwd='/root',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),resource_state=s,actual_resource_proc=proc,resource_completion=js(H/'actual_completion.json') if (H/'actual_completion.json').exists() else None,source_SHA256=sha(source),next_fixed200_job_exists=(B/'GLM46Flash_completed_true3_actual_capacity_restored_approved_fresh3_fixed200_v3_real_FLOCK_1723_20261118/job.json').exists(),free_bytes=shutil.disk_usage(R).free,selector=dict(actual_returncode=r.returncode,stderr=r.stderr,value=json.loads(r.stdout) if r.returncode==0 else r.stdout),actual_project_processes=project_processes(),actual_original_FLOCK=runner_lock_owners()),ensure_ascii=False))
