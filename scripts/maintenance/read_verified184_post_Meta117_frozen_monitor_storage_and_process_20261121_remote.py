import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
M=R/'scripts/A100_alignment_incremental_monitor_20261103.py'
assert sha(M)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run([sys.executable,'-B',str(M)],cwd=R,env=env,text=True,capture_output=True)
out=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,frozen_source_unchanged=sha(M)=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d')
out['complete_latest_and_checkpoint']={str(p):json.loads(p.read_bytes()) for p in [D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json']}
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
out['actual_project_processes']=project_processes();out['actual_original_FLOCK']=runner_lock_owners()
out['actual_free_bytes']=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
out['canonical_SHA256']={str(D/n):sha(D/n) for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json','PAUSE','POLICY.json']}
st=json.loads((D/'state.json').read_bytes())
out['done_state_schema']={mid:{k:v for k,v in e.items() if any(w in k.lower() for w in ['qualif','formal','proof','protocol'])} for mid,e in st['models'].items() if e['status']=='done'}
du=subprocess.run(['du','-x','--max-depth=2',str(R/'workspace/diagnostics')],capture_output=True,text=True,timeout=45)
out['diagnostics_du']={'returncode':du.returncode,'stdout':du.stdout,'stderr':du.stderr}
smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
out['npu_smi']={'returncode':smi.returncode,'stdout':smi.stdout,'stderr':smi.stderr}
print(json.dumps(out,ensure_ascii=False))
