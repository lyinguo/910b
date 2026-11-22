"""Read pending metadata only while the sole Hancom model continues; no learned work."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';sys.path[:0]=[str(R),str(R/'scripts')]
p=D/'state.json';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='47eb91c93d13fae449ec89fee6e22fd925c68066cb15c20293869737df1b00c3';x=json.loads(raw)
models=[]
for mid,row in x['models'].items():
 if row.get('status') not in ['done','skipped_no_complete_weights']:
  models.append(dict(model_id=mid,state=row))
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_Hancom_private_original_registration_v3_actual_serial_v4_all_proc_cwd_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),pending_metadata=models,current_serial_actual=json.loads(r.stdout),new_learned_or_NPU_model_started=False),ensure_ascii=False))
