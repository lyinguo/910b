import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';M=D/'incremental_monitor';p=R/'scripts/A100_alignment_incremental_monitor_20261103.py';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
run=subprocess.run(['python3','-B',str(p)],cwd=R,capture_output=True,text=True,timeout=65,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
files={}
for name in ['latest.json','checkpoint.json']:
 q=M/name;b=q.read_bytes();obj=json.loads(b);files[str(q)]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_JSON=obj)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,complete_latest_and_checkpoint=files,frozen_source_unchanged=hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()),ensure_ascii=False))
