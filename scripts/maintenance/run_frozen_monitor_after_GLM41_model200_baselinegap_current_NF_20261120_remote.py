import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
src=R/'scripts/A100_alignment_incremental_monitor_20261103.py'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
r=subprocess.run(['python3','-B',str(src)],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps(dict(actual_returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode()),ensure_ascii=False))
