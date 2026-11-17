import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
H=B/'Prithiv_completed_card2_true3_exact_alias_standard_Fortran_full_head_storage_diagnostic_only_20261117';L=B/'Prithiv_complete_NPY_lossless_storage_diagnostic_launch_control_20261117';files={}
for p in [H/'state.json',H/'actual_completion.json',L/'stderr.log',L/'stdout.log']:
 if p.exists():
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw.decode()
  files[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=v)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=files,actual_procs=project_processes(),actual_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(R).free,new_storage_files=[dict(name=p.name,bytes=p.stat().st_size) for p in H.iterdir()] if H.exists() else [],canonical_SHA256={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['state.json','registry.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json']}),ensure_ascii=False))
