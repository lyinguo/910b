import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'OvisU1_completed3_complete_standard_Fortran_NPZ_storage_diagnostic_only_1916_20261117';L=B/'OvisU1_standard_Fortran_full_values_old3_diagnostic_launch_1916_20261117';js=lambda p:json.loads(Path(p).read_bytes());out={}
for p in [H/'state.json',H/'actual_completion.json',H/'storage_review_before_optional_full_numeric_recheck.json',H/'review.json',L/'stderr.log']:
 if p.exists():
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw.decode(errors='replace')[-3500:]
  if isinstance(v,dict) and 'capacity_review' in v:v={k:x for k,x in v.items() if k not in ['full_original_numeric_recheck']};v['capacity_review'].pop('all200_projected_complete_archives',None)
  out[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=v)
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=out,project_processes=project_processes(),actual_original_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(D).free,all_current_tasks_are_storage_diagnostic_not_new_NPU=True),ensure_ascii=False))
