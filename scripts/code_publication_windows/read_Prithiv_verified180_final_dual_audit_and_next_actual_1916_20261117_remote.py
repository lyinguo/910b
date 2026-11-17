import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,shutil
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117';I=B/'PrithivGemma4E4B_verified180_publication_review_inputs_v1_20261117';out={}
for p in [I/'publication_actual_completion.json',I/'publication_stderr.log',H/'transaction.json',H/'dual_publication_completed.json',H/'full_target_dual_publication_audit.json',H/'verified180_export_complete_metadata.json']:
 if p.exists():
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw.decode(errors='replace')
  out[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=v)
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),files=out,actual_project_processes=project_processes(),FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(D).free),ensure_ascii=False))
