import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
H=B/'Pali10Pt448_existing_complete_real3_v9_original_async_EOS_commit_full_numeric_diagnostic_original_FLOCK_20261117';v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_procs=project_processes(),FLOCK=runner_lock_owners(),files={})
for p in H.iterdir():
 if not p.is_file() or p.suffix not in ['.json','.log']:continue
 raw=p.read_bytes()
 try:x=json.loads(raw)
 except Exception:x=raw[-7500:].decode(errors='replace')
 v['files'][str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),value=x)
print(json.dumps(v,ensure_ascii=False))
