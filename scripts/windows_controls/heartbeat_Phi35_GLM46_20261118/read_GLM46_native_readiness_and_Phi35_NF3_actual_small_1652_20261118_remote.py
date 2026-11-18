import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
names=['GLM46Flash_own_original_BF16_thinking4096_native_source_readiness_v1_1645_20261118','GLM46Flash_own_native_readiness_v1_no_NN_actual_launch_1645_20261118','Phi35_completed_genuine_true3_formal_NF3_actual_profiler_core_schema_no_NN_replay_1642_20261118','Phi35_formal_NF3_completed_true3_full_shape_capacity_fresh3_fixed200_continuation_v3_1642_20261118']
rows={}
for name in names:
 H=B/name;files={}
 for n in ['readiness.json','actual_failure.json','actual_completion.json','state.json','stderr.log','full_shape_complete_numeric_storage_capacity_review.json']:
  p=H/n
  if not p.exists():continue
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except ValueError:v=raw.decode()
  if isinstance(v,dict) and 'capacity' in v:v=dict(v,capacity={k:v['capacity'][k] for k in ['empirical_estimate_bytes','actual_free_bytes','passed']})
  if n=='full_shape_complete_numeric_storage_capacity_review.json':v={k:v[k] for k in ['empirical_estimate_bytes','actual_free_bytes','passed','full_small_archive_bytes']}
  files[n]=dict(SHA256=hashlib.sha256(raw).hexdigest(),value=v)
 rows[name]=files
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),homes=rows,processes=project_processes(),locks=runner_lock_owners(),free=shutil.disk_usage(R).free),ensure_ascii=False))
