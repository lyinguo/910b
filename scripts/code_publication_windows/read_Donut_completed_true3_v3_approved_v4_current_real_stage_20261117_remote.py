import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
H=B/'DonutInvoice_completed_true3_v3_full_numeric_capacity_approved_fresh3_fixed200_v4_continuation_20261117';J=B/'DonutInvoice_own_original_FP16_poscap2048_approved_fresh3_fixed200_v4_real_FLOCK_20261117';I=B/'DonutInvoice_completed_true3_v3_first200_capacity_continuation_actual_launch_20261117'
o=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_procs=project_processes(),FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(D).free,files={},model_homes=[])
for home in [H,J,I,J/'fresh3',J/'formal200']:
 if not home.exists():continue
 for f in home.iterdir():
  if not f.is_file() or f.name=='continuation.guard':continue
  raw=f.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw[-12000:].decode(errors='replace')
  o['files'][str(f)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),value=v)
for h in sorted((R/'workspace/diagnostics/model_runs').glob('DonutInvoice_own_FP16_*')):
 f=h/'actual_sample_trace.jsonl';rows=[json.loads(s) for s in f.read_text().splitlines()] if f.exists() else []
 o['model_homes'].append(dict(home=str(h),completed_samples=len(rows),forwards=sum(e.get('actual_native_forwards',0) for e in rows),errors=[dict(index=e['index'],error=e['error']) for e in rows if e.get('error')],source484_loading_done=(h/'all484_own_source_CPUcast_actual_NPU_SHA256.json').exists(),report_proof=(h/'independent_report_verification.json').exists()))
print(json.dumps(o,ensure_ascii=False))
