import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
pointer=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');out=dict(UTC=datetime.now(timezone.utc).isoformat(),pointer=pointer,pointer_SHA256=sha(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'),files={},canonical_SHA256={p:sha(p) for p in pointer['canonical_SHA256']},free_bytes=shutil.disk_usage(R).free)
homes=['GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118','GLM46Flash_own_ready_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v2_actual_D965_1653_20261118','GLM46Flash_completed_true3_complete_standard_Fortran_LZMA_storage_capacity_diagnostic_only_1714_20261118','after_GLM46_completed_own_true3_and_all_real_lock_release_immediate_original_resource_queue_20261118']
for name in homes:
 row={};H=B/name
 for n in ['completion.json','actual_completion.json','review.json','state.json','NOTIFY_no_next_original_queue_stage.json','next_original_selector_stdout.json','next_original_selector_stderr.log']:
  p=H/n
  if not p.exists():continue
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except ValueError:v=raw.decode()
  if isinstance(v,dict) and 'capacity' in v:v=dict(v,capacity={k:v['capacity'][k] for k in ['empirical_estimate_bytes','actual_free_bytes','passed']})
  if isinstance(v,dict) and 'all_original_protected_SHA256' in v:v={k:q for k,q in v.items() if k!='all_original_protected_SHA256'}
  row[n]=dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),value=v)
 out['files'][name]=row
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
out.update(actual_processes=project_processes(),actual_original_FLOCK=runner_lock_owners());print(json.dumps(out,ensure_ascii=False))
