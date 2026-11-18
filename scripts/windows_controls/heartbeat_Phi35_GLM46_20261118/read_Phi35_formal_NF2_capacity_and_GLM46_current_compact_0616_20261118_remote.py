import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
paths={
'Phi35_NF2':'Phi35_completed_genuine_true3_formal_NF2_original_BF16_RoPE_no_NN_replay_0616_20261118',
'Phi35_capacity':'Phi35_formal_NF2_completed_true3_full_shape_capacity_fresh3_fixed200_continuation_v2_0616_20261118',
'GLM46_frontend':'GLM46Flash_own_original_BF16_thinking4096_frontend200_meta_v1_0616_20261118'}
result=dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,homes={})
for key,name in paths.items():
 H=B/name;rows={}
 for rel in ['state.json','completion.json','actual_completion.json','fresh3/verification_stderr.log','progress.json','review.json','full_shape_complete_numeric_storage_capacity_review.json']:
  p=H/rel
  if not p.is_file():continue
  raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
  try:v=json.loads(raw)
  except ValueError:v=raw.decode('utf8')
  if rel=='full_shape_complete_numeric_storage_capacity_review.json':v={k:v[k] for k in ['passed','actual_free_bytes','empirical_estimate_bytes','full_small_archive_bytes','ratios','maximum_full200_shape_ratio']}
  rows[rel]=dict(path=str(p),bytes=len(raw),SHA256=h,value=v)
 if key=='Phi35_NF2':
  p=H/'fresh3/verification_stdout.json'
  if p.exists():
   raw=p.read_bytes();rows['proof_stat']=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
   try:
    v=json.loads(raw);rows['proof_summary']={k:v[k] for k in ['passed','sample_count','metrics','actual_forward_count','NPU_profiler_kernel_events','formal_independent_acceptance_phase_not_diagnostic_only','NEW_real_NPU_inference','actual_NEW_model_forwards_in_this_acceptance']};rows['proof_summary']['full_numeric_checks']=len(v['full_numeric_checks'])
   except ValueError:rows['unparsed_stdout']=raw.decode('utf8')
 result['homes'][key]=rows
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
result['actual_processes']=project_processes();result['actual_original_FLOCK']=runner_lock_owners()
print(json.dumps(result,ensure_ascii=False))
