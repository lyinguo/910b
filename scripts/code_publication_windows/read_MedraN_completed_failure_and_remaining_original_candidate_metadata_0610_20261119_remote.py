"""Finite existing-stage and pending-state metadata only; never load weight arrays."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
o=dict(UTC=datetime.now(timezone.utc).isoformat(),stage_files={},pending_models=[])
for home,names in [
 ('MedraNE4B_own_original_FP16_native_v1_readiness_20261116',['readiness.json']),
 ('MedraNE4B_own_original_FP16_true3_v1_real_FLOCK_20261116',['completion.json','state.json','fresh3/verification_stdout.json','fresh3/verification_stderr.log','fresh3/model_stderr.log']),
 ('MedraNE4B_original_FP16_true3_capacity_fresh3_fixed200_continuation_v1_20261116',['actual_completion.json','state.json']),
 ('MedraNE4B_own_FP16_actual_vision_CPU_same_operands_diagnostic_20261116',['actual_completion.json','review.json','all548_own_vision_source_finiteness.json']),
]:
 for name in names:
  p=B/home/name;v=dict(path=str(p),exists=p.is_file())
  if p.is_file():
   b=p.read_bytes();v.update(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
   try:v['value']=json.loads(b)
   except ValueError:v['text']=b.decode('utf8',errors='replace')[-12000:]
  o['stage_files'][home+'/'+name]=v
s=json.loads((D/'state.json').read_bytes());o['state_top_level_keys']=list(s)
for mid,v in s['models'].items():
 if v.get('score') is not None or v.get('status') in ['missing_weights','skipped_missing_weights']:continue
 o['pending_models'].append(dict(model_id=mid,record=v))
print(json.dumps(o,ensure_ascii=False))
