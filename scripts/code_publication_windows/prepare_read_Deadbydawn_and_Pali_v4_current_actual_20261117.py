import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast
P=Path(__file__).parent
s=(P/'read_Deadbydawn_completed419_capacity_recovered_v3_current_actual_20261117_remote.py').read_text(encoding='utf8')
s=s.replace("'authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117']", "'authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117','Pali10Pt448_v4_named_RPC_after_current_Deadbydawn_real_lock_then_missing_full_native_value_true3_continuation_20261117','Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v4_named_worker_RPC_real_FLOCK_20261117','Pali10Pt448_v4_named_worker_RPC_after_Deadbydawn_immediate_queue_actual_launch_20261117','Pali10Pt448_missing_full_native_value_true3_v4_named_worker_RPC_readiness_20261117']")
s=s.replace("print(json.dumps(o,ensure_ascii=False))", """for h in (R/'workspace/diagnostics/model_runs').glob('pali10_pt448_native_vllm_*'):
 if h.stat().st_mtime<(B/'Pali10Pt448_missing_full_native_value_true3_v4_named_worker_RPC_readiness_20261117/readiness.json').stat().st_mtime-1:continue
 for n in ['requests.jsonl','full_numeric_group_index.json']:
  p=h/n
  if p.exists():
   raw=p.read_bytes()
   try:v=json.loads(raw)
   except Exception:v=[json.loads(x) for x in raw.splitlines()] if raw else []
   o.setdefault('Pali_actual_native_trace',[]).append(dict(path=str(p),SHA256=sha(p),bytes=len(raw),value=v))
print(json.dumps(o,ensure_ascii=False))""")
ast.parse(s)
(P/'read_Deadbydawn_and_Pali_v4_current_actual_20261117_remote.py').write_bytes(s.encode())
