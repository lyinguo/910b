import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,shutil
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
homes=[B/'Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v3_private_clean_imports_continuation_20261117',B/'Granite41_own_original_HF_BF16_4096_true3_v2_private_clean_imports_real_FLOCK_20261117',B/'Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v2_private_clean_imports_real_FLOCK_20261117'];files={}
for H in homes:
 for name in ['state.json','completion.json','actual_completion.json','first200_measurement_approval.json','fresh3/console.log','fresh3/verification_stderr.log','formal200/console.log','formal200/verification_stderr.log']:
  p=H/name
  if p.exists():
   raw=p.read_bytes();files[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),value=json.loads(raw) if p.suffix=='.json' else raw.decode('utf8',errors='replace').splitlines()[-18:])
traces=[]
for h in (R/'workspace/diagnostics/model_runs').glob('Granite41_own_HF_BF16_*'):
 p=h/'actual_sample_trace.jsonl'
 if p.exists():
  rows=[json.loads(s) for s in p.read_text().splitlines()];traces.append(dict(home=str(h),completed=len(rows),submitted_forwards=sum(x.get('actual_native_forwards',0) for x in rows),weight_proof=(h/'own1083_all_actual_NPU_sources.json').exists(),failure_files=[str(p) for p in h.glob('*failure.json')]))
q=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),procs=project_processes(),FLOCK=runner_lock_owners(),free=shutil.disk_usage(B).free,files=files,traces=traces,npu_smi_returncode=q.returncode,npu_smi=q.stdout),ensure_ascii=False))
