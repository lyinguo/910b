import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
files={}
for H in sorted(B.glob('Granite41_*')):
 for n in ['state.json','actual_completion.json','completion.json','first200_measurement_approval.json','complete_true3_all200_shape_capacity_review.json','fresh3/console.log','fresh3/verification_stderr.log','formal200/console.log','formal200/verification_stderr.log']:
  p=H/n
  if p.exists():
   raw=p.read_bytes();value=json.loads(raw) if p.suffix=='.json' else raw.decode('utf8',errors='replace').splitlines()[-12:]
   if 'capacity_review' in n:value={k:v for k,v in value.items() if k not in ['full_all200_own_input_shapes','all_complete_small3_ZIP_parts']}
   files[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),value=value)
traces=[]
for H in sorted((R/'workspace/diagnostics/model_runs').glob('Granite41_own_HF_BF16_*')):
 p=H/'actual_sample_trace.jsonl'
 if p.exists():
  rows=[json.loads(s) for s in p.read_text().splitlines()];traces.append(dict(home=str(H),completed=len(rows),submitted_forwards=sum(x.get('actual_native_forwards',0) for x in rows),error_indices=[x['index'] for x in rows if x.get('error')],last_rows=rows[-1:],failure_files={str(f):json.loads(f.read_bytes()) for f in H.glob('*failure.json')},weights_source_proof_exists=(H/'own1083_all_actual_NPU_sources.json').exists()))
q=subprocess.run(['npu-smi','info'],text=True,capture_output=True)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),procs=project_processes(),FLOCK=runner_lock_owners(),free=shutil.disk_usage(D).free,files=files,traces=traces,npu_smi=q.stdout,npu_smi_returncode=q.returncode),ensure_ascii=False))
