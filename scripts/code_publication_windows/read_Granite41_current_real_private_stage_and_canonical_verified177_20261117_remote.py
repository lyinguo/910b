import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
homes=[B/'Granite41_after_Pali_actual_release_true3_shape_capacity_fresh3_fixed200_v2_no_legacy_startup_continuation_20261117',B/'Granite41_own_original_HF_BF16_4096_true3_v1_real_FLOCK_20261117',B/'Granite41_own_original_HF_BF16_4096_approved_fresh3_fixed200_v1_real_FLOCK_20261117'];files={}
for home in homes:
 for name in ['state.json','actual_completion.json','completion.json','NOTIFY_error.json','first200_measurement_approval.json','complete_true3_all200_shape_capacity_review.json','fresh3/console.log','fresh3/verification_stderr.log','formal200/console.log','formal200/verification_stderr.log']:
  p=home/name
  if not p.exists():continue
  raw=p.read_bytes();x=dict(SHA256=sha(p),bytes=len(raw))
  if p.suffix=='.json':
   z=json.loads(raw);x['value']=z if 'capacity_review' not in name else {k:v for k,v in z.items() if k not in ['full_all200_own_input_shapes','all_complete_small3_ZIP_parts']}
  else:x['tail']=raw.decode('utf8',errors='replace').splitlines()[-14:]
  files[str(p)]=x
traces=[]
for home in (R/'workspace/diagnostics/model_runs').glob('Granite41_own_HF_BF16_*'):
 trace=home/'actual_sample_trace.jsonl'
 if trace.exists():
  rows=[json.loads(s) for s in trace.read_text().splitlines()];traces.append(dict(home=str(home),completed=len(rows),submitted_native_forwards=sum(x['actual_native_forwards'] for x in rows),weight_proof_exists=(home/'own1083_all_actual_NPU_sources.json').exists(),failure_files=[str(p) for p in home.glob('*failure.json')]))
n=subprocess.run(['npu-smi','info'],capture_output=True,text=True);canonical=[D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),procs=project_processes(),FLOCK=runner_lock_owners(),free=shutil.disk_usage(D).free,files=files,traces=traces,npu_smi=dict(returncode=n.returncode,stdout=n.stdout,stderr=n.stderr),canonical_SHA256={str(p):sha(p) for p in canonical}),ensure_ascii=False))
