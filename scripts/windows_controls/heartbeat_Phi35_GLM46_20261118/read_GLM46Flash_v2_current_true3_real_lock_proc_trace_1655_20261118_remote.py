import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'GLM46Flash_own_ready_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v2_actual_D965_1653_20261118';J=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118';READY=B/'GLM46Flash_own_original_BF16_thinking4096_native_source_readiness_v2_actual_D965_1653_20261118';L=B/'GLM46Flash_actual_D965_readiness_continuation_v2_no_math_change_actual_launch_1653_20261118'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),free_bytes=shutil.disk_usage(R).free,files={})
for label,p in [(f'READY/{n}',READY/n) for n in ['readiness.json','actual_failure.json','actual_completion.json']]+[(f'H/{n}',H/n) for n in ['state.json','actual_completion.json']]+[(f'J/{n}',J/n) for n in ['job.json','state.json','completion.json','fresh3/verification_stderr.log','fresh3/verification_stdout.json','fresh3/console.log']]+[(f'L/{n}',L/n) for n in ['readiness_stderr.log','continuation_stderr.log']]:
 if not p.exists():continue
 raw=p.read_bytes();content=raw.decode('utf8',errors='replace')
 try:value=json.loads(content)
 except ValueError:value=content[-16000:]
 if label=='J/job.json':value=dict(model_id=value['model_id'],stages=value['stages'],config=value['spec']['config'],verifier=value['spec']['verifier'],source_SHA256=value['spec']['source_sha256'])
 out['files'][label]=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),value=value)
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
out['processes']=project_processes();out['actual_original_kernel_FLOCK']=runner_lock_owners()
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);out['npu_smi']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(out,ensure_ascii=False))
