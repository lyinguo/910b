import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json,hashlib
R=Path('/root/projects/comperror');H=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GLM46Flash_complete_bitplane_storage_diagnostic_actual_launch_for_next_serial_20261119'
source=R/'scripts/diagnostic_glm46_completed3_full_bitplane_exact_sharing_next_serial_20261119.py'
start=datetime.now(timezone.utc).isoformat()
with (H/'diagnostic_stdout.json').open('xb') as out,(H/'diagnostic_stderr.log').open('xb') as err:
 result=subprocess.run(['python3','-B',str(source)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=result.returncode,diagnostic_only_no_NEW_model_or_full_NF=True)
(H/'actual_completion.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');raise SystemExit(result.returncode)
