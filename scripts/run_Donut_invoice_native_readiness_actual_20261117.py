import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'DonutInvoice_own_native_readiness_actual_launch_20261117'
with (I/'readiness_stdout.json').open('xb') as out,(I/'readiness_stderr.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(R/'scripts/prepare_Donut_invoice_own_original_FP16_native_readiness_true3_20261117.py')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
(I/'actual_completion.json').write_bytes(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,no_learned_CPU_or_NPU_forward=True,no_formal_score_or_qualification_claim=True),indent=2).encode());raise SystemExit(p.returncode)
