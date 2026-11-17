import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,subprocess,hashlib,json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_actual_launch_20261117';J=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_explicit_CPU_diagnostic_role_real_FLOCK_20261117';runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
started=datetime.now(timezone.utc).isoformat()
with (I/'runner_stdout.log').open('xb') as out,(I/'runner_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',str(runner),str(J)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(started_UTC=started,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,original_real_FLOCK_runner_used=True,no_G_or_canonical_write=True,actual_job_completion=json.loads((J/'completion.json').read_bytes()) if (J/'completion.json').exists() else None);(I/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
