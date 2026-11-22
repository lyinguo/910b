import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
def read(p):
 if not p.is_file():return None
 raw=p.read_bytes()
 try:v=json.loads(raw)
 except ValueError:v=raw.decode('utf8',errors='replace')
 return dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),complete_value=v)
J=B/'HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122';H=B/'HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122';M=B/'LlavaMore_own_original_probability_ready_after_Xtuner_actual_release_serial_v1_20261122';MJ=B/'LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122'
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_More_and_Hancom_own_original_native_serial_actual_v4_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
paths=[J/'completion.json',J/'fresh3/console.log',J/'fresh3/verification_stderr.log',H/'actual_completion.json',H/'own_genuine_NEWtrue3_fullNF_original_runner_active_stderr.log',M/'actual_completion.json',M/'own_passed3_complete_all200_shape_capacity.json',MJ/'completion.json',MJ/'fresh3/verification_stdout.json',MJ/'fresh3/verification_stderr.log']
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),current_full=json.loads(r.stdout),completed_model_and_failure_context={str(p):read(p) for p in paths}),ensure_ascii=False))
