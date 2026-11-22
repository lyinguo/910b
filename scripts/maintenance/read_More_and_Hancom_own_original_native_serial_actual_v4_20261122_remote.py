"""Inspect immutable current More and own next Hancom native serial controller."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_More_actual_serial_Xtuner_completed3_Hancom_own_front_CPU_no_NN_v3_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
x=json.loads(r.stdout);files={}
for stem in ['HancomHiLlava31_own_original_Transformers_FP16_true3_capacity_fresh3_fixed200_serial_v1_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_source_readiness_v1_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122','HancomHiLlava31_own_native_serial_after_More_v1_actual_launcher_20261122']:
 for name in ['state.json','readiness.json','job.json','actual_completion.json','actual_failure.json','completion.json','launch.json','stderr.log','readiness_stdout.log','readiness_stderr.log']:
  p=B/stem/name
  if p.is_file():
   raw=p.read_bytes()
   try:value=json.loads(raw)
   except ValueError:value=raw.decode('utf8',errors='replace')
   files[str(p)]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),complete_value=value)
x['UTC']=datetime.now(timezone.utc).isoformat();x['Hancom_own_original_native_serial_actual']=files
print(json.dumps(x,ensure_ascii=False))
