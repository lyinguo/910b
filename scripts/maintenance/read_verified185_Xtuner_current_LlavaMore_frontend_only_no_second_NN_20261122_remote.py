import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'LlavaMore_own_original_FP16_discriminative_letter_pad_frontend200_meta_no_NN_v1_20261122';I=B/'LlavaMore_own_frontend200_meta_no_NN_v1_actual_launcher_20261122'
def read(p):
 if not p.is_file():return None
 b=p.read_bytes()
 try:return json.loads(b)
 except Exception:return b.decode('utf8',errors='replace')
r=subprocess.run(['python3','-B',str(R/'scripts/maintenance/read_XtunerLlava3_private_IModel_v2_fullNF_serial_current_20261122_remote.py')],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr
p=R/'scripts/prepare_LlavaMore_own_original_FP16_letter_pad_frontend200_meta_no_NN_v1_20261122.py'
out=dict(UTC=datetime.now(timezone.utc).isoformat(),current_Xtuner_same_serial_actual=json.loads(r.stdout),LlavaMore_frontend_only={str(p):read(p) for p in [H/'progress.json',H/'actual_completion.json',H/'readiness.json',I/'launch.json',I/'stderr.log']},actual_LlavaMore_frontend_source_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),old185_canonical_fresh_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json',R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx']})
print(json.dumps(out,ensure_ascii=False))
