import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'AutoGLMPhone9B_verified182_reviewed_single_lock_dual_publication_20261120';source=R/'scripts/publish_AutoGLMPhone9B_verified182_original_single_lock_dual_transaction_v1_20261120.py';review=I/'reviewed_inputs.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='fab4bab195788d36959c6d5ac4ad39f8e170d2456772cd1727690b52928ea6e8' and hashlib.sha256(review.read_bytes()).hexdigest()==sys.argv[1]
start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:r=subprocess.run(['python3','-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=r.returncode,source=str(source),source_SHA256='fab4bab195788d36959c6d5ac4ad39f8e170d2456772cd1727690b52928ea6e8',reviewed_inputs_SHA256=sys.argv[1]);(I/'publication_actual_completion.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf8');raise SystemExit(r.returncode)
