import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'AutoGLMPhone9BSingle_verified183_reviewed_single_lock_dual_publication_20261120';source=R/'scripts/publish_AutoGLMPhone9BSingle_verified183_single_original_lock_dual_transaction_v1_20261120.py';review=H/'reviewed_inputs.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='504a20985debd6d956a37451e373683819677e86b5dc2aa56963b0b439842755' and hashlib.sha256(review.read_bytes()).hexdigest()==sys.argv[1]
start=datetime.now(timezone.utc).isoformat()
with (H/'publication_stdout.json').open('xb') as out,(H/'publication_stderr.log').open('xb') as err:r=subprocess.run(['python3','-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=r.returncode,source=str(source),source_SHA256='504a20985debd6d956a37451e373683819677e86b5dc2aa56963b0b439842755',reviewed_inputs_SHA256=sys.argv[1]);(H/'publication_actual_completion.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');raise SystemExit(r.returncode)
