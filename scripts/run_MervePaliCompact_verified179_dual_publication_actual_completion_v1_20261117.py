import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'MervePaliCompact_verified179_publication_review_inputs_v1_20261117';source=R/'scripts/publish_MervePaliCompact_verified179_single_original_lock_dual_transaction_v1_20261117.py';review=I/'reviewed_inputs.json';assert hashlib.sha256(source.read_bytes()).hexdigest()=='0a2ceb8dbd126ee1ab5e8549e76a32700edcdbe337621f6abec6c3913d84935e' and hashlib.sha256(review.read_bytes()).hexdigest()=='b4131f2f1ad4db785fef55d4882917f4d2c8ed7784ddf884217825244461414e';start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='0a2ceb8dbd126ee1ab5e8549e76a32700edcdbe337621f6abec6c3913d84935e',reviewed_inputs_SHA256='b4131f2f1ad4db785fef55d4882917f4d2c8ed7784ddf884217825244461414e',original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
