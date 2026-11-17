import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'Granite41_verified178_publication_review_inputs_v1_20261117'
source=R/'scripts/publish_Granite41_verified178_single_original_lock_dual_transaction_v1_20261117.py';review=I/'reviewed_inputs.json';assert hashlib.sha256(source.read_bytes()).hexdigest()=='39819715c06c32954ae27e7a4c50b04205aed4bd5c239842d14202e264045069' and hashlib.sha256(review.read_bytes()).hexdigest()=='cf1c04a7f600593764c40987ae80cd4728deca5cad03efdd86cbbea92dd19bd2';assert not (I/'publication_actual_completion.json').exists();start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='39819715c06c32954ae27e7a4c50b04205aed4bd5c239842d14202e264045069',reviewed_inputs_SHA256='cf1c04a7f600593764c40987ae80cd4728deca5cad03efdd86cbbea92dd19bd2',original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
