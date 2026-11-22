import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'Gemma4E4BMLX8_verified185_publication_review_inputs_v1_20261122'
source=R/'scripts/publish_Gemma4E4BMLX8_verified185_single_original_lock_dual_transaction_v1_20261122.py';review=I/'reviewed_inputs.json';assert hashlib.sha256(source.read_bytes()).hexdigest()=='567f27b07fde53904eb6916f7519a8a5980a7ee6078967743361e22fd3f6ed77' and hashlib.sha256(review.read_bytes()).hexdigest()=='5838611ddbc303b98e81c631a9291e222e04d94aeda4af468e77bc4dc9969a24';assert not (I/'publication_actual_completion.json').exists();start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='567f27b07fde53904eb6916f7519a8a5980a7ee6078967743361e22fd3f6ed77',reviewed_inputs_SHA256='5838611ddbc303b98e81c631a9291e222e04d94aeda4af468e77bc4dc9969a24',original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
