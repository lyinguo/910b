import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'DeadbydawnGemma4E2B_verified177_publication_review_inputs_v1_20261117'
source=R/'scripts/publish_DeadbydawnGemma4E2B_verified177_single_original_lock_dual_transaction_v2_finish_reference_scope_20261117.py';review=I/'reviewed_inputs.json';assert hashlib.sha256(source.read_bytes()).hexdigest()=='cda1648ff5cd99a7945678eb31f764ce690f338696ef73b751f9bf6e23fc06c9' and hashlib.sha256(review.read_bytes()).hexdigest()=='394c75ff46c8ff649213226ce5ce3e644261c8e8f73779d47f98539b02431cc1';assert not (I/'publication_actual_completion.json').exists();start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='cda1648ff5cd99a7945678eb31f764ce690f338696ef73b751f9bf6e23fc06c9',reviewed_inputs_SHA256='394c75ff46c8ff649213226ce5ce3e644261c8e8f73779d47f98539b02431cc1',original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
