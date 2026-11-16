import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'RedHatGemma4E4B_verified174_publication_review_inputs_v1_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=R/'scripts/publish_RedHatGemma4E4B_verified174_single_lock_dual_transaction_v1_20261116.py';review=I/'reviewed_inputs.json';assert sha(source)=='455fbadb1a3bb260971c2ab874551ba2e718dfe77797d8e60b4c37f0ac524955' and sha(review)=='5177ef36cd35e092c1099b96e417bf85d49f0bbdd00c17f60f39fdc1ffedfb59'
assert not (I/'publication_actual_completion_v1.json').exists()
started=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout_v1.json').open('xb') as out,(I/'publication_stderr_v1.log').open('xb') as err:
 p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
receipt=dict(started_UTC=started,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='455fbadb1a3bb260971c2ab874551ba2e718dfe77797d8e60b4c37f0ac524955',reviewed_inputs_SHA256='5177ef36cd35e092c1099b96e417bf85d49f0bbdd00c17f60f39fdc1ffedfb59',original_five_reports_and_model_sources_untouched=True)
(I/'publication_actual_completion_v1.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
raise SystemExit(p.returncode)
