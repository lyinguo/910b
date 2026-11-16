import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'CoderGemma4E2B_verified173_publication_review_inputs_v1_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=R/'scripts/publish_CoderGemma4E2B_verified173_single_lock_dual_transaction_v1_20261116.py';review=I/'reviewed_inputs.json';assert sha(source)=='13c408a503b1090e60d57fcd4a2c5b76cc5b15b8712b76aaf78e684edf66c2ca' and sha(review)=='2f1ed8bb5fc1723394876cdfaf2919ecc1a4ae9eb5dc87da0d15bec7a5132acb'
assert not (I/'publication_actual_completion_v2.json').exists()
started=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout_v2.json').open('xb') as out,(I/'publication_stderr_v2.log').open('xb') as err:
 p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
receipt=dict(started_UTC=started,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='13c408a503b1090e60d57fcd4a2c5b76cc5b15b8712b76aaf78e684edf66c2ca',reviewed_inputs_SHA256='2f1ed8bb5fc1723394876cdfaf2919ecc1a4ae9eb5dc87da0d15bec7a5132acb',original_five_reports_and_model_sources_untouched=True)
(I/'publication_actual_completion_v2.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
raise SystemExit(p.returncode)
