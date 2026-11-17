import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'CoderGemma4E4B_verified176_publication_review_inputs_v1_20261117'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=R/'scripts/publish_CoderGemma4E4B_verified176_single_lock_dual_transaction_v2_exact_loaded_source_reference_20261117.py';review=I/'reviewed_inputs.json';assert sha(source)=='fbd2e4d61e21df0ab369066486772ff9ecb6197a160ca920c552be0b0d5dcad3' and sha(review)=='5594bc8c733b896427ac15bf592e6df6a8bd36a18b85bbff7078988c38cffa23'
assert not (I/'publication_actual_completion_v1.json').exists();started=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout_v1.json').open('xb') as out,(I/'publication_stderr_v1.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=started,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='fbd2e4d61e21df0ab369066486772ff9ecb6197a160ca920c552be0b0d5dcad3',reviewed_inputs_SHA256='5594bc8c733b896427ac15bf592e6df6a8bd36a18b85bbff7078988c38cffa23',original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion_v1.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
