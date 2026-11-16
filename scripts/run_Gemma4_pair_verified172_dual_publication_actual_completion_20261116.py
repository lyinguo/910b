import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,hashlib,json,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'Gemma4_pair_verified172_publication_review_inputs_v1_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=R/'scripts/publish_Gemma4_pair_verified172_single_lock_dual_transaction_v1_20261116.py';review=I/'reviewed_inputs.json';assert sha(source)=='bd5713f32285d3f19486b5fcde230ff4a47657ca3e9c12cc8bd077399ee61b44' and sha(review)=='5cf48d28e597b6eefba6a77a48b82da337f11abd1943bebc9c4f9a417166cf80'
assert not (I/'publication_actual_completion.json').exists()
started=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:
 p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
receipt=dict(started_UTC=started,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='bd5713f32285d3f19486b5fcde230ff4a47657ca3e9c12cc8bd077399ee61b44',reviewed_inputs_SHA256='5cf48d28e597b6eefba6a77a48b82da337f11abd1943bebc9c4f9a417166cf80',original_five_reports_and_model_sources_untouched=True)
(I/'publication_actual_completion.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
raise SystemExit(p.returncode)
