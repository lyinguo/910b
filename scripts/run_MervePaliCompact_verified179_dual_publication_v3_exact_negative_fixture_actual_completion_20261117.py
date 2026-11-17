import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'MervePaliCompact_verified179_publication_review_inputs_v3_exact_negative_fixture_20261117';source=R/'scripts/publish_MervePaliCompact_verified179_dual_transaction_v3_exact_negative_fixture_20261117.py';review=I/'reviewed_inputs.json';assert hashlib.sha256(source.read_bytes()).hexdigest()=='96b8f91ecbd645f8a4e920e4209c9f840a90c6829c28436796fe2a6265ef4317' and hashlib.sha256(review.read_bytes()).hexdigest()=='6c140358fe6419a14c664d7a46f25f5fe395104e65e2f97caa3ae973bf05cff3';start=datetime.now(timezone.utc).isoformat()
with (I/'publication_stdout.json').open('xb') as out,(I/'publication_stderr.log').open('xb') as err:p=subprocess.run([sys.executable,'-B',str(source),str(review)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(started_UTC=start,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,source=str(source),source_SHA256='96b8f91ecbd645f8a4e920e4209c9f840a90c6829c28436796fe2a6265ef4317',reviewed_inputs_SHA256='6c140358fe6419a14c664d7a46f25f5fe395104e65e2f97caa3ae973bf05cff3',original_five_reports_and_model_sources_untouched=True);(I/'publication_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
