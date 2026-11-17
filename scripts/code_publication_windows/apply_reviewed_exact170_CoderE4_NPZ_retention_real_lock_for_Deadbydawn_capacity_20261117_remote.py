import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117';I=B/'single_CoderE4_verified176_retention_for_Deadbydawn_exact_inputs_actual_plan_launch_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
plan_sha='c4ea74b93a90e4f2246caba2590556ea46e86c7ffa2a4ea5609e74781f2b3ddb';assert sha(H/'plan.json')==plan_sha
assert sha(H/'all_current_foreign_evidence_metadata_review.json')=='aae9311389ac2b57f5e978cd451c36e2a3ff81e360b6cc9c5b66f9510aa2f55d'
p=json.loads((H/'plan.json').read_bytes());assert p['remove_count']==170 and p['remove_bytes']==9747351006 and {x['model_id'] for x in p['remove']}=={'coder3101/gemma-4-E4B-it-qat-q4_0-unquantized-heretic'} and {x['index'] for x in p['keep']}==set(range(10))|set(range(90,100))|set(range(190,200))
source=R/'scripts/retain_single_CoderE4_verified176_nonexample_matrices_for_Deadbydawn_capacity_20261117.py';assert sha(source)=='3c2f8906f3d8532cc17e4305d46bf9525bd9fd2b867563a21c140623697d3cb8'
assert not (H/'actual_completion.json').exists() and not (H/'removed_exact_original_paths_SHA256.jsonl').exists()
with (I/'apply_stdout.json').open('xb') as out,(I/'apply_stderr.log').open('xb') as err:r=subprocess.run(['python3','-B',str(source),'apply',plan_sha],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=r.returncode,stdout=(I/'apply_stdout.json').read_text(),stderr=(I/'apply_stderr.log').read_text(),actual_completion=json.loads((H/'actual_completion.json').read_bytes()) if (H/'actual_completion.json').exists() else None);(I/'apply_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v,ensure_ascii=False));raise SystemExit(r.returncode)
