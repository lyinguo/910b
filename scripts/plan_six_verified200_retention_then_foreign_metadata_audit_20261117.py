import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os,traceback
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'six_verified200_retention_plan_then_foreign_metadata_audit_20261117';P=B/'authorized_six_complete_verified_runs_nonexample_retention_v3_shared_examples_protected_for_CoderE4_capacity_20261117';H.mkdir(exist_ok=False)
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='plan_full_protected_SHA_no_deletion'))
 with (H/'plan_stdout.json').open('xb') as o,(H/'plan_stderr.log').open('xb') as e:
  rr=subprocess.run([sys.executable,'-B','scripts/retain_six_complete_verified_runs_nonexample_matrices_for_CoderE4_capacity_v1_20261117.py','plan','/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/authorized_six_complete_verified_runs_retention_reviewed_inputs_for_CoderE4_20261117/reviewed_inputs.json','ed02357412ec0b5a4a38160ea2538471c2aebb2f8f06654a054e614dbf013368'],cwd=R,stdout=o,stderr=e,env=ENV)
 assert rr.returncode==0
 plan=sha(P/'plan.json');put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),phase='all_verified175_foreign_metadata_audit_no_deletion',plan_SHA256=plan))
 with (H/'foreign_audit_stdout.json').open('xb') as o,(H/'foreign_audit_stderr.log').open('xb') as e:
  rr=subprocess.run([sys.executable,'-B','scripts/audit_six_retention_all_verified175_foreign_metadata_20261117.py',plan],cwd=R,stdout=o,stderr=e,env=ENV)
 assert rr.returncode==0
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=0,plan_SHA256=plan,foreign_review_SHA256=sha(P/'all_current_foreign_evidence_metadata_review.json'),no_deletion_performed=True))
except BaseException:
 put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_deletion_performed=True));raise
