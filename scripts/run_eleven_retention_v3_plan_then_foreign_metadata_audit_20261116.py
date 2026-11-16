import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'authorized_eleven_complete_verified_runs_retention_reviewed_input_for_RedHatE2B_20261116';H=B/'authorized_eleven_complete_verified_runs_nonexample_retention_v3_shared_examples_protected_for_RedHatE2B_capacity_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def record(v):(I/'v3_plan_and_foreign_audit_actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode())
inp=I/'reviewed_inputs.json';assert sha(inp)=='f2f795f8a582e72954b52761de5eecadf237891d2683b619518f51ce5bb01706';assert not H.exists() and not (I/'v3_plan_and_foreign_audit_actual_completion.json').exists()
rc=None
with (I/'v3_plan_stdout.json').open('xb') as out,(I/'v3_plan_stderr.log').open('xb') as err:r=subprocess.run(['python3','-B',str(R/'scripts/retain_eleven_complete_verified_runs_nonexample_matrices_for_RedHatE2B_capacity_v3_shared_examples_protected_20261116.py'),'plan',str(inp),sha(inp)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
if r.returncode!=0:record(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=r.returncode,phase='plan_failed_before_any_deletion',no_deletion=True));sys.exit(r.returncode)
digest=sha(H/'plan.json')
with (I/'v3_foreign_metadata_stdout.json').open('xb') as out,(I/'v3_foreign_metadata_stderr.log').open('xb') as err:r=subprocess.run(['python3','-B',str(R/'scripts/audit_eleven_retention_v3_all_current_accepted_foreign_metadata_20261116.py'),digest],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
record(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=r.returncode,phase='plan_and_foreign_metadata_audit_completed',plan_SHA256=digest,no_deletion=True));sys.exit(r.returncode)
