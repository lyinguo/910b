import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from A100_alignment_evidence_runner_20261103 import unchanged
import batch_910b as batch
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=B/'DonutInvoice_own_original_FP16_poscap2048_native_v3_explicit_CPU_diagnostic_role_readiness_20261117';ready=json.loads((H/'readiness.json').read_bytes());assert ready['passed'] and ready['job_SHA256']=='60d079550f48419df9d05b1c4d60cf1a4ae08203c672be9bac4b7f33c6d9cea0'
J=Path(ready['true3_job']);assert sha(J/'job.json')==ready['job_SHA256'] and not (J/'state.json').exists() and not (J/'completion.json').exists();job=json.loads((J/'job.json').read_bytes());batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes']);assert batch.device_idle(job['spec'])
assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
I=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_actual_launch_20261117';I.mkdir(exist_ok=False);worker=R/'scripts/run_Donut_invoice_own_native_true3_v3_actual_20261117.py';assert not worker.exists()
s='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,subprocess,hashlib,json
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';I=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_actual_launch_20261117';J=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v3_explicit_CPU_diagnostic_role_real_FLOCK_20261117';runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
started=datetime.now(timezone.utc).isoformat()
with (I/'runner_stdout.log').open('xb') as out,(I/'runner_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',str(runner),str(J)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
v=dict(started_UTC=started,completed_UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,original_real_FLOCK_runner_used=True,no_G_or_canonical_write=True,actual_job_completion=json.loads((J/'completion.json').read_bytes()) if (J/'completion.json').exists() else None);(I/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
''';worker.write_bytes(s.encode())
with (I/'supervisor_stdout.log').open('xb') as out,(I/'supervisor_stderr.log').open('xb') as err:p=subprocess.Popen(['python3','-B',str(worker)],cwd=R,stdout=out,stderr=err,start_new_session=True,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,job=str(J),job_SHA256=sha(J/'job.json'),source=str(worker),source_SHA256=sha(worker),real_true3_dispatched_not_claimed_completed=True,original_runner_real_FLOCK_required=True);(I/'launch.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());print(json.dumps(v))
