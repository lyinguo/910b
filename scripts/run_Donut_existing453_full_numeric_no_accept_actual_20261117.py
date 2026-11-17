import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,fcntl,subprocess,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'DonutInvoice_failed_runner_453_completed_NPU3_full_value_readonly_no_accept_numeric_diagnostic_v2_20261117';folder=R/'workspace/reports/katanaml-org_invoices-donut-model-v1/image-to-text/20261117_041911';J=B/'DonutInvoice_own_original_FP16_poscap2048_true3_v1_real_FLOCK_20261117';nf=R/'scripts/verify_donut_invoice_own_native_fp16_v2_source_YAML_and_exact_caption_metrics_20261117.py';assert hashlib.sha256(nf.read_bytes()).hexdigest()=='d1130454f3b7328f8f1b65fc2be76573788ba5b590a60ade0a77a0a7a6ed8a28'
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(folder.iterdir())+[J/'job.json',J/'completion.json']}
with (D/'runner.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with (H/'numeric_stdout.json').open('xb') as out,(H/'numeric_stderr.log').open('xb') as err:p=subprocess.run(['python3','-B',str(nf),str(folder),'3','--diagnostic-existing3-no-accept',str(H/'review.json')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
 assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in before.items())
v=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=p.returncode,new_NPU_inference=False,original_failed_runner_completion_not_accepted_or_modified=True,original_five_report_source_and_job_SHA256_unchanged=before,no_G_or_canonical_write=True);(H/'actual_completion.json').write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());raise SystemExit(p.returncode)
