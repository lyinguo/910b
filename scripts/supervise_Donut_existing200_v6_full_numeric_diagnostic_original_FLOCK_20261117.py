import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,fcntl,subprocess,os,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'DonutInvoice_existing_complete200_v6_present_raw_full_numeric_independent_diagnostic_real_FLOCK_20261117';F=R/'workspace/reports/katanaml-org_invoices-donut-model-v1/image-to-text/20261117_045854';J=B/'DonutInvoice_own_original_FP16_poscap2048_approved_fresh3_fixed200_v4_real_FLOCK_20261117'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 f=H/n;t=f.with_name(f.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,f)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);assert not runner_lock_owners() and not [x for x in project_processes() if x['PID']!=os.getpid() and x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protected=[f for f in F.iterdir() if f.is_file()]+[J/'job.json',J/'completion.json',D/'state.json',D/'registry.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];before={str(f):sha(f) for f in protected};put('protected_before.json',before)
 assert json.loads((J/'completion.json').read_bytes())['actual_returncode']==1
 source=R/'scripts/verify_donut_invoice_existing200_v6_present_raw_field_full_numeric_diagnostic_20261117.py';started=time.monotonic();timeout=False
 with (H/'stdout.json').open('xb') as out,(H/'stderr.log').open('xb') as err:
  p=subprocess.Popen(['python3','-B',str(source),str(F),'200','--diagnostic-existing-complete-report-no-accept',str(H/'review.json')],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),start_new_session=True)
  while p.poll() is None:
   timeout=timeout or time.monotonic()-started>21600;put('state.json',dict(UTC=now(),PID=os.getpid(),phase='independent_full_numeric_review_of_existing_complete200_no_new_NN',child_PID=p.pid,source_SHA256=sha(source),actual_unique_original_FLOCK=True,timeout_review_required_no_signal=timeout,original_verify_budget_seconds=21600));time.sleep(10)
 after={str(f):sha(f) for f in protected};assert after==before;put('protected_after.json',after);rc=p.returncode
 if rc==0:
  review=json.loads((H/'review.json').read_bytes());assert review['numeric_review_passed'] and not review['passed'] and not review['NEW_real_NPU_inference'] and review['sample_count']==200
 put('actual_completion.json',dict(UTC=now(),actual_returncode=rc if not timeout else 1,numeric_process_actual_returncode=rc,timeout_review_required_no_signal=timeout,elapsed_seconds=time.monotonic()-started,original_failed_job_unchanged=True,all_original_five_reports_and_canonical_hashes_unchanged=True,no_NPU_inference_or_G_write=True,review_SHA256=sha(H/'review.json') if (H/'review.json').exists() else None));return rc if not timeout else 1
if __name__=='__main__':
 try:raise SystemExit(main())
 except SystemExit:raise
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),original_job_and_reports_unchanged=True))
  raise
