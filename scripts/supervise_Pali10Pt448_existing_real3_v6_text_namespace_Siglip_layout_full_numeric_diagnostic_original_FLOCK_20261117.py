import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'Pali10Pt448_existing_complete_real3_v6_text_namespace_Siglip_layout_full_numeric_diagnostic_original_FLOCK_20261117';J=B/'Pali10Pt448_preserved_completed_compat3_B10_new_full_native_value_true3_v4_named_worker_RPC_real_FLOCK_20261117';REPORT=R/'workspace/reports/google_paligemma2-10b-pt-448/image-text-to-text/20261117_081350';NF=R/'scripts/verify_pali10pt448_own_preserved_vllm_full_native_true3_v6_text_namespace_and_CPU_Siglip_layout_20261117.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert sha(NF)=='41de72c494751267224859d7a120b70ce05d85bff07704a42f0cf2086ef069e4';done=json.loads((J/'completion.json').read_bytes());assert done['actual_returncode']==done['verifier_returncode']==1 and done['model_returncode']==0
 assert {x.name for x in REPORT.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
 assert not runner_lock_owners() and not [x for x in project_processes() if x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','independent_verifier']]
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert len(runner_lock_owners())==1 and runner_lock_owners()[0]['PID']==os.getpid()
  protected=[D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'),J/'job.json',J/'completion.json']+list(REPORT.iterdir());before={str(p):sha(p) for p in protected};start=datetime.now(timezone.utc).isoformat()
  with (H/'verification_stdout.json').open('xb') as out,(H/'verification_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(NF),str(REPORT),'3','--existing-v4-report-diagnostic'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=start,PID=os.getpid(),child_PID=child.pid,phase='independent_full_value_existing_real3_diagnostic',no_new_NPU=True,not_old_failed_job_acceptance=True));t0=time.monotonic();late=False
   while child.poll() is None:
    if time.monotonic()-t0>21600 and not late:late=True;put('timeout_review_required_no_signal.json',dict(UTC=datetime.now(timezone.utc).isoformat(),timeout_seconds=21600,no_signal_sent=True))
    time.sleep(2)
   rc=child.wait()
  assert before=={str(p):sha(p) for p in protected},'Protected old reports, job or canonical changed'
  result=json.loads((H/'verification_stdout.json').read_bytes()) if rc==0 else None
  if rc==0:assert result['numeric_review_passed'] and result['diagnostic_only'] and not result['passed'] and not result['NEW_real_NPU_inference'] and result['original_failed_v4_job_or_report_not_promoted']
  final=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=rc if not late else 1,underlying_verifier_returncode=rc,timeout_review_required=late,protected_original_SHA256=before,all_old_reports_job_and_canonical_SHA_unchanged=True,no_new_NPU_or_score_G=True,elapsed_seconds=time.monotonic()-t0,review_SHA256=sha(H/'verification_stdout.json'))
  put('actual_completion.json',final);print(json.dumps(final,ensure_ascii=False));assert rc==0 and not late
if __name__=='__main__':
 H.mkdir(exist_ok=False)
 try:main()
 except BaseException:
  if not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_new_NPU=True))
  raise
