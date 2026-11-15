"""Immediate next adaptation preflight after the real model/audit/lock release."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,time,subprocess,fcntl,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'MiniCPMV4_own_native_approved_fresh3_fixed200_v4_standard_approval_real_FLOCK_user_continue_20261115';H=B/'next_DotsOCR_own_resource_source_stage_after_MiniCPMV4_actual_release_user_continue_20261115';SOURCE=R/'scripts/prepare_dotsocr_own_resources_sources_after_current_model.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);g=(H/'continuation.guard').open('a+');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists();expected=sha(SOURCE);job_sha=sha(J/'job.json')
 while True:
  assert sha(SOURCE)==expected and sha(J/'job.json')==job_sha
  active=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
  if (J/'completion.json').exists() and not active and not runner_lock_owners():break
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_same_MiniCPMV4_model_and_full_audit_and_original_lock_release',current_job=str(J),next_model='rednote-hilab/dots.ocr',next_stage='own_original_resource_and_source_preflight_not_NPU_ready'));time.sleep(10)
 done=json.loads((J/'completion.json').read_bytes());put('state.json',dict(UTC=now(),PID=os.getpid(),phase='next_own_DotsOCR_resource_source_preflight_running',prior_actual_returncode=done['actual_returncode'],prior_failure_not_suppressed=True,no_new_NPU_inference_claim=True))
 with (H/'next_stage_stdout.json').open('xb') as out,(H/'next_stage_stderr.log').open('xb') as err:
  c=subprocess.Popen([sys.executable,'-B',str(SOURCE)],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'));rc=c.wait()
 put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,next_stage='own_resource_source_preflight_only',next_native_adapter_and_YAML_still_require_completion=True,no_G_or_canonical_write=True))
 put('NOTIFY_stage_needs_native_implementation.json',dict(UTC=now(),prior_model_completed_or_failed=done['actual_returncode'],next_resource_stage_returncode=rc,next_model='rednote-hilab/dots.ocr',NPU_stage_not_yet_ready=True,disk_alert_is_not_chat_delivery=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),disk_alert_is_not_chat_delivery=True))
  raise
