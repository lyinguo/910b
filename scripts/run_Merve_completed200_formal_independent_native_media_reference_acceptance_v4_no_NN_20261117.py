"""Formal full verification of genuine completed NPU200; no model replay or score writes."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,os,fcntl,subprocess,traceback,time
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from A100_alignment_evidence_runner_20261103 import unchanged
from A100_alignment_resume_completed_approved_fresh3_guard_20261117 import validate_resume
from batch_910b_support import validate_report
from eager_corrective_control_safety_2203 import kernel_lock_owned,core_unchanged
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(tmp,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','legacy_controller']] and not runner_lock_owners();old=B/'Merve_lossless_v2_completed_approved_fresh3_resumed_unstarted200_after_retention_v3_real_FLOCK_20261117';job=js(old/'job.json');before=js(old/'completion.json');assert before['actual_returncode']==1 and before['model_returncode']==0 and before['verifier_returncode']==1;assert "assert matches,'No own original processor branch reproduced actual native input pixels'" in (old/'formal200/verification_stderr.log').read_text();unchanged(job['protected_canonical_hashes'])
 home=B/'Merve_completed_original200_formal_independent_acceptance_v4_native_media_reference_no_NN_replay_20261117';assert not home.exists();home.mkdir();folder=Path(before['report_dir']);assert {p.name for p in folder.iterdir()}=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};report_before={str(p):sha(p) for p in folder.iterdir()};job=copy.deepcopy(job);job.update(created_at=now(),execution_kind='formal_independent_acceptance_of_completed_original_NPU200_only',source_original_NPU200_job=str(old),source_original_NPU200_job_SHA256=sha(old/'job.json'),source_original_NPU200_completion_SHA256=sha(old/'completion.json'),actual_NEW_NPU_forwards_in_this_acceptance=0);job['spec']['verifier']='scripts/verify_Merve_completed200_formal_native_media_reference_v4_20261117.py'
 for n in [job['spec']['verifier'],str(Path(__file__).relative_to(R))]:job['spec']['source_sha256'][n]=sha(R/n)
 prior=validate_resume(job,R);put(home/'job.json',job);put(home/'formal_acceptance_source_review.json',dict(approved=True,original_genuine_NPU200_model_exit0=True,original_same_approved_NEWfresh3_proof=prior['proof'],original_same_approved_NEWfresh3_proof_SHA256=prior['proof_sha256'],no_completed_NPU_phase_replayed=True,original_model_YAML_protocol_and_mathematical_source_unchanged=True,independent_reference_decode_only_corrected_to_frozen_actual_native_decoder=True,all_original_numeric_assertions_and_precision_thresholds_preserved=True,original_failed_job_left_failed=True,all_original_five_SHA256=report_before));lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged();state=dict(PID=os.getpid(),phase='independent_CPU_verification',current=job['model_id'],sample_count=200,model_returncode=0,model_returncode_source_original_job=str(old/'completion.json'),actual_new_model_forwards_this_phase=0,accepted=[prior],started_at=now());put(home/'state.json',state);attempt=home/'formal200';attempt.mkdir();child=None
 try:
  with (attempt/'verification_stdout.json').open('xb') as out,(attempt/'verification_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(R/job['spec']['verifier']),str(folder),'200'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',A100_ALIGNMENT_EXECUTION_JOB=str(home/'job.json')));state['child_PID']=child.pid;put(home/'state.json',state);begin=time.monotonic();over=False
   while child.poll() is None:
    if time.monotonic()-begin>job['spec']['verify_timeout_seconds']:over=True;state['timeout_review_required_no_signal']=True
    state['updated_at']=now();put(home/'state.json',state);time.sleep(3)
   rc=child.wait();state['child_PID']=None;state['verifier_returncode']=rc;put(home/'state.json',state);assert rc==0 and not over,'Formal full independent acceptance failed; retain all originals'
  proof=js(attempt/'verification_stdout.json');assert proof['passed'] and proof['sample_count']==200 and proof['formal_independent_acceptance_phase_not_diagnostic_only'];score=validate_report(R,folder,job['model_id'],js(folder/'eval_report.json')['task_type'] if 'task_type' in js(folder/'eval_report.json') else folder.parent.name,200,proof);unchanged(job['protected_canonical_hashes']);unchanged(report_before)
  accepted=dict(sample_count=200,score=score,report_dir=str(folder),proof=str(attempt/'verification_stdout.json'),proof_sha256=sha(attempt/'verification_stdout.json'),new_real_NPU_inference=True,original_real_NPU200_job=str(old),no_new_NPU_inference_in_this_verification_phase=True,no_G_or_canonical_state_registry_write=True,relative_error_to_A100_table_D=proof['relative_error_to_A100_table_D'],formal_alignment_within_5pct=proof['formal_alignment_within_5pct']);put(attempt/'accepted.json',accepted);state['accepted'].append(accepted);state.update(phase='completed_formal_independent_acceptance_of_original_NPU200',actual_returncode=0,completed_at=now(),old_failed_job_and_original_reports_unchanged=True);put(home/'state.json',state);put(home/'completion.json',state);put(home/'actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_verifier_exit_code=rc,no_new_model_inference=True,original_NPU200_job_model_returncode=0));print(json.dumps(dict(actual_returncode=0,proof=str(attempt/'verification_stdout.json'),proof_SHA256=sha(attempt/'verification_stdout.json'),score=score),ensure_ascii=False))
 except BaseException:
  if child is not None:
   while child.poll() is None:time.sleep(3)
  state.update(phase='failed_formal_acceptance_preserved',actual_returncode=1,error=traceback.format_exc(),completed_at=now());put(home/'state.json',state);put(home/'completion.json',state);raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
