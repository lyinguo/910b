import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,copy,shutil,os,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from A100_alignment_evidence_runner_20261103 import unchanged,scope
from validate_alignment_adapter_spec_1813_20261103 import validate_alignment_adapter_spec
import batch_910b as batch
H=B/'E4B_actual_extra_capacity_recovered_same_unstarted_true3_fixed200_v11_1605_20261113';J3=B/'E4B_complete_failure_records_source_bound_true3_v9_real_FLOCK_1605_20261113';J200=B/'E4B_capacity_recovered_complete_failure_records_approved_fresh3_fixed200_v10_real_FLOCK_1605_20261113';CLEAN=B/'four_completed_numeric_verified_formal_selective_matrix_retention_for_E4B_capacity_1605_20261113/actual_prune_v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def state(phase,**extra):
 p=H/'state.json';q=p.with_suffix('.json.tmp');q.write_text(json.dumps(dict(UTC=now(),PID=os.getpid(),phase=phase,no_G_or_canonical_write=True,**extra),ensure_ascii=False,indent=2),encoding='utf8');os.replace(q,p)
def quiet():return not runner_lock_owners() and not [p for p in project_processes() if p['PID']!=os.getpid() and p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
def run_job(job,phase):
 job=Path(job);state(phase,job=str(job),job_SHA256=sha(job/'job.json'))
 with (job/'controller_console.log').open('xb') as f:
  p=subprocess.Popen(['python3','-B',str(R/'scripts/A100_alignment_evidence_runner_20261103.py'),str(job)],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(R)),stdout=f,stderr=subprocess.STDOUT)
  write(job/'controller_started.json',dict(UTC=now(),PID=os.getpid(),runner_PID=p.pid,phase=phase));rc=p.wait()
 completion=js(job/'completion.json') if (job/'completion.json').exists() else None
 write(job/'controller_actual_completion.json',dict(UTC=now(),actual_returncode=rc,runner_PID=p.pid,completion=completion))
 assert rc==0 and completion and completion['actual_returncode']==0,('Actual frozen runner rejected; original evidence retained',str(job),rc)
 return completion
try:
 assert sha(R/'scripts/A100_alignment_evidence_runner_20261103.py')=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 EXTRA=B/'completed_E2B_numeric200_minimal_nonexample_retention_for_actual_E4B_capacity_1605_20261113/actual_prune_v1'
 start=time.monotonic();state('waiting_exact_extra_authorized_retention_completion_no_NN')
 while not (CLEAN/'actual_completion.json').exists() or not (EXTRA/'actual_completion.json').exists() or not quiet():
  if (CLEAN/'actual_completion.json').exists():assert js(CLEAN/'actual_completion.json')['passed']
  if (EXTRA/'actual_completion.json').exists():assert js(EXTRA/'actual_completion.json')['passed']
  assert time.monotonic()-start<7200,'Capacity cleanup or actual owned process still active; review required, no duplicate launch'
  time.sleep(10)
 cleanup=js(CLEAN/'actual_completion.json');assert cleanup['passed'] and cleanup['actual_returncode']==0 and cleanup['removed_files']==516 and cleanup['removed_bytes']==165717815323
 extra=js(EXTRA/'actual_completion.json');assert extra['passed'] and extra['actual_returncode']==0 and extra['removed_files']==122 and extra['removed_bytes']==12278505547
 source_review=js(H/'source_math_and_failure_record_review.json');assert source_review['passed'] and source_review['original_predict_math_and_full_observer_function_AST_identical_after_method_rename']
 forecast=js(H/'actual_prefix160_based_next200_lossless_forecast.json');free=shutil.disk_usage(R).free;assert free>forecast['required_bytes_including_original_reserves'],('Actual capacity still below the new empirical estimate',free,forecast)
 write(H/'actual_capacity_after_retention_preflight.json',dict(passed=True,UTC=now(),available_bytes=free,required_bytes=forecast['required_bytes_including_original_reserves'],empirical_not_worst_case_guarantee=True,original_per_sample_2GiB_guard_hard3GiB_log1GiB_preflight_halfGiB_kept=True,cleanup_receipt=str(CLEAN/'actual_completion.json'),cleanup_receipt_SHA256=sha(CLEAN/'actual_completion.json')))
 job=js(J3/'job.json');unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);validate_alignment_adapter_spec(job['spec'],'google/gemma-3n-E4B');scope(job['spec'],[3],False)
 done3=run_job(J3,'running_new_source_bound_true3_and_independent_verification')
 accepted=next(x for x in done3['accepted'] if x['sample_count']==3);assert sha(accepted['proof'])==accepted['proof_sha256'];proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3 and proof['actual_NPU_inference'] and not proof['historical_record_reuse'] and proof['all1631_own_source_actual_AA_NPU_SHA256']
 DIAG=B/'E4B_original10_empty_raw_full_extra_schema_independent_diagnostic_recheck_v1_20261113';assert js(DIAG/'actual_completion.json')['actual_returncode']==0 and js(DIAG/'verification_stdout.json')['passed']
 assert quiet();free=shutil.disk_usage(R).free;assert free>forecast['required_bytes_including_original_reserves']
 approval=dict(approved=True,UTC=now(),model_id='google/gemma-3n-E4B',approval_kind='Actual resource-guard retry after complete source-bound genuine NPU3 and independent full numerics, then NEWfresh3 and fixed0:199. Original failed160/40 and cancelled extra readonly diagnostic remain immutable. Same10 precision gap remains disclosed; only new full200 threeD5pct can qualify.',real_NPU3_independent_proof_verified=True,numerical_review_verified=True,real_NPU3_independent_proof=accepted['proof'],real_NPU3_independent_proof_SHA256=accepted['proof_sha256'],full_original_ten_numeric_proof=str(DIAG/'verification_stdout.json'),full_original_ten_numeric_proof_SHA256=sha(DIAG/'verification_stdout.json'),same_original_ten_AA_accuracy=.9,same_original_ten_NPU_accuracy=.6,same_ten_relative_gap='0.3333333333333333333333333333',same_ten_precision_pass_claim=False,only_exception_recording_source_change_verified=True,source_AST_review=str(H/'source_math_and_failure_record_review.json'),source_AST_review_SHA256=sha(H/'source_math_and_failure_record_review.json'),new_fresh3_mandatory=True,fixed200_indices='0:199',final_current_server_local_and_preserved_D_threshold='.05',no_G_or_canonical_write=True,space_actual_free_bytes=free,space_required_bytes=forecast['required_bytes_including_original_reserves'])
 write(H/'new_source_true3_verified_fresh3_fixed200_approval_review.json',approval)
 formal=copy.deepcopy(job);formal['created_at']=now();formal['stages']=[3,200];spec=formal['spec'];spec.update(approved=True,review_revision='20261113_E4B_v10_retry_after_actual_capacity_guard_and_complete_failure_records',alignment_approval_review=str(H/'new_source_true3_verified_fresh3_fixed200_approval_review.json'),alignment_approval_review_sha256=sha(H/'new_source_true3_verified_fresh3_fixed200_approval_review.json'))
 validate_alignment_adapter_spec(spec,'google/gemma-3n-E4B');batch.check_sources(spec);scope(spec,[3,200],False);unchanged(formal['protected_canonical_hashes']);assert batch.device_idle(spec)
 J200.mkdir(exist_ok=False);write(J200/'job.json',formal)
 done=run_job(J200,'approved_new_fresh3_then_fixed200_actual_frozen_runner')
 final=next(x for x in done['accepted'] if x['sample_count']==200);assert sha(final['proof'])==final['proof_sha256'];pf=js(final['proof']);assert pf['passed'] and pf['sample_count']==200
 state('new_fixed200_full_independent_complete_pending_threeD_and_publication_review',formal200=final,original_protocol_qualification=pf.get('original_protocol_final_qualification_complete',False),final_threeD_within5=pf['formal_alignment_within_5pct'])
 write(H/'actual_completion.json',dict(passed=True,UTC=now(),actual_returncode=0,new_true3_complete=True,new_fresh3_fixed200_complete=True,formal200=final,final_threeD_within5=pf['formal_alignment_within_5pct'],no_G_or_canonical_write=True))
except BaseException:
 error=traceback.format_exc();state('failed_evidence_preserved_no_automatic_duplicate',error=error);write(H/'actual_completion.json',dict(passed=False,UTC=now(),actual_returncode=1,error=error,no_G_or_canonical_write=True));raise
