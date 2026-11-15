"""Continue original own option scoring from independently repaired real3 evidence.
Preserve the original failed runner/verifier receipts. No model math or scores are edited.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,fcntl,os,subprocess,shutil,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'SmolBase_fullsource_v3_CANN_schema_recovered_then_formal200_user_continue_20261115';J3=B/'SmolBase_own_fullsource_original_option_true3_v2_real_FLOCK_user_continue_20261115';J200=B/'SmolBase_own_fullsource_original_option_approved_fresh3_fixed200_v3_real_FLOCK_user_continue_20261115'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';VERIFIER='scripts/verify_smolbase_own_fullsource_bf16_v3_actual_CANN.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from batch_910b_support import validate_report
 import batch_910b as batch
 H.mkdir(exist_ok=True);g=(H/'continuation.guard').open('a+');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists() and not J200.exists()
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 job=js(J3/'job.json');done=js(J3/'completion.json');assert done['actual_returncode']==1 and done['verifier_returncode']==1 and done['model_returncode']==1
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec'])
 proof_path=H/'recovered_existing_true3_full_numeric_proof.json'
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='independent_readonly_existing_true3_CANN_schema_repair_no_NPU_rerun',report=done['report_dir']))
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (H/'true3_reaudit_stdout.json').open('xb') as out,(H/'true3_reaudit_stderr.log').open('xb') as err:
  c=subprocess.Popen([sys.executable,'-B',str(R/VERIFIER),done['report_dir'],'3',str(proof_path)],cwd=R,stdout=out,stderr=err,env=env);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='independent_readonly_existing_true3_CANN_schema_repair_no_NPU_rerun',child_PID=c.pid));rc=c.wait()
 assert rc==0,(H/'true3_reaudit_stderr.log').read_text();proof=js(proof_path);assert proof['passed'] and proof['source_tensor_count']==471 and proof['actual_forward_count']==6 and proof['all_own_source_actual_NPU_SHA256_verified'] and proof['original_candidate_probability_protocol_preserved']
 validate_report(R,done['report_dir'],job['model_id'],'image-text-to-text',3,proof)
 line=js(Path(done['report_dir'])/'operator_lineage.json')['actual_execution'];home=Path(line['full_source_evidence']).parent
 arrays=[Path(p) for p in proof['external_evidence_sha256'] if Path(p).suffix=='.npz' and Path(p).is_relative_to(home)];size=sum(p.stat().st_size for p in arrays);estimate=4*(2*size+(size/3)*197)+6*1024**3;free=shutil.disk_usage(D).free
 capacity=dict(passed=True,capacity_passed=free>=estimate,measured_true3_complete_array_bytes=size,empirical_estimate_bytes=estimate,actual_free_bytes=free,formula='4 * (2*complete_true3_bytes + 197*complete_true3_bytes/3) + 6GiB',empirical_not_worst_case_guarantee=True,all_original_hard_log_preflight_per_sample_guards_preserved=True,no_current_or_old_matrix_deleted=True);put('capacity_review.json',capacity);assert capacity['capacity_passed']
 recovery=dict(passed=True,original_job=str(J3),original_runner_actual_returncode=1,original_verifier_actual_returncode=1,original_failure_not_rewritten=True,new_independent_readonly_reaudit_returncode=0,new_independent_proof=str(proof_path),new_independent_proof_SHA256=sha(proof_path),all_numeric_source_and_complete_five_checks_rerun=True,actual_CANN_device4_kernel_count=proof['NPU_profiler_kernel_events'],no_completed_NPU_true3_repeated=True,model_math_weights_scoring_and_budget_unchanged=True);put('original_failure_schema_recovery_review.json',recovery)
 approval=dict(approved=True,UTC=now(),model_id=job['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=str(proof_path),true3_proof_SHA256=sha(proof_path),capacity_review=str(H/'capacity_review.json'),capacity_review_SHA256=sha(H/'capacity_review.json'),prior_schema_failure_recovery=str(H/'original_failure_schema_recovery_review.json'),prior_schema_failure_recovery_SHA256=sha(H/'original_failure_schema_recovery_review.json'),small3_score=proof['metrics']['Strict_MC_Accuracy'],small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True,no_model_math_precision_budget_or_current_evidence_deletion=True);put('first200_measurement_approval.json',approval)
 new=copy.deepcopy(job);new['stages']=[3,200];new['spec'].update(approved=True,verifier=VERIFIER,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));new['spec']['source_sha256'][VERIFIER]=sha(R/VERIFIER);scope(new['spec'],[3,200],False);batch.check_sources(new['spec']);unchanged(job['protected_canonical_hashes']);assert not runner_lock_owners() and batch.device_idle(new['spec'])
 J200.mkdir();(J200/'job.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with (H/'formal_runner.log').open('xb') as log:
  child=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(J200)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200',job=str(J200),job_SHA256=sha(J200/'job.json'),child_PID=child.pid,no_G_or_canonical_write=True));rc=child.wait()
 completed=js(J200/'completion.json');assert rc==completed['actual_returncode']==0;put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=completed,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
