"""Use the real original EX lock to audit unchanged genuine NPU3; no inference replay."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,os,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from A100_alignment_evidence_runner_20261103 import unchanged
from batch_910b_support import validate_report
from eager_corrective_control_safety_2203 import kernel_lock_owned
import batch_910b as batch
OLD=B/'Phi35_own_original_native_BF16_4096_true3_v2_metadata_guard_real_FLOCK_0616_20261118'
HOME=B/'Phi35_completed_genuine_true3_formal_NF3_actual_profiler_core_schema_no_NN_replay_1642_20261118'
VERIFIER='scripts/verify_phi35_completed_true3_formal_NF3_actual_profiler_schema_no_NN_replay_1642_20261118.py'
BASE='scripts/verify_phi35_own_original_native_BF16_4096_v3_actual_profiler_core_schema_1642_20261118.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode());os.replace(tmp,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 active=[p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 assert not active and not runner_lock_owners(),active
 assert sha(OLD/'job.json')=='a433f2d64b56da90d215f8532913424528f4a6d7acbe05a6a3d6bde72870ccb6'
 assert sha(OLD/'completion.json')=='5be1b579d0bcf0b271d84900fa4c4a0a3b5420230f8bd01407a598285851f691'
 assert sha(R/BASE)=='56d7e6f41f4465bb9a30b585a574a76be2520aa556b3be9198dc9a50d25a1a99'
 before=js(OLD/'completion.json');assert before['actual_returncode']==1 and before['model_returncode']==1 and before['verifier_returncode']==1 and before['sample_count']==3
 error=(OLD/'fresh3/verification_stderr.log').read_text();assert 'same own native LongRoPE operand key layer1' in error and '0.04949951171875' in error
 job=copy.deepcopy(js(OLD/'job.json'));batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes'])
 folder=Path(before['report_dir']);names={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
 assert {p.name for p in folder.iterdir()}==names;original={n:sha(folder/n) for n in names}
 assert js(folder/'eval_report.json')['statistics']['total_samples']==3 and js(folder/'eval_report.json')['statistics']['error_count']==0
 job.update(created_at=now(),execution_kind='formal_independent_acceptance_of_completed_original_NPU3_only',source_original_NPU3_job=str(OLD),source_original_NPU3_job_SHA256=sha(OLD/'job.json'),source_original_NPU3_completion_SHA256=sha(OLD/'completion.json'),actual_NEW_NPU_forwards_in_this_acceptance=0,original_full_five_SHA256=original)
 job['spec']['verifier']=VERIFIER
 for rel in [BASE,VERIFIER,str(Path(__file__).relative_to(R))]:job['spec']['source_sha256'][rel]=sha(R/rel)
 HOME.mkdir(exist_ok=False);put(HOME/'job.json',job)
 put(HOME/'formal_acceptance_source_review.json',dict(approved=True,original_model_exit1_is_complete_baselinegap_only=True,original_model_math_and_original_numeric_matrices_and_complete_five_unmodified=True,original_separate_BF16_RoPE_reference_order_restored=True,no_numeric_threshold_relaxed=True,actual_NEW_model_forwards=0,original_failed_job_remains_failed=True,old_original_completion_SHA256=sha(OLD/'completion.json'),all_original_five_SHA256=original))
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock')
 batch.check_sources(job['spec']);unchanged(job['protected_canonical_hashes'])
 attempt=HOME/'fresh3';attempt.mkdir();state=dict(PID=os.getpid(),phase='independent_CPU_verification',current=job['model_id'],sample_count=3,actual_NEW_model_forwards=0,source_original_genuine_NPU3=str(OLD),original_model_returncode=1,started_at=now(),accepted=[])
 put(HOME/'state.json',state);child=None
 try:
  with (attempt/'verification_stdout.json').open('xb') as out,(attempt/'verification_stderr.log').open('xb') as err:
   child=subprocess.Popen(['python3','-B',str(R/VERIFIER),str(folder),'3'],cwd=R,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',A100_ALIGNMENT_EXECUTION_JOB=str(HOME/'job.json')))
   state['child_PID']=child.pid;put(HOME/'state.json',state);start=time.monotonic();over=False
   while child.poll() is None:
    if time.monotonic()-start>job['spec']['verify_timeout_seconds']:over=True;state['timeout_review_required_no_signal']=True
    state['updated_at']=now();put(HOME/'state.json',state);time.sleep(3)
   code=child.wait();state['child_PID']=None;state['verifier_returncode']=code;put(HOME/'state.json',state)
   assert code==0 and not over,'Full formal NF2 failed; preserve original inference and failure receipts'
  proof=js(attempt/'verification_stdout.json');assert proof['passed'] and proof['sample_count']==3 and proof['formal_independent_acceptance_phase_not_diagnostic_only'] and proof['actual_NEW_model_forwards_in_this_acceptance']==0
  score=validate_report(R,folder,job['model_id'],'image-text-to-text',3,proof)
  unchanged(job['protected_canonical_hashes']);assert all(sha(folder/n)==h for n,h in original.items())
  accepted=dict(sample_count=3,score=score,report_dir=str(folder),proof=str(attempt/'verification_stdout.json'),proof_sha256=sha(attempt/'verification_stdout.json'),new_real_NPU_inference=True,original_real_NPU3_job=str(OLD),no_new_model_inference_in_this_acceptance_phase=True,formal_independent_acceptance_not_diagnostic_only=True,no_G_or_canonical_write=True)
  put(attempt/'accepted.json',accepted);state.update(phase='completed_formal_independent_acceptance_of_original_NPU3',actual_returncode=0,accepted=[accepted],completed_at=now(),original_failed_job_and_all_original_five_unchanged=True)
  put(HOME/'state.json',state);put(HOME/'completion.json',state);put(HOME/'actual_completion.json',dict(UTC=now(),actual_returncode=0,actual_verifier_exit_code=code,actual_NEW_model_forwards=0,proof=str(attempt/'verification_stdout.json'),proof_SHA256=sha(attempt/'verification_stdout.json')))
  return 0
 except BaseException:
  if child is not None:
   while child.poll() is None:time.sleep(3)
  state.update(phase='failed_formal_acceptance_preserved',actual_returncode=1,error=traceback.format_exc(),completed_at=now());put(HOME/'state.json',state);put(HOME/'completion.json',state);put(HOME/'actual_completion.json',state);raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':sys.exit(main())
