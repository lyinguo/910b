"""Recover standard approval schema before NN; preserve passed real3 and old failure."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,fcntl,subprocess,traceback,copy,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'MiniCPMV4_passed_true3_standard_approval_schema_recovered_v4_user_continue_20261115'
J3=B/'MiniCPMV4_own_native_true3_v3_raw_tokenizer_real_FLOCK_user_continue_20261115';OLD=B/'MiniCPMV4_own_native_approved_fresh3_fixed200_v3_raw_tokenizer_real_FLOCK_user_continue_20261115';PARENT=B/'MiniCPMV4_true3_then_measured_capacity_approved_fresh3_fixed200_v3_raw_tokenizer_user_continue_20261115';J=B/'MiniCPMV4_own_native_approved_fresh3_fixed200_v4_standard_approval_real_FLOCK_user_continue_20261115'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists() and not J.exists()
 assert not (OLD/'state.json').exists() and not (OLD/'completion.json').exists()
 oldlog=(PARENT/'running_approved_NEWfresh3_then_fixed200_and_independent_audit.log').read_text();assert "KeyError: 'real_NPU3_independent_proof_verified'" in oldlog
 done=js(J3/'completion.json');assert done['actual_returncode']==0 and len(done['accepted'])==1;rec=done['accepted'][0];assert rec['sample_count']==3 and sha(rec['proof'])==rec['proof_sha256']
 proof=js(rec['proof']);job=js(OLD/'job.json');assert proof['passed'] and proof['all_own_source_actual_NPU_SHA256_verified'] and proof['source_tensor_count']==741 and proof['actual_forward_count']==284
 put('state.json',dict(UTC=now(),PID=os.getpid(),phase='fresh_SHA_all_existing_passed_true3_and_capacity_no_NPU_rerun'))
 validate_report(R,rec['report_dir'],job['model_id'],'image-text-to-text',3,proof)
 for path,h in proof['external_evidence_sha256'].items():assert sha(path)==h,(path,'changed')
 capacity=js(PARENT/'capacity_review.json');assert capacity['capacity_passed'] and shutil.disk_usage(D).free>=capacity['empirical_estimate_bytes']
 unchanged(job['protected_canonical_hashes']);batch.check_sources(job['spec']);assert not runner_lock_owners() and batch.device_idle(job['spec'])
 assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 review=dict(UTC=now(),approved=True,model_id=job['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=rec['proof'],true3_proof_SHA256=rec['proof_sha256'],capacity_review=str(PARENT/'capacity_review.json'),capacity_review_SHA256=sha(PARENT/'capacity_review.json'),all741_own_source_head_masks_visual_injection_KV_inputs_full_five_verified=True,small3_metric=proof['metrics'],small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True,original_missing_standard_fields_failure_preserved=True,no_completed_NPU_true3_repeated=True)
 put('first200_standard_schema_measurement_approval.json',review)
 job['spec'].update(alignment_approval_review=str(H/'first200_standard_schema_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_standard_schema_measurement_approval.json'));job['spec']['source_sha256'][str(Path(__file__).relative_to(R))]=sha(__file__)
 scope(job['spec'],[3,200],False);batch.check_sources(job['spec']);J.mkdir();(J/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 with (H/'formal_runner_stdout.log').open('xb') as out,(H/'formal_runner_stderr.log').open('xb') as err:
  c=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(J)],cwd=R,stdout=out,stderr=err,env=env,start_new_session=True);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200_and_full_independent_audit',child_PID=c.pid,job=str(J),job_SHA256=sha(J/'job.json'),no_G_or_canonical_write=True));rc=c.wait()
 final=js(J/'completion.json');assert rc==final['actual_returncode']==0;put('actual_completion.json',dict(UTC=now(),actual_returncode=0,formal_job_completion=final,no_G_or_canonical_write=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():
   q=dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True);put('actual_completion.json',q);put('NOTIFY_error.json',dict(q,disk_alert_is_not_chat_delivery=True))
  raise
