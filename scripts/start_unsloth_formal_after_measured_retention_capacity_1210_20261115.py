"""Approve first200 from completed real3 and measured unchanged retention.

No model/metric changes, evidence pruning, score publication or deadline edits.
The original real-lock runner performs NEWfresh3 then fixed200 independently.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,os,shutil,subprocess,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'UnslothE2Bit_measured_retention_capacity_formal_continuation_1210_20261115'
J3=B/'UnslothE2Bit_own_chat_native_true3_v3_completed_batch_real_FLOCK_1210_20261115'
J200=B/'UnslothE2Bit_own_chat_native_approved_fresh3_fixed200_v3_completed_batch_real_FLOCK_1210_20261115'
CAP=B/'UnslothE2Bit_actual_first3_retention_capacity_review_v2_shared_observer_1210_20261115/review.json'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(name,value):
 p=H/name;q=p.with_name(p.name+'.tmp.'+str(os.getpid()));q.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(q,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);lock=(H/'continuation.guard').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (H/'actual_completion.json').exists() and not J200.exists()
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']]
 completed=js(J3/'completion.json');assert completed['actual_returncode']==0
 item=completed['accepted'][-1];proof=js(item['proof']);assert proof['passed'] and proof['sample_count']==3 and sha(item['proof'])==item['proof_sha256']=='3fff60223ac5753ff8135870780c55136ee63d9703f2b2480f790efe60f823fa'
 validate_report(R,item['report_dir'],'unsloth/gemma-3n-E2B-it','image-text-to-text',3,proof)
 capacity=js(CAP);assert capacity['passed'] and capacity['capacity_passed'] and capacity['no_matrix_deleted'] and capacity['no_model_math_precision_budget_or_retention_change']
 assert shutil.disk_usage(D).free>=capacity['empirical_estimate_bytes']
 job=copy.deepcopy(js(J3/'job.json'));batch.check_sources(job['spec'])
 for path,digest in job['protected_canonical_hashes'].items():assert sha(path)==digest,path
 assert batch.device_idle(job['spec'])
 review=dict(approved=True,UTC=now(),model_id=job['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=str(item['proof']),true3_proof_SHA256=item['proof_sha256'],capacity_review=str(CAP),capacity_review_SHA256=sha(CAP),actual_free_bytes=shutil.disk_usage(D).free,empirical_space_estimate_bytes=capacity['empirical_estimate_bytes'],empirical_not_worst_case_guarantee=True,original_hard_log_and_per_sample_guards_preserved=True,no_retention_or_model_math_change=True,small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True)
 put('first200_measurement_approval.json',review)
 job['stages']=[3,200];job['spec'].update(approved=True,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
 scope(job['spec'],job['stages'],False);J200.mkdir();(J200/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 runner=R/'scripts/A100_alignment_evidence_runner_20261103.py';assert sha(runner)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner.log').open('xb') as log:
  child=subprocess.Popen([sys.executable,'-B',str(runner),str(J200)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='actual_approved_runner_started',job=str(J200),job_SHA256=sha(J200/'job.json'),child_PID=child.pid,no_G_or_canonical_write=True))
  rc=child.wait()
 completion=js(J200/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_formal_job_completion=completion,no_G_or_canonical_write=True));assert rc==completion['actual_returncode']==0
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
