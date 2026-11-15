"""Resume accepted own true3 after authorized storage recovery; never rerun it."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,subprocess,os,shutil,fcntl,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J3=B/'InternVL2_2B_own_native_true3_v5_real_FLOCK_1535_20261115';J=B/'InternVL2_2B_shape_capacity_retention_approved_fresh3_fixed200_v6_real_FLOCK_1535_20261115';H=B/'InternVL2_2B_verified3_after_authorized_retention_fresh3_fixed200_v6_1535_20261115';T=B/'authorized_verified_E4B_FigCodifier_nonexample_retention_1535_20261115';CAP=B/'InternVL2_2B_actual_all_frontend_shape_capacity_v6_1535_20261115/review.json';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=True);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (H/'actual_completion.json').exists() and not J.exists()
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','independent_verifier']]
 done=js(J3/'completion.json');assert done['actual_returncode']==0;a=done['accepted'][-1];proof=js(a['proof']);assert sha(a['proof'])==a['proof_sha256']=='9baf80c786825386d4d07ee853ac8c779beee93952d7f51053f2f5bc064e5743' and proof['passed'] and proof['sample_count']==3 and proof['all_own_source_actual_NPU_SHA256_verified'] and proof['all_native_KV_values_losslessly_reconstructed'];old=js(J3/'job.json');batch.check_sources(old['spec']);validate_report(R,a['report_dir'],old['model_id'],'image-text-to-text',3,proof)
 receipt=js(T/'actual_completion.json');plan=js(T/'plan.json');assert receipt['actual_returncode']==0 and receipt['all_protected_SHA256_unchanged'] and receipt['plan_SHA256']==sha(T/'plan.json') and sha(receipt['removed_manifest'])==receipt['removed_manifest_SHA256'];before=js(T/'retention_ledger_before_original_bytes.json');ledger=js(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json');assert ledger['events'][:-1]==before['events'] and ledger['events'][-1]['receipt_SHA256']==sha(T/'actual_completion.json')
 protected={}
 for n,h in old['protected_canonical_hashes'].items():
  actual=sha(n)
  if n==str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'):assert h==plan['ledger_before_SHA256'];protected[n]=actual
  else:assert actual==h,n;protected[n]=h
 cap=js(CAP);assert cap['passed'] and cap['true3_proof_SHA256']==a['proof_sha256'] and cap['original_hard_log_preflight_and_per_sample_guards_unchanged'] and cap['no_matrix_deleted'] and cap['no_model_math_precision_budget_or_retention_change'];free=shutil.disk_usage(D).free;assert free>=cap['empirical_estimate_bytes'],(free,cap['empirical_estimate_bytes']);assert batch.device_idle(old['spec'])
 approval=dict(approved=True,UTC=now(),model_id=old['model_id'],real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],capacity_review=str(CAP),capacity_review_SHA256=sha(CAP),authorized_retention_receipt=str(T/'actual_completion.json'),authorized_retention_receipt_SHA256=sha(T/'actual_completion.json'),actual_free_bytes=free,empirical_space_estimate_bytes=cap['empirical_estimate_bytes'],empirical_not_worst_case_guarantee=True,original_hard_log_and_per_sample_guards_preserved=True,small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True,accepted_true3_not_repeated=True,no_model_math_precision_budget_or_current_evidence_deletion=True)
 put('first200_measurement_approval.json',approval);new=copy.deepcopy(old);new.update(stages=[3,200],protected_canonical_hashes=protected);new['spec'].update(approved=True,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));scope(new['spec'],[3,200],False);unchanged(protected);batch.check_sources(new['spec']);J.mkdir();(J/'job.json').write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n');assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 with (H/'runner.log').open('xb') as out:
  child=subprocess.Popen([sys.executable,'-B',str(RUNNER),str(J)],cwd=R,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'));put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_fixed200',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=child.pid,no_G_or_canonical_write=True));rc=child.wait()
 completion=js(J/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_formal_job_completion=completion,no_G_or_canonical_write=True));assert rc==completion['actual_returncode']==0
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
