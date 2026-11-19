"""Audit complete standard lossless copies of already accepted own real3 values."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,math,os,shutil,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
J=B/'HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118'
OLD=B/'HuihuiGLM46_completed_true3_serial_efficiency_original_capacity_resumer_v2_1155_20261119'
H=B/'HuihuiGLM46_completed_true3_complete_standard_Fortran_LZMA_capacity_after_verified4972_cleanup_20261119'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
 p=H/n
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def verified_cleanup_ledger_transition_guard(before):
 from A100_alignment_evidence_runner_20261103 import unchanged
 ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
 cleanup=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119'
 assert sha(cleanup/'actual_completion.json')=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3'
 actual=js(cleanup/'actual_completion.json');assert actual['actual_returncode']==0 and actual['all_protected_SHA256_unchanged'] and actual['removed_files']==4972
 assert sha(ledger)=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
 assert before[str(ledger)]=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5'
 unchanged({p:h for p,h in before.items() if p!=str(ledger)})
 assert js(ledger)['events'][-1]['receipt_SHA256']==sha(cleanup/'actual_completion.json')
 return dict(only_authorized_retention_ledger_changed=True,ledger_SHA256=sha(ledger),cleanup_receipt_SHA256=sha(cleanup/'actual_completion.json'))
def main():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 from A100_alignment_evidence_runner_20261103 import unchanged
 from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import copy_and_check_all_values
 assert not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit']] and not runner_lock_owners()
 done=js(J/'completion.json');assert done['actual_returncode']==0 and done['model_returncode'] in [0,1] and done['verifier_returncode']==0
 accepted,=done['accepted'];assert sha(accepted['proof'])==accepted['proof_sha256']=='7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59';proof=js(accepted['proof']);assert proof['passed'] and proof['sample_count']==3
 job=js(J/'job.json');verified_cleanup_ledger_transition_guard(job['protected_canonical_hashes']);H.mkdir(exist_ok=False);put('authorized_cleanup_ledger_transition.json',verified_cleanup_ledger_transition_guard(job['protected_canonical_hashes']));folder=Path(accepted['report_dir']);protected={str(p):sha(p) for p in folder.iterdir()};line=js(folder/'operator_lineage.json')['actual_execution'];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()]
 protected.update({str(J/'job.json'):sha(J/'job.json'),str(J/'completion.json'):sha(J/'completion.json'),accepted['proof']:sha(accepted['proof'])})
 lock=(D/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');results=[]
 try:
  for row in trace:
   e=js(row['actual_execution_file']);source=Path(e['archive']);assert sha(source)==e['archive_SHA256'];protected[str(source)]=e['archive_SHA256'];dest=H/source.name
   review=copy_and_check_all_values(source,dest);review.update(original_path=str(source),original_SHA256=sha(source),new_path=str(dest),new_SHA256=sha(dest),index=e['index']);put('complete_array_review_%d.json'%e['index'],review);results.append(review)
  oldcap=js(OLD/'full_shape_complete_numeric_storage_capacity_review.json');total=sum(r['new_bytes']+Path(trace[r['index']]['actual_execution_file']).stat().st_size for r in results);estimate=max(math.ceil(total*max(oldcap['ratios'].values())*4)+6*1024**3,int(4.5*1024**3)+20*1024**3);free=shutil.disk_usage(R).free
  unchanged(protected);verified_cleanup_ledger_transition_guard(job['protected_canonical_hashes'])
  review=dict(UTC=datetime.now(timezone.utc).isoformat(),diagnostic_only=True,actual_NEW_model_forwards=0,no_full_NF_rerun_or_formal_storage_adoption_yet=True,all_complete_original_names_shapes_dtypes_every_raw_bit_equal=True,all_original_reports_source_old_proof_and_arrays_unmodified=True,original_three_archive_bytes=sum(r['original_bytes'] for r in results),new_three_archive_bytes=sum(r['new_bytes'] for r in results),capacity=dict(empirical_estimate_bytes=estimate,actual_free_bytes=free,passed=free>=estimate,all_original200_shape_ratios=oldcap['ratios'],factor4_plus6GiB_and_all_original_guards_preserved=True,empirical_not_worst_case=True),source=str(Path(__file__).relative_to(R)),source_SHA256=sha(__file__),original_true3_proof=accepted['proof'],original_true3_proof_SHA256=accepted['proof_sha256'],all_original_protected_SHA256=protected,complete_per_array_value_reviews=[str(H/('complete_array_review_%d.json'%r['index'])) for r in results],NEW_NPU_approval_fresh3_200_or_G=False,no_deletion=True)
  put('review.json',review);put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,diagnostic_only=True,no_NEW_model_inference=True));print(json.dumps(review,ensure_ascii=False))
 except BaseException:
  put('actual_failure.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),diagnostic_only=True,no_NEW_model_inference=True));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
