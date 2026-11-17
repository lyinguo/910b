"""Reuse passed own real3; audited storage/current-book transitions only.

Preserve old jobs/proofs/failures. No model math, input, dtype, cache, prompt,
budget, scoring, online A100 prerequisite, small-sample 5% gate or G write.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import json,hashlib,copy,fcntl,os,shutil,subprocess,traceback,zipfile,xml.etree.ElementTree as ET
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='prithivMLmods/gemma-4-E4B-it-Uncensored-MAX';J3=B/'PrithivGemma4E4B_own_original_BF16_true3_v2_own_config_real_FLOCK_20261116';J=B/'PrithivGemma4E4B_completed_true3_capacity_recovered_approved_fresh3_fixed200_v3_static_current_binding_real_FLOCK_20261117';H=B/'PrithivGemma4E4B_completed_true3_four_retention_capacity_recovery_v3_static_current_binding_20261117'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py';ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
def main():
 import batch_910b as batch
 from batch_910b_support import workbook_rows,validate_config,validate_report
 from A100_alignment_evidence_runner_20261103 import scope
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 reviewed=Path(sys.argv[1]);assert sha(reviewed)==sys.argv[2];review=js(reviewed);assert review['model_id']==MID and review['no_new_true3_repetition'] and review['explicitly_static_reference_and_authorized_retention_only']
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=js(J3/'job.json');assert sha(J3/'job.json')=='97f6d021494d40a7d2c7bb64e8a6a395e8b55476c3841c87cc57ec110868f0d4' and not J.exists();done=js(J3/'completion.json');assert done['actual_returncode']==0;accepted,=done['accepted'];proof=js(accepted['proof']);assert sha(accepted['proof'])==accepted['proof_sha256']=='fd3bdcaf169547fe4e90f3fb6e1ea269ccb7d2490a085090d07d9d942a42c9b8'
 assert proof['passed'] and proof['sample_count']==3 and proof['actual_forward_count']==475 and proof['all2130_own_source_actual_NPU_and_unused_SHA256_verified'] and proof['all54_unused_shared_KV_source_preserved'] and proof['all_full_head_rows_independently_recomputed'] and proof['all_native_KV_values_losslessly_reconstructed'];cfg=js(R/old['spec']['config']);validate_report(R,accepted['report_dir'],MID,cfg['task_type'],3,proof)
 preserved={str(J3/n):sha(J3/n) for n in ['job.json','completion.json']};preserved[str(Path(accepted['proof']))]=sha(accepted['proof'])
 for n,h in proof['report_sha256'].items():p=Path(accepted['report_dir'])/n;assert sha(p)==h;preserved[str(p)]=h
 batch.check_sources(old['spec']);assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 publications=[]
 for chain in review['authorized_completed_publication_chain']:
  pubdir=Path(chain['publication_home']);assert sha(pubdir/'full_target_dual_publication_audit.json')==chain['audit_SHA256'] and sha(pubdir/'publication_receipt.json')==chain['receipt_SHA256']
  au=js(pubdir/'full_target_dual_publication_audit.json');rcp=js(pubdir/'publication_receipt.json');assert au['passed'] and rcp['completed'] and js(pubdir/'dual_publication_completed.json')['completed']
  publications.append((pubdir,au,rcp))
 assert len(publications)==3 and publications[-1][1]['canonical_counts']['done']==176
 pub,audit,receipt=publications[-1]
 first_pub,first_audit,first_receipt=publications[0]
 for before_pub,after_pub in zip(publications,publications[1:]):
  for name,h in before_pub[1]['canonical_SHA256'].items():assert after_pub[2]['canonical_before_sha256'][name]==h,('Unproved canonical chain',name)
 retain=Path(review['authorized_retention_actual']);assert sha(retain)==review['authorized_retention_actual_SHA256'];ret=js(retain);assert ret['actual_returncode']==0 and ret['all_protected_SHA256_unchanged'] and ret['all_full_five_reports_all_records_source_YAML_snapshots_logs_preserved']
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')==review['authorized_retention_after_ledger_SHA256'] and sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json')=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7'
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']==review['current_dual_book_SHA256'];state=js(D/'state.json');entry=state['models'][MID];assert state['current'] is None and entry['status']=='needs_adaptation' and entry['score'] is None
 rows,sheet,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['score'] is None and row['task']==cfg['task_type']=='any-to-any';ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(book) as z:d=ET.fromstring(z.read(sheet)).find('.//s:c[@r="D'+str(row['row'])+'"]',ns).findtext('s:v',namespaces=ns)
 p0=js(cfg['model']['kwargs']['protocol_file']);assert Decimal(d)==Decimal(p0['current_A100_D_values']['server'])==Decimal('.8700') and js(binding['current_user_workbook_D_reference_snapshot'])['models'][MID]['A100_D']==p0['current_A100_D_values']['local']
 # This exact immutable workbook is the already-proved before-publication
 # book, not a replacement/changed D reference. Every other old source and
 # external evidence still has to match its original SHA.
 oldbook=Path(review['immutable_before_publication_book']);assert sha(oldbook)==p0['reference_workbook_SHA256']==first_receipt['canonical_before_sha256'][str(book)]
 for name,h in proof['external_evidence_sha256'].items():
  if name==p0['reference_workbook']:assert h==sha(oldbook)
  else:assert sha(name)==h,name
 for name,h in old['protected_canonical_hashes'].items():
  actual=sha(name)
  if actual==h:continue
  if name==str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'):
   assert h==review['original_job_ledger_SHA256'] and sha(review['immutable_original_job_ledger_backup'])==h and sha(review['immutable_before_latest_retention_ledger_backup'])==review['authorized_retention_before_ledger_SHA256'] and actual==review['authorized_retention_after_ledger_SHA256']
   previous=js(review['immutable_before_latest_retention_ledger_backup']);original_ledger=js(review['immutable_original_job_ledger_backup']);assert previous['events'][:len(original_ledger['events'])]==original_ledger['events'];current=js(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json');assert current['events'][:-1]==previous['events'] and current['events'][-1]['receipt']==str(retain) and current['events'][-1]['receipt_SHA256']==sha(retain)
  else:assert first_receipt['canonical_before_sha256'].get(name)==h and actual==audit['canonical_SHA256'].get(name),name
 cap0=Path(review['original_capacity_review']);assert sha(cap0)==review['original_capacity_review_SHA256'];cap=js(cap0);assert cap['passed'] and not cap['capacity_passed'] and cap['true3_proof_SHA256']==accepted['proof_sha256'] and cap['empirical_estimate_bytes']==123344571732
 cap['original_pre_recovery_capacity_review']=str(cap0);cap['original_pre_recovery_capacity_review_SHA256']=sha(cap0);cap['UTC']=now();cap['actual_free_bytes']=shutil.disk_usage(D).free;cap['capacity_passed']=cap['actual_free_bytes']>=cap['empirical_estimate_bytes'];cap['prior_explicitly_authorized_other_verified_runs_retention_actual']=str(retain);cap['prior_explicitly_authorized_other_verified_runs_retention_actual_SHA256']=sha(retain);cap['no_matrix_deleted_scope']='This measurement only; preceding authorized retention is recorded separately';put('capacity_review_after_actual_authorized_retention.json',cap);assert cap['capacity_passed'] and batch.device_idle(old['spec'])
 approval=dict(approved=True,UTC=now(),model_id=MID,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=accepted['proof'],true3_proof_SHA256=accepted['proof_sha256'],capacity_review=str(H/'capacity_review_after_actual_authorized_retention.json'),capacity_review_SHA256=sha(H/'capacity_review_after_actual_authorized_retention.json'),actual_free_bytes=cap['actual_free_bytes'],empirical_space_estimate_bytes=cap['empirical_estimate_bytes'],empirical_not_worst_case_guarantee=True,original_hard_log_and_per_sample_guards_preserved=True,authorized_retention_completed_before_this_job=True,small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True,old_passed_true3_not_repeated=True,static_immutable_reference_path_only_rebound=True,no_model_math_input_dtype_cache_budget_scoring_change=True);put('first200_measurement_approval.json',approval)
 approval.update(original_true3_reference_workbook=p0['reference_workbook'],original_true3_reference_workbook_SHA256=p0['reference_workbook_SHA256'],immutable_equal_original_true3_reference_workbook=str(oldbook),immutable_equal_original_true3_reference_workbook_SHA256=sha(oldbook),current_D_reconfirmed_after_legitimate_other_model_publication=True,current_other_model_publication_full_audit=str(pub/'full_target_dual_publication_audit.json'),current_other_model_publication_full_audit_SHA256=sha(pub/'full_target_dual_publication_audit.json'));put('first200_measurement_approval.json',approval)
 p=copy.deepcopy(p0);p['reference_workbook']=str(oldbook);p['static_evidence_SHA256'].update({str(oldbook):sha(oldbook),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json'),str(H/'capacity_review_after_actual_authorized_retention.json'):sha(H/'capacity_review_after_actual_authorized_retention.json'),str(retain):sha(retain)})
 runtime=lambda v:{k:x for k,x in v.items() if k not in ['reference_workbook','static_evidence_SHA256']};assert runtime(p)==runtime(p0);put('exact_nonstatic_protocol_identity_review.json',dict(passed=True,UTC=now(),old_protocol=str(cfg['model']['kwargs']['protocol_file']),old_protocol_SHA256=sha(cfg['model']['kwargs']['protocol_file']),all_nonstatic_fields_exactly_identical=True,only_reference_path_changed_to_equal_original_bytes_and_approval_evidence_added=True,original_passed_true3_and_complete_reports_unchanged=True))
 pp=R/'configs/protocols/prithiv_gemma4_e4b_own_original_BF16_capacity_recovered_first200_v3_static_current_binding_20261117.json';create(pp,p);cf=R/'configs/Gemma4ForConditionalGeneration/prithivMLmods--gemma-4-E4B-it-Uncensored-MAX-own-original-BF16-2048-v3-static-current-binding-capacity-recovered-approved-first200-20261117.yaml';cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg)
 new=copy.deepcopy(old);new['stages']=[3,200];new['created_at']=now();new['protected_canonical_hashes']={n:sha(n) for n in old['protected_canonical_hashes']};new['spec'].update(approved=True,config=str(cf.relative_to(R)),alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));new['spec']['source_sha256'].update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp),str(Path(__file__).resolve().relative_to(R)):sha(__file__)})
 batch.check_sources(new['spec']);scope(new['spec'],[3,200],False)
 for count in [3,200]:_,c=batch.config_for(new['spec'],entry,count);validate_config(c,MID)
 for n,h in preserved.items():assert sha(n)==h
 create(J/'job.json',new)
 with (H/'runner_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=ENV);put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_new_approved_fresh3_fixed200_reusing_passed_true3',job=str(J),job_SHA256=sha(J/'job.json'),child_PID=child.pid));rc=child.wait()
 final=js(J/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_formal_job_completion=final,no_G_or_canonical_write=True));assert rc==final['actual_returncode']==0
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
