"""Resume the passed own 3/419 via a new approved fresh3/200 private job.

Only explicit verified nonexample retention restores capacity. Existing successful
3/419, old failed jobs, original math and all generation guards stay unchanged.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,fcntl,os,subprocess,shutil,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
MID='deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit'
H=B/'DeadbydawnGemma4E2B_completed_true3_419_exact_retention_capacity_recovered_approved_fresh3_fixed200_v3_continuation_20261117'
J3=B/'DeadbydawnGemma4E2B_own_original_BF16_true3_v2_own_constructor_binding_real_FLOCK_20261117'
J200=B/'DeadbydawnGemma4E2B_completed_true3_419_capacity_recovered_approved_fresh3_fixed200_v3_real_FLOCK_20261117'
RET=B/'authorized_single_CoderE4_verified176_nonexample_retention_for_Deadbydawn_capacity_20261117'
CAP=B/'DeadbydawnGemma4E2B_own_original_BF16_complete_true3_capacity_review_v2_own_constructor_binding_20261117/review.json'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def create(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
 import batch_910b as batch
 from batch_910b_support import validate_report,validate_config,workbook_rows
 from A100_alignment_evidence_runner_20261103 import scope,unchanged
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not runner_lock_owners() and not [x for x in project_processes() if x['PID']!=os.getpid() and x['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 retention=js(RET/'actual_completion.json');assert retention['actual_returncode']==0 and retention['removed_files']==170 and retention['removed_bytes']==9747351006 and retention['all_protected_SHA256_unchanged'] and retention['protected_files']==8855
 assert sha(RET/'plan.json')==retention['plan_SHA256']=='c4ea74b93a90e4f2246caba2590556ea46e86c7ffa2a4ea5609e74781f2b3ddb';assert sha(retention['removed_manifest'])==retention['removed_manifest_SHA256']
 done=js(J3/'completion.json');assert done['actual_returncode']==0 and done['model_returncode']==1 and done['verifier_returncode']==0;a,=done['accepted'];assert a['sample_count']==3 and a['proof_sha256']==sha(a['proof'])=='13b0835212cc55ac129236c325e3db472c16366ccc997c5a9cc4c9edb9f94b2a'
 proof=js(a['proof']);assert proof['passed'] and proof['actual_forward_count']==419 and proof['sample_count']==3 and proof['all1951_own_source_actual_NPU_SHA256_verified'] and proof['all2011_own_source_SHA_verified_with60_original_unused_sharedKV'] and proof['all_full_head_rows_independently_recomputed'] and proof['all_native_KV_values_losslessly_reconstructed']
 old=js(J3/'job.json');assert sha(J3/'job.json')=='49a1e98488bdc46da51be165894a777e24d5469e7fa3ee580df2719700837d00';batch.check_sources(old['spec']);parentcfg=js(R/old['spec']['config']);validate_report(R,a['report_dir'],MID,parentcfg['task_type'],3,proof)
 before={str(f):sha(f) for f in [J3/'job.json',J3/'completion.json',Path(a['proof']),D/'state.json',D/'registry.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]}
 cap=js(CAP);assert cap['passed'] and not cap['capacity_passed'] and cap['empirical_estimate_bytes']==79982650744 and cap['original2048_budget_preserved_not_a_worst_case_guarantee'] and cap['no_model_math_precision_budget_or_retention_change']
 for entry in cap['entries']:
  f=Path(entry['archive']);assert f.stat().st_size==entry['archive_bytes'] and sha(f)==entry['archive_SHA256']
 current=copy.deepcopy(cap);current.update(UTC=now(),actual_free_bytes=shutil.disk_usage(D).free,capacity_passed=shutil.disk_usage(D).free>=cap['empirical_estimate_bytes'],old_capacity_rejection_review=str(CAP),old_capacity_rejection_review_SHA256=sha(CAP),capacity_restoration_only_by_explicit_exact_verified_nonexample_retention=True,retention_actual_receipt=str(RET/'actual_completion.json'),retention_actual_receipt_SHA256=sha(RET/'actual_completion.json'),no_matrix_deleted=False,no_additional_matrix_deleted_by_this_controller=True)
 put('capacity_recovered_review.json',current);assert current['capacity_passed']
 state=js(D/'state.json');assert state['models'][MID]['status']=='needs_adaptation' and state['models'][MID]['score'] is None
 rows,_,_=workbook_rows(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'));row,=[x for x in rows if x['model_id']==MID];assert row['score'] is None
 approval=dict(approved=True,UTC=now(),model_id=MID,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],completed_true3_samples=3,completed_true3_actual_forwards=419,completed_true3_not_repeated=True,capacity_review=str(H/'capacity_recovered_review.json'),capacity_review_SHA256=sha(H/'capacity_recovered_review.json'),space_preflight_passed=True,empirical_space_estimate_bytes=current['empirical_estimate_bytes'],actual_free_bytes=shutil.disk_usage(D).free,empirical_not_worst_case_guarantee=True,original_hard_log_preflight_and_per_sample_guards_preserved=True,no_model_math_precision_budget_change=True,small3_relative_error_to_D=proof['relative_error_to_A100_table_D'],first200_is_measurement_not_predetermined_5pct_qualification=True)
 put('first200_measurement_approval.json',approval)
 p=js(parentcfg['model']['kwargs']['protocol_file']);childp=copy.deepcopy(p);ledger=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json';(H/'retention_ledger_after_exact170_original_bytes.json').write_bytes(ledger.read_bytes())
 static=[Path(a['proof']),H/'capacity_recovered_review.json',H/'first200_measurement_approval.json',RET/'actual_completion.json',Path(retention['removed_manifest']),H/'retention_ledger_after_exact170_original_bytes.json']
 childp['static_evidence_SHA256'].update({str(f):sha(f) for f in static})
 assert {k:v for k,v in childp.items() if k!='static_evidence_SHA256'}=={k:v for k,v in p.items() if k!='static_evidence_SHA256'}
 pp=R/'configs/protocols/deadbydawn_gemma4_e2b_completed419_capacity_recovered_approved_first200_v3_20261117.json';create(pp,childp)
 cf=R/'configs/Gemma4ForConditionalGeneration/deadbydawn101--gemma-4-E2B-Heretic-Uncensored-mlx-4bit-own-original-BF16-2048-v3-capacity-recovered-approved-first200-20261117.yaml';cfg=copy.deepcopy(parentcfg);cfg['name']=cf.stem;cfg['model']['kwargs'].update(protocol_file=str(pp),protocol_sha256=sha(pp));create(cf,cfg)
 for key in ['data','algorithm','metrics','baseline','output','task_type']:assert cfg[key]==parentcfg[key]
 assert cfg['model']['adapter']==parentcfg['model']['adapter'] and cfg['model']['model_path']==parentcfg['model']['model_path']
 job=copy.deepcopy(old);job.update(stages=[3,200],created_at=now(),protected_canonical_hashes={name:sha(name) for name in old['protected_canonical_hashes']});job['spec'].update(approved=True,config=str(cf.relative_to(R)),review_revision='completed419_real_capacity_recovery_static_only_v3',alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'));job['spec']['source_sha256'].update({str(cf.relative_to(R)):sha(cf),str(pp.relative_to(R)):sha(pp),str(Path(__file__).resolve().relative_to(R)):sha(__file__)})
 batch.check_sources(job['spec']);scope(job['spec'],[3,200],False)
 for count in [3,200]:_,c=batch.config_for(job['spec'],state['models'][MID],count);validate_config(c,MID)
 assert before=={name:sha(name) for name in before};unchanged(job['protected_canonical_hashes']);assert batch.device_idle(job['spec']) and not runner_lock_owners() and shutil.disk_usage(D).free>=current['empirical_estimate_bytes'];create(J200/'job.json',job)
 with (H/'runner_stdout.log').open('xb') as log:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J200)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put('state.json',dict(UTC=now(),PID=os.getpid(),phase='running_approved_NEWfresh3_then_fixed200_same_passed419_model',job=str(J200),job_SHA256=sha(J200/'job.json'),child_PID=child.pid,no_G_or_canonical_write=True));rc=child.wait()
 actual=js(J200/'completion.json');put('actual_completion.json',dict(UTC=now(),actual_returncode=rc,actual_job_completion=actual,old_passed_true3_419_not_repeated=True,no_G_or_canonical_write=True));assert rc==actual['actual_returncode']==0
if __name__=='__main__':
 try:main()
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_G_or_canonical_write=True))
  raise
