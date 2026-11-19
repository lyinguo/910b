"""Serial first200 measurement after the preceding measured baseline gap.

Only the exact authorized retention-ledger transition is accepted. Neither
the preceding gap nor this first200 is preclaimed qualified. Model math and
the existing complete independent verifier remain unchanged.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import copy,fcntl,hashlib,json,os,shutil,subprocess,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
SELF='scripts/continue_GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.py'
CFG='configs/queues/GLM46_own_passed3_after_Huihui_numerical_gap_verified_cleanup_serial_v2_20261119.json'
MID='zai-org/GLM-4.6V-Flash';H=B/'GLM46_own_passed3_after_Huihui_full_NF_baselinegap_verified_correct_nonexample_cleanup_serial_v2_20261119'
J3=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118'
J200=B/'GLM46Flash_own_completed3_Huihui_numerical_gap_verified_cleanup_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v2_20261119'
CAP=B/'GLM46Flash_completed_true3_existing_full_LZMA_all200_component_shape_capacity_review_v3_actual_one_based_20261119/review.json'
MEASURE=B/'Huihui_completed200_full_numerical_baselinegap_target_audit_and_correct_nonexample_retention_20261119'
MODEL='universal_eval/model/adapters/glm46flash_own_original_BF16_thinking4096_standard_LZMA_serial_v2_20261119.py'
YAML='configs/Glm4vForConditionalGeneration/zai-org--GLM-4.6V-Flash-own-original-BF16-thinking4096-LZMA-component-capacity-serial-approved-first200-v2-20261119.yaml'
PROTO='configs/protocols/glm46flash_own_original_BF16_thinking4096_LZMA_component_capacity_serial_first200_v2_20261119.json'
LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json';RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def idle():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 others=[p for p in project_processes() if p['PID']!=os.getpid()];assert not others and not runner_lock_owners(),(others,runner_lock_owners())
def authorized_retention_chain(cfg,parent):
 from A100_alignment_evidence_runner_20261103 import unchanged
 assert sha(MEASURE/'measurement_target_audit.json')==cfg['parent_measurement_audit_SHA256'];audit=js(MEASURE/'measurement_target_audit.json')
 assert audit['passed'] and audit['sample_count']==200 and audit['score']==.695 and not audit['G_publication_permitted'] and not audit['original_protocol_final_qualification_complete']
 assert sha(MEASURE/'retention_actual_completion.json')==cfg['retention_receipt_SHA256'];receipt=js(MEASURE/'retention_actual_completion.json')
 assert receipt['actual_returncode']==0 and receipt['all_protected_SHA256_unchanged'] and receipt['all61_incorrect_debug_control_cases_and_full_original_examples_retained'] and receipt['canonical_scores_state_registry_binding_and_both_workbooks_unchanged']
 assert receipt['audit_SHA256']==cfg['parent_measurement_audit_SHA256'] and receipt['manifest_SHA256']==sha(receipt['manifest'])==cfg['retention_manifest_SHA256']
 assert sha(LED)==cfg['retention_ledger_after_SHA256'] and sha(MEASURE/'ledger_before_selective_retention.original')=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
 oldreceipt=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/actual_completion.json';assert sha(oldreceipt)=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3'
 ledger=js(LED);assert ledger['events'][-1]['receipt_SHA256']==cfg['retention_receipt_SHA256'] and ledger['events'][-2]['receipt_SHA256']==sha(oldreceipt)
 assert parent['protected_canonical_hashes'][str(LED)]=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5'
 unchanged({n:h for n,h in parent['protected_canonical_hashes'].items() if n!=str(LED)})
 assert all(sha(n)==h for n,h in audit['canonical_SHA256'].items() if n!=str(LED));protected=dict(parent['protected_canonical_hashes']);protected[str(LED)]=sha(LED)
 return protected
def main():
 import batch_910b as batch,yaml
 from batch_910b_support import validate_report,workbook_rows
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 cfg=js(R/CFG);assert cfg['max_concurrent_models']==1 and cfg['next_original_true3_replay_forbidden'] and cfg['parent_not_qualified_no_G_write']
 assert not H.exists() and not J200.exists() and sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 idle();H.mkdir();guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(J3/'job.json')=='b922c1c10b5a847c8fc1a54774a6c2e4f22cd9357abfc42f241e5950453a2e31';parent=js(J3/'job.json');protected=authorized_retention_chain(cfg,parent)
 a,=js(J3/'completion.json')['accepted'];assert sha(a['proof'])==a['proof_sha256']=='d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950';pf=js(a['proof']);assert pf['passed'] and pf['sample_count']==3 and pf['actual_NPU_inference'] and pf['actual_forward_count']==586
 validate_report(R,a['report_dir'],MID,'image-text-to-text',3,pf);batch.check_sources(parent['spec']);unchanged({str(R/n):h for n,h in cfg['source_SHA256'].items()})
 assert sha(CAP)==cfg['component_capacity_review_SHA256'];cap=copy.deepcopy(js(CAP));assert cap['passed'] and cap['original_factor']==4 and cap['original_extra_reserve_bytes']==6*1024**3 and cap['all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved']
 cap.update(UTC=now(),actual_free_bytes=shutil.disk_usage(R).free,capacity_passed=shutil.disk_usage(R).free>=cap['empirical_estimate_bytes']);put(H/'fresh_complete_component_capacity_review.json',cap)
 assert cap['capacity_passed'],('Actual original disk still insufficient',cap['empirical_estimate_bytes'],cap['actual_free_bytes'])
 original=yaml.safe_load((R/parent['spec']['config']).read_bytes());p=js(original['model']['kwargs']['protocol_file']);assert sha(original['model']['kwargs']['protocol_file'])==original['model']['kwargs']['protocol_sha256']
 for filename,v in p['own_weight_files'].items():assert sha(Path(p['weight_path'])/filename)==v['SHA256']
 for n,h in p['own_assets_SHA256'].items():assert sha(Path(p['weight_path'])/n)==h
 for n,h in pf['external_evidence_sha256'].items():
  if Path(n)==LED:assert h in [parent['protected_canonical_hashes'][str(LED)],cfg['retention_ledger_after_SHA256']]
  else:assert sha(n)==h,('Own true3 external changed',n)
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,_,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],capacity_review=str(H/'fresh_complete_component_capacity_review.json'),capacity_review_SHA256=sha(H/'fresh_complete_component_capacity_review.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,parent_Huihui_actual_completed_all_NN_NF_and_original_FLOCK_released=True,parent_baselinegap_preserved_not_published=True,parent_measurement_audit=str(MEASURE/'measurement_target_audit.json'),parent_measurement_audit_SHA256=cfg['parent_measurement_audit_SHA256'],authorized_retention_receipt=str(MEASURE/'retention_actual_completion.json'),authorized_retention_receipt_SHA256=cfg['retention_receipt_SHA256'],original_factor4_and_all_runtime_guards_unchanged=True)
 put(H/'first200_measurement_approval.json',approval)
 impl=dict(p['implementation_source_SHA256']);impl.update({n:sha(R/n) for n in [SELF,CFG,MODEL,'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py','scripts/glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py']})
 p2=copy.deepcopy(p);p2['implementation_source_SHA256']=impl;p2['reference_workbook_SHA256']=sha(book)
 p2['static_evidence_SHA256'].update({a['proof']:a['proof_sha256'],str(CAP):sha(CAP),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json'),str(MEASURE/'measurement_target_audit.json'):cfg['parent_measurement_audit_SHA256'],str(MEASURE/'retention_actual_completion.json'):cfg['retention_receipt_SHA256']})
 if str(LED) in p2['static_evidence_SHA256']:p2['static_evidence_SHA256'][str(LED)]=sha(LED)
 assert not (R/PROTO).exists() and not (R/YAML).exists();put(R/PROTO,p2);c2=copy.deepcopy(original);c2['name']=Path(YAML).stem;c2['model']['adapter']=MODEL+':Glm46FlashOwnNativeBF16';c2['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/YAML,c2)
 sources={n:h for n,h in parent['spec']['source_sha256'].items() if n not in [parent['spec']['config'],str(Path(original['model']['kwargs']['protocol_file']).relative_to(R))]};sources.update(impl);sources.update({YAML:sha(R/YAML),PROTO:sha(R/PROTO)})
 s2=copy.deepcopy(parent['spec']);s2.update(approved=True,config=YAML,source_sha256=sources,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
 scope(s2,[3,200],False);batch.validate_config(c2,MID)
 for n in [3,200]:
  path,actual=batch.config_for(s2,js(D/'state.json')['models'][MID],n);assert actual['sample_range']==f'0:{n-1}'
  if path.resolve().is_relative_to(R/'configs'):s2['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(s2);unchanged(protected);idle();assert batch.device_idle(s2)
 job=copy.deepcopy(parent);job.update(spec=s2,stages=[3,200],protected_canonical_hashes=protected,created_at=now(),private_stage_type='Own passed genuine3 then NEWfresh3 fixed200 serial; authorized numerical-measurement retention, original thinking4096 and full NF')
 J200.mkdir();put(J200/'job.json',job)
 with (H/'next_original_runner_stdout.log').open('xb') as out,(H/'next_original_runner_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J200)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='own_GLM46_approved_NEWfresh3_fixed200_original_runner_running',runner_PID=child.pid,job=str(J200),job_SHA256=sha(J200/'job.json'),no_G_or_state_registry_write=True));rc=child.wait()
 final=js(J200/'completion.json');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,original_job_completion=final,no_G_or_state_registry_write=True));assert rc==final['actual_returncode']==0;return 0
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=1,phase='failed_evidence_preserved',error=traceback.format_exc(),no_G_or_state_registry_write=True))
  raise
