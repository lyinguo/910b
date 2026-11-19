"""Event-driven serial continuation; no original lock is held while waiting.

Only a complete audited Huihui can be published/pruned. The next model reuses
its already passed own genuine3, then runs NEWfresh3 and fixed200 once. Every
model parameter and original numeric verifier remains unchanged.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,fcntl,hashlib,json,os,shutil,subprocess,time,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
SELF='scripts/continue_after_Huihui_real_dual181_then_GLM46_component_capacity_serial_v1_20261119.py'
CFG='configs/queues/Huihui_real_dual181_then_own_GLM46_complete_component_capacity_serial_20261119.json'
MID='zai-org/GLM-4.6V-Flash';HH='huihui-ai/Huihui-GLM-4.6V-Flash-abliterated'
H=B/'after_Huihui_real_dual181_then_GLM46_own_completed3_component_capacity_serial_v1_20261119'
JP=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v4_approval_schema_real_FLOCK_20261119'
HPUB=B/'HuihuiGLM46Flash_original_BF16_2048_lossless_verified181_blank_G_publication_20261119'
J3=B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118'
CAP=B/'GLM46Flash_completed_true3_existing_full_LZMA_all200_component_shape_capacity_review_v3_actual_one_based_20261119/review.json'
J200=B/'GLM46Flash_own_completed3_Huihui_dual181_actual_release_component_capacity_LZMA_approved_NEWfresh3_fixed200_serial_v1_20261119'
MODEL='universal_eval/model/adapters/glm46flash_own_original_BF16_thinking4096_standard_LZMA_serial_v2_20261119.py'
YAML='configs/Glm4vForConditionalGeneration/zai-org--GLM-4.6V-Flash-own-original-BF16-thinking4096-LZMA-component-capacity-serial-approved-first200-v1-20261119.yaml'
PROTO='configs/protocols/glm46flash_own_original_BF16_thinking4096_LZMA_component_capacity_serial_first200_v1_20261119.json'
RET=B/'authorized_verified181_Huihui_only_accepted_nonexample_serial_next_20261119'
RETCFG='configs/retention/verified181_Huihui_only_accepted_nonexample_serial_next_20261119.json'
LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
RUNNER=R/'scripts/A100_alignment_evidence_runner_20261103.py'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def live_others():
 from A100_alignment_incremental_monitor_20261103 import project_processes
 return [x for x in project_processes() if x['PID']!=os.getpid()]
def legal_base(parent,published):
 from A100_alignment_evidence_runner_20261103 import unchanged
 changes={}
 if published:
  audit=js(HPUB/'full_target_dual_publication_audit.json');pub=js(HPUB/'publication_receipt.json');dual=js(HPUB/'dual_publication_completed.json')
  assert audit['passed'] and audit['canonical_counts']=={'done':181,'skipped_no_complete_weights':310,'needs_adaptation':209} and audit['numeric_within5pct']==166
  assert pub['completed'] and pub['old180_state_registry_and_G_preserved'] and len(pub['targets'])==1 and pub['targets'][0]['model_id']==HH
  assert dual['completed'] and dual['local_and_server_workbook_sha256']==sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
  for n,new in audit['canonical_SHA256'].items():
   assert sha(n)==new and parent['protected_canonical_hashes'][n]==pub['canonical_before_sha256'][n]
   changes[n]=new
 assert sha(LED)=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
 receipt=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/actual_completion.json';assert sha(receipt)=='3fc6d85e83cdfe8ebb2b9a9f9079c6ea7fc33caeb393d5bd8cf0c2b307615eb3'
 assert parent['protected_canonical_hashes'][str(LED)]=='4ae138778ac41754487c3c116b341c845efb7d890d0b4ba9d08e061d8d1688e5' and js(LED)['events'][-1]['receipt_SHA256']==sha(receipt)
 changes[str(LED)]=sha(LED);unchanged({n:h for n,h in parent['protected_canonical_hashes'].items() if n not in changes})
 return {n:changes.get(n,h) for n,h in parent['protected_canonical_hashes'].items()}
def cleanup_current_accepted_Huihui(cfg):
 from A100_alignment_evidence_runner_20261103 import unchanged
 import maintenance.retain_verified181_Huihui_accepted_nonexamples_serial_next_20261119 as retention
 from batch_910b_support import validate_report
 pub=js(HPUB/'publication_receipt.json');t,=pub['targets'];proof=js(t['formal_proof']);assert sha(t['formal_proof'])==t['formal_proof_sha256'] and proof['passed'] and proof['sample_count']==200 and proof['original_protocol_final_qualification_complete']
 state=js(D/'state.json');assert state['models'][HH]['status']=='done' and state['models'][HH]['report_dir']==t['new_report'];validate_report(R,t['new_report'],HH,'image-text-to-text',200,proof)
 folder=Path(t['new_report']);line=js(folder/'operator_lineage.json')['actual_execution'];home=Path(line['evidence_root']).resolve();assert home.is_relative_to(R/'workspace/diagnostics/model_runs') and home.name.startswith('HuihuiGLM46Flash_own_original_BF16_2048_')
 trace=[json.loads(x) for x in Path(line['sample_trace']).read_text().splitlines()];assert [x['index'] for x in trace]==list(range(200))
 oldcfg=js(R/'configs/retention/verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json');helper=retention.old_helpers(oldcfg)
 keep=set(helper.KEEP)|set(proof['error_indices'])|set(proof['empty_indices'])|set(proof['invalid_indices'])|set(proof['truncated_indices']);remove=[]
 for row in trace:
  e=js(row['actual_execution_file']);p=Path(e['archive']).resolve();assert p.parent==home and p.name=='sample_%05d.npz'%row['index'] and proof['external_evidence_sha256'][str(p)]==e['archive_SHA256']
  if row['index'] in keep:continue
  assert helper.removable_file(p,home) and sha(p)==e['archive_SHA256'];s=p.stat();remove.append(dict(path=str(p),home=str(home),model_id=HH,logical_indices=[row['index']],bytes=s.st_size,allocated_bytes=s.st_blocks*512,SHA256=sha(p),proof=t['formal_proof'],proof_SHA256=t['formal_proof_sha256']))
 assert remove and len(remove)<=160
 RET.mkdir(exist_ok=False);old=js(B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/discovery.json')
 selected=[x for x in old['all_selected_proof_metadata'] if x['model_id']!=HH]+[dict(model_id=HH,proof=t['formal_proof'],proof_SHA256=t['formal_proof_sha256'],report_dir=t['new_report'])]
 canonical={n:sha(n) for n in js(JP/'job.json')['protected_canonical_hashes']}
 disc=dict(UTC=now(),canonical_SHA256=canonical,all_selected_proof_metadata=selected,scopes=[dict(model_id=HH,proof=t['formal_proof'],proof_SHA256=t['formal_proof_sha256'],report_dir=t['new_report'],home=str(home),remove=remove)],explicit_scope_only_new_current_Huihui=True)
 put(RET/'discovery.json',disc)
 rcfg=dict(oldcfg,expected_done_count=181,latest_actual_user_authorization=cfg['latest_actual_user_authorization'],discovery_SHA256=sha(RET/'discovery.json'),allowed_model_ids=[HH],known_no_NN_waiters=[],max_remove_files=160,max_remove_bytes=sum(x['bytes'] for x in remove))
 put(R/RETCFG,rcfg);planned=retention.plan(sha(RET/'discovery.json'));put(H/'current_Huihui_retention_plan_review.json',planned)
 assert planned['remove_count']<=160 and all(s['model_id']==HH for s in planned['scopes']);result=retention.apply(planned['plan_SHA256']);put(H/'current_Huihui_retention_actual.json',result)
 assert result['actual']['actual_returncode']==0 and result['actual']['all_protected_SHA256_unchanged'];unchanged({n:h for n,h in canonical.items() if n!=str(LED)});return result
def main():
 import batch_910b as batch,yaml
 from batch_910b_support import validate_report,workbook_rows
 from A100_alignment_incremental_monitor_20261103 import runner_lock_owners
 from A100_alignment_evidence_runner_20261103 import unchanged,scope
 cfg=js(R/CFG);cfg_SHA256=sha(R/CFG);assert cfg['max_concurrent_models']==1
 H.mkdir(exist_ok=False);guard=(H/'continuation.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(RUNNER)=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
 while True:
  assert sha(JP/'job.json')==cfg['parent_job_SHA256'] and sha(R/CFG)==cfg_SHA256
  unchanged({str(R/n):h for n,h in cfg['source_SHA256'].items()})
  done=js(JP/'completion.json') if (JP/'completion.json').exists() else None
  published=(HPUB/'full_target_dual_publication_audit.json').exists() and (HPUB/'dual_publication_completed.json').exists()
  needs_publication=True
  if done:
   successful200=[x for x in done.get('accepted',[]) if x['sample_count']==200]
   if done['actual_returncode']!=0 or not successful200:needs_publication=False
   elif successful200:
    final=js(successful200[0]['proof']);needs_publication=bool(final['formal_alignment_within_5pct'] and final['original_protocol_final_qualification_complete'])
  others=live_others();locks=runner_lock_owners();put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='waiting_for_actual_Huihui_full200_then_required_dual_publication_and_all_project_process_lock_release',parent_completed=bool(done),parent_actual_returncode=done.get('actual_returncode') if done else None,published181=published,parent_requires_publication=needs_publication,actual_NN_NF=others,original_FLOCK=locks,continuation_holds_original_runner_lock=False))
  if done and (published or not needs_publication) and not others and not locks:break
  time.sleep(10)
 parent=js(J3/'job.json');assert sha(J3/'job.json')==cfg['own_completed_true3_job_SHA256'];protected=legal_base(parent,published)
 a,=js(J3/'completion.json')['accepted'];assert sha(a['proof'])==a['proof_sha256']=='d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950';pf=js(a['proof']);assert pf['passed'] and pf['sample_count']==3 and pf['actual_NPU_inference'] and pf['actual_forward_count']==586
 validate_report(R,a['report_dir'],MID,'image-text-to-text',3,pf);assert sha(CAP)==cfg['component_capacity_review_SHA256'];cap=js(CAP);assert cap['passed'] and cap['original_factor']==4 and cap['original_extra_reserve_bytes']==6*1024**3 and cap['all_original_hard3GiB_log1GiB_precheckHalfGiB_per_sample2GiB_times_outer10_guards_preserved']
 if shutil.disk_usage(R).free<cap['empirical_estimate_bytes'] and published:
  cleanup=cleanup_current_accepted_Huihui(cfg);protected[str(LED)]=cleanup['ledger_SHA256']
 cap=copy.deepcopy(cap);cap.update(UTC=now(),actual_free_bytes=shutil.disk_usage(R).free,capacity_passed=shutil.disk_usage(R).free>=cap['empirical_estimate_bytes'])
 put(H/'fresh_complete_component_capacity_review.json',cap)
 if not cap['capacity_passed']:
  value=dict(UTC=now(),actual_returncode=1,phase='capacity_blocked_serial_candidate_after_actual_parent_release',model_id=MID,need=cap['empirical_estimate_bytes'],free=cap['actual_free_bytes'],no_NPU_replay_or_G_write=True);put(H/'actual_completion.json',value);put(H/'NOTIFY_capacity_blocked.json',dict(actual=value,notification_file_is_not_chat_delivery=True));return 1
 original=yaml.safe_load((R/parent['spec']['config']).read_bytes());p=js(original['model']['kwargs']['protocol_file']);assert sha(original['model']['kwargs']['protocol_file'])==original['model']['kwargs']['protocol_sha256']
 for filename,v in p['own_weight_files'].items():assert sha(Path(p['weight_path'])/filename)==v['SHA256']
 for n,h in p['own_assets_SHA256'].items():assert sha(Path(p['weight_path'])/n)==h
 book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');rows,_,_=workbook_rows(book);row,=[r for r in rows if r['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
 approval=dict(UTC=now(),approved=True,real_NPU3_independent_proof_verified=True,numerical_review_verified=True,first200_measurement_only_not_preclaimed_qualification=True,first200_is_measurement_not_predetermined_5pct_qualification=True,true3_job=str(J3),true3_job_SHA256=sha(J3/'job.json'),true3_proof=a['proof'],true3_proof_SHA256=a['proof_sha256'],capacity_review=str(H/'fresh_complete_component_capacity_review.json'),capacity_review_SHA256=sha(H/'fresh_complete_component_capacity_review.json'),own_small3_complete_values_retained=True,small3_not_an_additional_5pct_gate=True,parent_Huihui_actual_completed_and_all_model_CPU_and_FLOCK_released=True,parent_qualified_dual_publication_complete=published,original_factor4_and_all_runtime_guards_unchanged=True)
 put(H/'first200_measurement_approval.json',approval)
 p2=copy.deepcopy(p);impl=dict(p['implementation_source_SHA256']);impl.update({n:sha(R/n) for n in [SELF,CFG,MODEL,'universal_eval/utils/complete_standard_LZMA_NPZ_values_20261118.py','scripts/glm46_complete_shape_component_capacity_v3_actual_one_based_20261119.py']});p2['implementation_source_SHA256']=impl
 p2['reference_workbook_SHA256']=sha(book);p2['static_evidence_SHA256'].update({a['proof']:a['proof_sha256'],str(CAP):sha(CAP),str(H/'first200_measurement_approval.json'):sha(H/'first200_measurement_approval.json')})
 assert not (R/PROTO).exists() and not (R/YAML).exists();put(R/PROTO,p2);c2=copy.deepcopy(original);c2['name']=Path(YAML).stem;c2['model']['adapter']=MODEL+':Glm46FlashOwnNativeBF16';c2['model']['kwargs'].update(protocol_file=str(R/PROTO),protocol_sha256=sha(R/PROTO));put(R/YAML,c2)
 sources={n:h for n,h in parent['spec']['source_sha256'].items() if n not in [parent['spec']['config'],str(Path(original['model']['kwargs']['protocol_file']).relative_to(R))]};sources.update(impl);sources.update({YAML:sha(R/YAML),PROTO:sha(R/PROTO)})
 s2=copy.deepcopy(parent['spec']);s2.update(approved=True,config=YAML,source_sha256=sources,alignment_approval_review=str(H/'first200_measurement_approval.json'),alignment_approval_review_sha256=sha(H/'first200_measurement_approval.json'))
 if (R/RETCFG).exists():s2['source_sha256'][RETCFG]=sha(R/RETCFG)
 job=copy.deepcopy(parent);job.update(spec=s2,stages=[3,200],protected_canonical_hashes=protected,created_at=now(),private_stage_type='Serial own completed3 then NEWfresh3 fixed200; original thinking4096 and full NF, proven complete LZMA storage')
 scope(s2,[3,200],False);batch.validate_config(c2,MID)
 for n in [3,200]:
  path,actual=batch.config_for(s2,js(D/'state.json')['models'][MID],n);assert actual['sample_range']==f'0:{n-1}'
  if path.resolve().is_relative_to(R/'configs'):s2['source_sha256'][str(path.resolve().relative_to(R))]=sha(path)
 batch.check_sources(s2);unchanged(protected);assert not live_others() and not runner_lock_owners() and batch.device_idle(s2)
 J200.mkdir(exist_ok=False);put(J200/'job.json',job)
 with (H/'next_original_runner_stdout.log').open('xb') as out,(H/'next_original_runner_stderr.log').open('xb') as err:
  child=subprocess.Popen(['python3','-B',str(RUNNER),str(J200)],cwd='/root',stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
  put(H/'state.json',dict(UTC=now(),PID=os.getpid(),phase='next_own_GLM46_approved_NEWfresh3_fixed200_original_runner_running',runner_PID=child.pid,job=str(J200),job_SHA256=sha(J200/'job.json'),no_G_or_state_registry_write=True));rc=child.wait()
 final=js(J200/'completion.json');put(H/'actual_completion.json',dict(UTC=now(),actual_returncode=rc,original_job_completion=final,no_G_or_state_registry_write=True));assert rc==final['actual_returncode']==0;return 0
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException:
  if H.exists() and not (H/'actual_completion.json').exists():
   value=dict(UTC=now(),actual_returncode=1,phase='failed_evidence_preserved',error=traceback.format_exc(),no_G_or_state_registry_write=True);put(H/'actual_completion.json',value);put(H/'NOTIFY_error.json',dict(actual=value,notification_file_is_not_chat_delivery=True))
  raise
