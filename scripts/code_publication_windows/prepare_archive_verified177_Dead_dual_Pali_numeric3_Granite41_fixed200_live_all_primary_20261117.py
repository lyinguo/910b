import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast,tomllib
P=Path(__file__).parent
packets=[
'R_Pali_v9_original_async_EOS_commit_full_numeric_diagnostic_actual1_20261117.json',
'R_verified177_frozen_monitor_full_incremental_20261117_0940.json',
'R_Pali10pt_new_full_numeric_true3_v10_actual_launch_20261117.json',
'R_Pali_v10_and_Granite_source_scope_precheck_actual_20261117.json',
'R_Pali_v10_completed_full200_capacity_component_actual_20261117.json',
'R_Granite41_v2_first_real_failure_and_original_algorithm_v2_20261117.json',
'R_Granite41_completed_numeric3_metadata_review_all200_capacity_20261117.json',
'R_Granite41_completed_numeric3_return_schema_recovery_approved_actual_launch_20261117.json',
'R_Granite41_all_private_exact_trace_actual2_20261117.json',
'R_Deadbydawn_verified177_dual_publication_Pali_Granite_actual2_20261117.json',
'R_Deadbydawn_v3_full_evidence_for_publication_final_20261117.json',
'R_verified177_Pali_v6_numeric_Granite41_v3_export_exact_actual1_20261117.json',
'DeadbydawnGemma4E2B_verified177_incremental_real_results_export_20261117.json',
'R_Granite41_own_original_remote_source_frontend_contract_no_NN_20261117.json',
'R_Granite41_META_buffers_and_Pali_v10_actual2_20261117.json',
'R_root_preload_source_explicit_static_binding_no_mutation_20261117.json']
helpers=[
'prepare_Deadbydawn_verified177_publication_sources_and_actual_evidence_reader_20261117.py',
'prepare_Deadbydawn_verified177_actual_numeric_local_plan_20261117.py',
'prepare_Deadbydawn_verified177_reviewed_dual_publication_launch_and_local_CAS_tools_20261117.py',
'prepare_upload_Deadbydawn_verified177_exact_local_CAS_receipt_20261117.py',
'sync_Deadbydawn_verified177_local_exact_zip_CAS_20261117.py',
'deadbydawn_gemma4_e2b_workbook_cell_verified177_20261117.mjs',
'render_verified177_actual_published_workbook_20261117.mjs',
'prepare_Pali_v5_native_default_head_dimension_full_numeric_existing3_diagnostic_20261117.py',
'prepare_Pali_v6_exact_text_weight_selector_Siglip_CPU_layout_full_numeric_diagnostic_20261117.py',
'prepare_Pali_v7_original_native_internal_public_request_ID_full_numeric_diagnostic_20261117.py',
'prepare_Pali_original_intrinsic_RoPE_cache_NPU_exact_committed_SHA_reconstruction_20261117.py',
'prepare_Pali_v8_exact_old_SHA_intrinsic_operand_full_numeric_diagnostic_20261117.py',
'prepare_Pali_v9_original_async_EOS_commit_prefix_full_numeric_diagnostic_20261117.py',
'prepare_Pali10pt_v10_new_source_bound_full_numeric_true3_20261117.py',
'read_Pali_v9_original_async_EOS_commit_full_numeric_diagnostic_actual_20261117_remote.py',
'read_Pali10pt_v10_actual_and_original_all200_capacity_inputs_20261117_remote.py',
'record_Pali10Pt448_completed_v10_true3_full200_current_format_capacity_refusal_20261117_remote.py',
'prepare_Granite41_own_original_HF_BF16_frontend200_and_META_primary_20261117.py',
'prepare_Granite41_frontend_v2_actual_SizeDict_tuple_guards_no_math_change_20261117.py',
'prepare_Granite41_frontend_v3_exact_own_1083_META_wrapper_layout_20261117.py',
'prepare_Granite41_primary_native_numeric_config_and_immediate_controller_20261117.py',
'prepare_Granite41_readiness_v2_real_source_files_only_no_NN_20261117.py',
'prepare_Granite41_v3_source_scope_and_disabled_legacy_startup_guard_20261117.py',
'prepare_Granite41_v4_private_inert_import_cleanup_original_SDPA_20261117.py',
'prepare_Granite41_v5_exact_legacy_mask_keyword_and_full_failure_records_20261117.py',
'prepare_Granite41_completed_numeric3_metadata_fixed_NF4_approved_fresh3_200_20261117.py',
'read_Granite41_native_adapter_base_contract_and_Pali_v10_current_20261117_remote.py',
'read_Granite41_META_native_buffers_source_map_and_Pali_v10_actual_20261117_remote.py',
'read_Granite41_current_real_private_stage_and_canonical_verified177_20261117_remote.py',
'read_Granite41_v4_actual_private_clean_imports_current_20261117_remote.py',
'read_Granite41_all_current_private_attempts_and_exact_trace_20261117_remote.py',
'review_Granite41_completed_numeric3_runner_metadata_failure_and_all200_capacity_20261117_remote.py',
'prepare_archive_verified177_Dead_dual_Pali_numeric3_Granite41_fixed200_live_all_primary_20261117.py']
def enc(p):
 raw=Path(p).read_bytes();return dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
Q={n:enc(P/n) for n in packets};W={n:enc(P/n) for n in helpers};body=(P/'H177_complete_handoff_body_20261117.txt').read_bytes().replace(b'\r\n',b'\n');AUTOPATH=Path('C:/Users/lyg/.codex/automations/910b/automation.toml');auto=AUTOPATH.read_bytes();av=tomllib.loads(auto.decode());assert av['id']=='910b' and av['status']=='ACTIVE' and av['kind']=='heartbeat' and av['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300';assert 'notification_policy' not in av
LOCAL=dict(original_automation_TOML=enc(AUTOPATH),original_prompt_UTF8=enc(AUTOPATH))
LOCAL['original_prompt_UTF8']=dict(data=base64.b64encode(av['prompt'].encode()).decode(),SHA256=hashlib.sha256(av['prompt'].encode()).hexdigest(),bytes=len(av['prompt'].encode()))
LOCAL['actual_published_target_PNG']=enc(P/'DeadbydawnGemma4E2B_local_result_sync_verified177_20261117/final_published_239.png')
book=Path('D:/code-lv/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');st=book.stat();LOCALSTAT=dict(path=str(book),SHA256=hashlib.sha256(book.read_bytes()).hexdigest(),bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns);assert LOCALSTAT['SHA256']=='f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d'
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,base64,ast,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114';sys.path[:0]=[str(R),str(R/'scripts')];Q=__Q__;W=__W__;LOCAL=__LOCAL__;LOCALSTAT=__LOCALSTAT__;BODY=base64.b64decode(__BODY__)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
ptrfile=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before_ptr=ptrfile.read_bytes();ptr=js(ptrfile);parent=Path(ptr['handoff_directory']);assert sha(parent/'handoff.txt')=='8f10df61a67b91fd818c527d39f8b17525e6445a20a6138fb78d7c392e0fe88c';assert sha(parent/'facts.json')=='c67b7bd2d322e680e8bddd337b9b2876b48203a998b3a39646546b94912c7996'
old=Path(ptr['new_primary_pending']);assert sha(old)==ptr['new_primary_pending_SHA256']=='6abb15ec88181cb311e813e965881f5010a0e808de1da46029463bd91cd8f991';pending=js(old);records=pending['all_current_primary_source_and_true_consumer_configs'];known={x['path']:x for x in records};stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');H=D/'heartbeat_handoffs'/('A100_verified177_Dead_dual_Pali_numeric3_capacity_Granite41_fixed200_live_'+stamp);H.mkdir(exist_ok=False);(H/'handoff.txt').write_bytes(BODY);assert b'\\r' not in BODY
local_originals={}
for group,values in [('local_original_bytes',Q),('local_environment_original_bytes',LOCAL)]:
 for n,x in values.items():
  raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256'];f=H/group/(n+'.original');f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw);local_originals[n]=dict(SHA256=x['SHA256'],bytes=len(raw),snapshot=str(f))
paths=[]
for n,x in W.items():
 raw=base64.b64decode(x['data']);assert hashlib.sha256(raw).hexdigest()==x['SHA256']
 if n.endswith('.py'):ast.parse(raw)
 f=R/'scripts/code_publication_windows'/n
 if f.exists():assert f.read_bytes()==raw
 else:f.write_bytes(raw)
 paths.append(f)
# Capture changed/new executable source only in the finite source tree; never
# scan workspace for publishable code or include arrays/weights/reports.
source_prefixes=['scripts','universal_eval','configs'];cutoff=1794861000
for rootname in source_prefixes:
 for f in (R/rootname).rglob('*'):
  if not f.is_file() or '__pycache__' in f.parts or f.suffix not in ['.py','.yaml','.yml','.json','.mjs','.jinja']:continue
  if f.stat().st_mtime>=cutoff:paths.append(f)
jobs=[B/'Pali10Pt448_new_source_bound_full_native_value_true3_v10_correct_numeric_contract_real_FLOCK_20261117',B/'Granite41_own_original_HF_BF16_4096_true3_v3_exact_mask_ABI_real_FLOCK_20261117',B/'Granite41_completed_numeric3_approved_NEWfresh3_fixed200_v4_runner_return_schema_real_FLOCK_20261117']
for home in jobs:
 if (home/'job.json').exists():
  for n,h in js(home/'job.json')['spec']['source_sha256'].items():f=R/n;assert sha(f)==h;paths.append(f)
excluded={str(R/'configs/protocols/granite41_own_original_HF_BF16_4096_native_v1_20261117.json'),str(R/'configs/Granite4VisionForConditionalGeneration/ibm-granite--granite-vision-4.1-4b-own-original-HF-BF16-4096-native-v1-20261117.yaml')};excluded_originals={}
for f in sorted(set(paths)):
 assert f.is_relative_to(R) and not f.is_relative_to(R/'workspace');raw=f.read_bytes();h=hashlib.sha256(raw).hexdigest()
 if f.suffix=='.py':ast.parse(raw)
 if str(f) in excluded:
  dest=H/'rejected_unconsumed_configuration_original_bytes'/(str(f.relative_to(R))+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);excluded_originals[str(f)]=dict(SHA256=h,snapshot=str(dest),reason='Prototype rejected before any model config consumer; historical evidence only, not automatic code publication');continue
 if str(f) in known:assert known[str(f)]['SHA256']==h,(str(f),'Reviewed source external change requires targeted review');continue
 snapshot=H/'primary_source_original_bytes'/(str(f.relative_to(R))+'.txt');snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(raw);record=dict(path=str(f),relative=str(f.relative_to(R)),SHA256=h,bytes=len(raw),snapshot=str(snapshot));records.append(record);known[str(f)]=record
new=C/'pending_after_daily_20261117_verified177_Dead_Pali_numeric3_Granite41_formal_all_primary_v10.json';assert not new.exists();pending.update(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(old),parent_pending_SHA256=sha(old),all_current_primary_source_and_true_consumer_configs=records,all_new_useful_code_kept_in_primary_source_scripts_configs=True,generated_approved_and_smoke_configs_captured=True,later_generated_configs_still_require_incremental_capture=True,non_consumer_rejected_prototypes_excluded_but_preserved=excluded_originals,CodeDay17_already_pushed_no_second_commit=True,Day18_MJS_suffix_publisher_support_and_tests_review_required=True);put(new,pending)
homes=[home for home in B.iterdir() if home.is_dir() and (home.name.startswith('Granite41_') or home.name.startswith('Pali10Pt448_existing') or home.name.startswith('Pali10Pt448_new_') or home.name.startswith('Pali10Pt448_completed_v10_') or home.name.startswith('DeadbydawnGemma4E2B_verified177_') or home.name=='DeadbydawnGemma4E2B_original_HF_BF16_verified177_blank_G_publication_20261117')];small={}
for home in homes:
 for f in home.rglob('*'):
  if not f.is_file() or any(x in f.parts for x in ['source_original_bytes','primary_source_original_bytes','source_snapshots']):continue
  if f.suffix not in ['.json','.jsonl','.log','.yaml','.txt','.diff'] or f.stat().st_size>8*1024*1024:continue
  raw=f.read_bytes();dest=H/'small_original_evidence_bytes'/(str(f.relative_to(B))+'.original');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);small[str(f)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),snapshot=str(dest),captured_while_live_possible=True)
for f in [D/'incremental_monitor/latest.json',D/'incremental_monitor/checkpoint.json',D/'A100_ALIGNMENT_PUBLICATION_POLICY.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',D/'A100_ALIGNMENT_HANDOFF_CURRENT.json']:
 raw=f.read_bytes();dest=H/'static_original_bytes'/(f.name+'.original');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);small[str(f)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),snapshot=str(dest))
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
current=jobs[-1];Hlive=B/'Granite41_completed_true3_numeric0_runner_return_schema_recovered_NEWfresh3_fixed200_v5_continuation_20261117';traces=[]
for home in (R/'workspace/diagnostics/model_runs').glob('Granite41_own_HF_BF16_*'):
 p=home/'actual_sample_trace.jsonl'
 if p.exists():
  rows=[json.loads(s) for s in p.read_text().splitlines()];traces.append(dict(home=str(home),completed_samples=len(rows),submitted_native_forwards=sum(x.get('actual_native_forwards',0) for x in rows),error_indices=[x['index'] for x in rows if x.get('error')]))
canonical=[D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];canon={str(f):sha(f) for f in canonical};assert canon[str(canonical[-1])]==LOCALSTAT['SHA256']=='f2f05d07e779323349b936d505c72984027ef147ba44ddf2f5fed751e34e164d'
facts=dict(UTC=datetime.now(timezone.utc).isoformat(),parent_handoff=str(parent/'handoff.txt'),parent_handoff_SHA256=sha(parent/'handoff.txt'),parent_facts=str(parent/'facts.json'),parent_facts_SHA256=sha(parent/'facts.json'),parent_closing_SHA256=sha(parent/'closing_addendum.json'),parent_current_pointer_original_SHA256=hashlib.sha256(before_ptr).hexdigest(),full_ancestry_and_all_original_families_user_decisions_failure_and_retention_exceptions_inherited=True,canonical_counts=dict(delivered=177,numeric_within=162,numeric_outside=15,own_complete_weights_pending_delivery=213,confirmed_missing=310,full_original_protocol_qualified=63),canonical_SHA256=canon,local_workbook_fresh_stat=LOCALSTAT,actual_procs=project_processes(),actual_FLOCK=runner_lock_owners(),free_bytes=shutil.disk_usage(D).free,current_Granite_job=str(current),current_Granite_job_SHA256=sha(current/'job.json'),current_Granite_state=js(current/'state.json') if (current/'state.json').exists() else None,current_Granite_completion=js(current/'completion.json') if (current/'completion.json').exists() else None,current_Granite_controller_state=js(Hlive/'state.json') if (Hlive/'state.json').exists() else None,current_Granite_controller_completion=js(Hlive/'actual_completion.json') if (Hlive/'actual_completion.json').exists() else None,Granite_all_actual_trace_snapshots=traces,Dead177_formal_score=.805,Dead177_all_three_relative_error='0.02547770700636942675159235669',Dead177_two_books_delivered=True,Pali_completed_true3_proof_SHA256='03f7e0fa3798b61836ac126f7980b2d53743aaf6b02f5c4f5caf8671a65286d9',Pali_formal200_not_approved_current_capacity_refused=True,all_complete_reports_sources_configs_logs_permanently_preserved=True,no_new_deletion_since_exact_Coder170=True,retention_ledger_SHA256='26c1b267f85c943f742f050b144e0c398339757dd5211d81bf6d96a4712a7d1b',CodeDay17_commit='4692dfe99e6c16b1fc0f9d609037e1d56dc5815d',CodeDay17_no_second_commit=True,new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_primary_source_count=len(records),new_primary_source_count=len(records)-len(js(old)['all_current_primary_source_and_true_consumer_configs']),all_new_primary_source_snapshots=[x for x in records if str(x['snapshot']).startswith(str(H))],small_originals=small,local_originals=local_originals,rejected_unconsumed_prototypes=excluded_originals)
put(H/'facts.json',facts);closing=dict(UTC=datetime.now(timezone.utc).isoformat(),latest_actual_snapshot_is_facts=True,late_actual_notes=[],new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_counts=facts['canonical_counts'],actual_NN_or_NF_and_original_FLOCK_must_be_rechecked=True);put(H/'closing_addendum.json',closing)
assert ptrfile.read_bytes()==before_ptr
newptr=dict(ptr);newptr.update(UTC=datetime.now(timezone.utc).isoformat(),handoff_directory=str(H),handoff=str(H/'handoff.txt'),handoff_path=str(H/'handoff.txt'),handoff_SHA256=sha(H/'handoff.txt'),handoff_sha256=sha(H/'handoff.txt'),facts=str(H/'facts.json'),facts_path=str(H/'facts.json'),facts_SHA256=sha(H/'facts.json'),facts_sha256=sha(H/'facts.json'),closing_addendum=str(H/'closing_addendum.json'),closing_addendum_SHA256=sha(H/'closing_addendum.json'),new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),parent_handoff_directory=str(parent),verified_delivered_count=177)
# Known alias keys from previous pointers are updated together, never stale.
for k in list(newptr):
 if k.lower() in ['handoff_txt','handoff_file']:newptr[k]=str(H/'handoff.txt')
 if k.lower() in ['handoff_txt_sha256','handoff_file_sha256']:newptr[k]=sha(H/'handoff.txt')
 if k.lower()=='facts_json':newptr[k]=str(H/'facts.json')
 if k.lower()=='facts_json_sha256':newptr[k]=sha(H/'facts.json')
tmp=ptrfile.with_name(ptrfile.name+'.verified177.tmp');assert not tmp.exists();tmp.write_bytes(json.dumps(newptr,ensure_ascii=False,indent=2).encode());os.replace(tmp,ptrfile)
print(json.dumps(dict(passed=True,handoff_directory=str(H),handoff_SHA256=sha(H/'handoff.txt'),facts_SHA256=sha(H/'facts.json'),closing_SHA256=sha(H/'closing_addendum.json'),pointer_SHA256=sha(ptrfile),pending=str(new),pending_SHA256=sha(new),primary_source_count=len(records),new_primary_source_count=facts['new_primary_source_count'],small_original_count=len(small),local_original_count=len(local_originals),actual_procs=facts['actual_procs'],FLOCK=facts['actual_FLOCK'],Granite_traces=traces,Granite_job_SHA256=sha(current/'job.json')),ensure_ascii=False))
'''.replace('__Q__',repr(Q)).replace('__W__',repr(W)).replace('__LOCAL__',repr(LOCAL)).replace('__LOCALSTAT__',repr(LOCALSTAT)).replace('__BODY__',repr(base64.b64encode(body).decode()))
ast.parse(remote);(P/'archive_verified177_Dead_Pali_Granite41_all_primary_and_handoff_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(local_packets=len(Q),useful_Windows_sources=len(W),body_SHA256=hashlib.sha256(body).hexdigest(),automation_before_SHA256=hashlib.sha256(auto).hexdigest(),local_workbook=LOCALSTAT,remote_source_bytes=len(remote.encode()))))
