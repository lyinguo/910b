"""Create one compact successor handoff; canonical/model/job/source bytes stay fixed."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114';sys.path[:0]=[str(R),str(R/'scripts')]
PUB=B/'AutoGLMPhone9BSingle_original_BF16_2048_LZMA_verified183_blank_G_publication_20261120'
STAGE=B/'GLM41Thinking_own_native_inputs_true3_full_capacity_fresh3_fixed200_serial_v1_20261120'
J3=B/'GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120'
J200=B/'GLM41Thinking_own_completed_true3_full_shape_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261120'
COUNTS=dict(delivered=183,numeric_within=168,numeric_outside=15,complete_weight_pending_delivery=207,confirmed_missing=310,full_protocol_qualified=69,numeric_only=99,complete_weight_still_needing_adaptation_qualification_or_repair=321)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def record(p):
 p=Path(p)
 if not p.is_file():return dict(path=str(p),exists=False)
 raw=p.read_bytes();v=dict(path=str(p),exists=True,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
 try:v['value']=json.loads(raw)
 except ValueError:v['complete_text']=raw.decode(errors='replace')
 return v
def actual():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 owners=runner_lock_owners();procs=project_processes();details=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   args=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
   if not args or not Path(args[0]).name.startswith('python'):continue
   cwd=os.readlink(p/'cwd')
   if not any(str(R) in s for s in args) and not Path(cwd).is_relative_to(R):continue
   if not any('GLM41Thinking' in s or 'GLM-4.1V-9B-Thinking' in s for s in args) and int(p.name) not in {x['PID'] for x in owners}:continue
   details.append(dict(PID=int(p.name),argv=args,cwd=cwd,stat=(p/'stat').read_text(),io=(p/'io').read_text(),status=(p/'status').read_text()))
  except FileNotFoundError:continue
 files={}
 for home in [STAGE,J3,J200]:
  for name in ['state.json','actual_completion.json','completion.json','own_passed3_complete_shape_capacity_review.json']:
   p=home/name
   if p.is_file():files[str(p)]=record(p)
 captures=[]
 for home in sorted((R/'workspace/diagnostics/model_runs').glob('GLM41Thinking_*')):
  tr=home/'actual_sample_trace.jsonl';lines=tr.read_bytes().splitlines(keepends=True) if tr.is_file() else [];rows=[json.loads(s) for s in lines if s.endswith(b'\n')]
  captures.append(dict(directory=str(home),trace_SHA256=sha(tr) if tr.is_file() else None,complete_samples=len(rows),committed_forwards=sum(x['actual_native_forwards'] for x in rows),pending_incomplete_line_bytes=sum(len(s) for s in lines if not s.endswith(b'\n')),last_complete_sample=rows[-1] if rows else None))
 model_process=any(any(Path(s).name=='run.py' for s in x['argv'][1:]) for x in details);nf_process=any(any(Path(s).name.startswith('verify_glm41thinking') for s in x['argv'][1:]) for x in details)
 nn_confirmed=model_process and any(x['complete_samples']>0 for x in captures);active=nn_confirmed or nf_process
 return dict(UTC=datetime.now(timezone.utc).isoformat(),actual_project_process_snapshot=procs,actual_current_owned_proc_details_including_rootcwd=details,actual_original_FLOCK=owners,current_stage_files=files,current_owned_captures=captures,current_actual_model_process_or_NF_active=model_process or nf_process,current_actual_model_NN_or_NF_active=active,current_actual_model_NN_committed_forward_confirmed=nn_confirmed,free_bytes=shutil.disk_usage(R).free,max_concurrent_models=1,no_model_signal_restart_replay_or_matrix_deletion=True)
def execute(plan_path,expected):
 assert sha(plan_path)==expected;plan=js(plan_path);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';oldraw=pointer.read_bytes();assert hashlib.sha256(oldraw).hexdigest()==plan['expected_pointer_SHA256'];old=json.loads(oldraw)
 H=Path(plan['handoff_directory']);assert H.parent==D/'heartbeat_handoffs' and not H.exists();parent=Path(old['handoff_directory'])
 assert sha(parent/'handoff.txt')==old['handoff_SHA256'] and sha(parent/'facts.json')==old['facts_SHA256']
 audit_path=PUB/'full_target_dual_publication_audit.json';assert sha(audit_path)=='2fe185cc1b6926e8e45adca502e4793168bea2261112b60f00738c5ed6150114';audit=js(audit_path)
 assert audit['passed'] and audit['qualified_delivered']==69 and audit['numeric_within5pct']==168 and audit['numeric_outside5pct']==15 and audit['complete_weights_pending_formal']==207 and audit['confirmed_missing']==310
 assert audit['all_non_G_formulas_styles_rows_ZIP_members_and_metadata_preserved'] and audit['old182_state_registry_G_C_B_D_preserved'] and js(PUB/'dual_publication_completed.json')['completed']
 canonical={n:sha(n) for n in old['canonical_SHA256']};expected_current=plan['canonical_SHA256'];assert canonical==expected_current
 changed={n for n,h in old['canonical_SHA256'].items() if canonical[n]!=h};allowed={str(D/'state.json'),str(D/'registry.json'),str(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json'),str(R.parent/'C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')};assert changed==allowed
 pending_path=Path(old['new_primary_pending']);assert sha(pending_path)==old['new_primary_pending_SHA256'];parent_pending=js(pending_path);rows={v['path']:v for v in parent_pending['all_current_primary_source_and_true_consumer_configs']}
 assert len(rows)==parent_pending['source_count']==2362
 for n,v in rows.items():assert sha(n)==v['SHA256'],n
 H.mkdir();(H/'parent_current_pointer_original.json').write_bytes(oldraw)
 text=Path(plan['full_handoff_text']);assert sha(text)==plan['full_handoff_text_SHA256'];(H/'handoff.txt').write_bytes(text.read_bytes())
 for item in plan['local_original_records']:
  source=Path(item['path']);assert source.is_relative_to(Path(plan['incoming_originals_root'])) and sha(source)==item['SHA256'];target=H/'local_original_records'/item['name'];assert target.is_relative_to(H/'local_original_records');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
 for item in plan['additional_primary_sources']:
  p=(R/item['relative']).resolve();assert p.is_relative_to(R) and sha(p)==item['SHA256'];rel=str(p.relative_to(R));assert rel.startswith(('scripts/','configs/','universal_eval/'))
  snap=H/'primary_source_snapshots'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=rel,SHA256=sha(p),bytes=p.stat().st_size,snapshot=str(snap),role=item['role'])
 # Include only this model's actual generated YAML/protocol/smoke declarations.
 for home in [J3,J200]:
  if not (home/'job.json').is_file():continue
  job=js(home/'job.json')
  for n,h in job['spec']['source_sha256'].items():
   p=R/n;assert sha(p)==h
   if str(p) in rows:continue
   snap=H/'primary_source_snapshots'/n;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(snap),role='current_owned_runtime_or_independent_verifier_dependency')
  base=R/job['spec']['config'];smoke=base.with_name(base.stem+'-smoke'+base.suffix)
  if smoke.is_file() and str(smoke) not in rows:
   rel=str(smoke.relative_to(R));snap=H/'primary_source_snapshots'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(smoke.read_bytes());rows[str(smoke)]=dict(path=str(smoke),relative=rel,SHA256=sha(smoke),bytes=smoke.stat().st_size,snapshot=str(snap),role='actual_frozen_runner_generated_own_smoke_configuration')
 for p in (R/'scripts/tests').glob('*AutoGLM_Phone_9B_verified183_original_writer_isolated_20261119.py'):
  rel=str(p.relative_to(R));snap=H/'primary_source_snapshots'/rel;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=rel,SHA256=sha(p),bytes=p.stat().st_size,snapshot=str(snap),role='actual183_publication_11_isolated_frozen_writer_tests')
 fresh=actual();pending_new=C/plan['new_pending_name'];assert not pending_new.exists()
 pending=dict(UTC=fresh['UTC'],parent_pending=str(pending_path),parent_pending_SHA256=sha(pending_path),Code_Day_already_pushed='2026-11-20',no_second_same_day_commit=True,all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],source_count=len(rows),next_daily_review='2026-11-21: complete new own source/config/controls/tests/dependencies; reviewed plan_SHA then one normal push; no second Day20 commit',only_source_or_actual_consumer_configuration_not_run_artifacts=True,handoff_directory=str(H))
 put(pending_new,pending)
 transition=dict(passed=True,authorization='Existing own target G/state/registry/receipt authorization after fullNPU200/allnumeric/originalprotocol/threeD5',old_verified_count=182,new_verified_count=183,old_canonical_SHA256=old['canonical_SHA256'],new_canonical_SHA256=canonical,exact_changed_canonical_paths=sorted(changed),target_model='zai-org/AutoGLM-Phone-9B',only_G_cell='G692',old_G=None,new_G=.795,full_target_audit=str(audit_path),full_target_audit_SHA256=sha(audit_path),no_other_G_or_non_G_or_old_report_changes=True)
 put(H/'exact_authorized_verified183_dual_canonical_transition.json',transition)
 put(H/'late_actual_notes/current_actual.json',fresh)
 refs={str(p):sha(p) for p in [parent/'handoff.txt',parent/'facts.json',parent/'closing_addendum.json',pending_path]}
 facts=dict(UTC=fresh['UTC'],automation_id='910b',canonical_counts=COUNTS,canonical_SHA256=canonical,parent_handoff=str(parent),parent_files_SHA256=refs,all_ancestors_and_original_full_65471_character_archive_inherited=True,current_publication=dict(home=str(PUB),target='zai-org/AutoGLM-Phone-9B',full200_samples=200,actual_native_forwards=12194,NPU=4,score=.795,three_D=['.7800','.7800','.78'],relative_errors_each='0.01923076923076923076923076923',actual_full_proof_SHA256='f1e8c91ba38b9c535962062e720904fe4f970e4f360dbf0133da2d176023ce62',full_dual_audit_SHA256=sha(audit_path),export=str(PUB/'verified183_export_complete_metadata.json'),export_SHA256=sha(PUB/'verified183_export_complete_metadata.json'),active_full_report_stat_baseline=str(PUB/'active183_full_report_stat_baseline.json'),active_full_report_stat_baseline_SHA256=sha(PUB/'active183_full_report_stat_baseline.json')),primary_source_manifest=str(pending_new),primary_source_manifest_SHA256=sha(pending_new),current_primary_source_count=len(rows),all_preexisting_2362_primary_SHA256_fresh_verified=True,new_primary_original_bytes_snapshotted=True,local_original_records=plan['local_original_records'],current_owned_actual=fresh,all_old_reports_and_accepted_proofs_reused_no_old_CPU200_or_weight_revalidation=True,retention_ledger_SHA256=canonical[str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json')],this_turn_matrix_deletions=0,original_report_source_config_log_and_necessary_value_examples_preserved=True,current_Code_Day_commit='be09d9a8b637a6dd54f9be65db8609e89af92f64',official_automation_update_pending=True)
 put(H/'facts.json',facts)
 closing=dict(UTC=fresh['UTC'],initial_body_and_facts_are_snapshot_later_real_completion_proc_FLOCK_priority=True,current_canonical_counts=COUNTS,current_canonical_SHA256=canonical,latest_actual_note=str(H/'late_actual_notes/current_actual.json'),latest_actual_note_SHA256=sha(H/'late_actual_notes/current_actual.json'),latest_pending=str(pending_new),latest_pending_SHA256=sha(pending_new),current_source_count=len(rows),official_automation_update_only=True,active_TOML_never_raw_edited=True,this_turn_matrix_deletions=0,all_full_reports_sources_configs_logs_examples_retained=True)
 put(H/'closing_addendum.json',closing)
 ptr=dict(old);ptr.update(UTC=fresh['UTC'],handoff_directory=str(H),parent_handoff_directory=str(parent),parent_handoff=str(parent/'handoff.txt'),parent_handoff_path=str(parent/'handoff.txt'),parent_handoff_txt=str(parent/'handoff.txt'),parent_handoff_SHA256=sha(parent/'handoff.txt'),parent_facts=str(parent/'facts.json'),parent_facts_path=str(parent/'facts.json'),parent_facts_json=str(parent/'facts.json'),parent_facts_SHA256=sha(parent/'facts.json'),canonical_counts=COUNTS,current_canonical_counts=COUNTS,verified_delivered_count=183,canonical_SHA256=canonical,new_primary_pending=str(pending_new),new_primary_pending_SHA256=sha(pending_new),current_primary_source_count=len(rows),current_actual_job=str(J200 if J200.exists() else J3),current_actual_controller=str(STAGE),current_actual_model_process_or_NF_active=fresh['current_actual_model_process_or_NF_active'],current_actual_model_NN_or_NF_active=fresh['current_actual_model_NN_or_NF_active'],current_actual_stage='GLM41Thinking_own_original8192_serial_actual_see_latest_note',current_formal_model_stage='own_true3_then_full_capacity_NEWfresh3_fixed200_serial',current_formal_acceptance_home=str(J3),current_formal_acceptance_completed=(J3/'completion.json').exists(),current_current_owned_candidate='zai-org/GLM-4.1V-9B-Thinking',current_owned_candidate='zai-org/GLM-4.1V-9B-Thinking',current_daily_code_commit='be09d9a8b637a6dd54f9be65db8609e89af92f64',Code_Day_already_pushed='2026-11-20',Day20_commit='be09d9a8b637a6dd54f9be65db8609e89af92f64',severity_first_line=None if fresh['current_actual_model_NN_or_NF_active'] else '严重阻塞｜正式评测未运行',first_proven_stoppage_BJT=None if fresh['current_actual_model_NN_or_NF_active'] else 'See latest actual stage completion; no cross-recovery accumulation',last_complete_result_BJT='2026-11-20 06:06:06.525367+08:00',latest_actual_note=str(H/'late_actual_notes/current_actual.json'),latest_actual_note_SHA256=sha(H/'late_actual_notes/current_actual.json'),latest_actual_snapshot=str(H/'late_actual_notes/current_actual.json'),latest_actual_snapshot_SHA256=sha(H/'late_actual_notes/current_actual.json'),exact_authorized_AutoGLM183_transition=str(H/'exact_authorized_verified183_dual_canonical_transition.json'),exact_authorized_AutoGLM183_transition_SHA256=sha(H/'exact_authorized_verified183_dual_canonical_transition.json'))
 for k in ['handoff','handoff_path','handoff_txt']:ptr[k]=str(H/'handoff.txt')
 for k in ['handoff_SHA256','handoff_sha256','handoff_txt_SHA256','handoff_text_SHA256']:ptr[k]=sha(H/'handoff.txt')
 for k in ['facts','facts_path','facts_json']:ptr[k]=str(H/'facts.json')
 for k in ['facts_SHA256','facts_sha256','facts_json_SHA256']:ptr[k]=sha(H/'facts.json')
 for k in ['closing_addendum','closing_addendum_path']:ptr[k]=str(H/'closing_addendum.json')
 ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');assert pointer.read_bytes()==oldraw
 assert {n:sha(n) for n in canonical}==canonical
 for n,v in rows.items():assert sha(n)==v['SHA256'],n
 put(pointer,ptr)
 result=dict(passed=True,handoff_directory=str(H),handoff_SHA256=sha(H/'handoff.txt'),facts_SHA256=sha(H/'facts.json'),closing_addendum_SHA256=sha(H/'closing_addendum.json'),pointer_SHA256=sha(pointer),pending=str(pending_new),pending_SHA256=sha(pending_new),current_primary_source_count=len(rows),canonical_counts=COUNTS,all_canonical_and_live_sources_jobs_unchanged=True,no_matrix_delete=True)
 put(H/'creation_actual_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('plan');ap.add_argument('SHA256');a=ap.parse_args();execute(Path(a.plan),a.SHA256)
