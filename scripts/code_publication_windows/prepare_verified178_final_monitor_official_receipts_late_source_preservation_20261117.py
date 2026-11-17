import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,base64,ast
P=Path(__file__).parent
monitor=json.loads((P/'R_verified178_final_complete_monitor_real_stage_20261117.json').read_bytes());latest=monitor['latest_complete'];cp=monitor['checkpoint_complete']
review=dict(UTC=monitor['UTC'],all_latest_and_checkpoint_records_parsed=True,checkpoint_field_counts={k:len(v) if isinstance(v,(dict,list)) else v for k,v in cp.items()},canonical_counts=latest['canonical_counts'],source_errors=latest['source_errors'],inspection_errors=latest['inspection_errors'],new_log_error_candidates=latest['new_log_error_candidates'],new_or_modified_sources=latest['new_or_modified_source_config_paths'],affected_models=latest['affected_models_requiring_targeted_review'],log_pending_bytes=sum(v.get('pending_bytes',0) for v in latest['log_deltas']),all_original_log_rows_retained_in_complete_packet=True,actual_project_procs=monitor['actual_procs'],actual_FLOCK=monitor['actual_FLOCK'],formal_NN_or_independent_verifier_active=monitor['actual_formal_NN_or_independent_verifier_running'],Merve_capacity_refusal_preserved=True,Phi35_CPU_frontend_only_not_formal_NN=True,no_new_matrix_deletion=True)
print(json.dumps(review,ensure_ascii=False,indent=2))
assert latest['canonical_counts']=={'done':178,'skipped_no_complete_weights':310,'needs_adaptation':212} and not latest['source_errors'] and not latest['inspection_errors'] and review['log_pending_bytes']==0
assert not latest['new_log_error_candidates'],latest['new_log_error_candidates']
assert not monitor['actual_procs'] and not monitor['actual_FLOCK'] and not monitor['actual_formal_NN_or_independent_verifier_running']
(P/'verified178_final_complete_incremental_review_20261117.json').write_bytes(json.dumps(review,ensure_ascii=False,indent=2).encode())
names=['read_verified178_exact_export_current_handoff_no_mutation_20261117_remote.py','save_verified178_exact_local_baseline_and_official_prompt_20261117.py','read_verified178_official_update_arguments_exact_20261117.py','capture_verified178_official_update_readonly_and_final_local_baseline_20261117.py','read_verified178_final_incremental_monitor_and_real_idle_20261117_remote.py','prepare_verified178_final_monitor_official_receipts_late_source_preservation_20261117.py']
def enc(p):
 raw=Path(p).read_bytes();return dict(data=base64.b64encode(raw).decode(),SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
W={n:enc(P/n) for n in names};Q={n:enc(P/n) for n in ['910b_verified178_official_update_readonly_verified_20261117.json','Granite41_verified178_fresh_local_workbook_stat_baseline_20261117.json','R_verified178_final_complete_monitor_real_stage_20261117.json','verified178_final_complete_incremental_review_20261117.json','Granite41_verified178_incremental_real_results_export_20261117.json','910b_verified178_expected_official_prompt_20261117.txt']}
remote='''import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114';W=__W__;Q=__Q__
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2).encode())
pf=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';rawptr=pf.read_bytes();ptr=js(pf);H=Path(ptr['handoff_directory']);assert sha(H/'handoff.txt')=='76306766f495fef6777762f6b2ab9018582de15d7df1d826e1b02f1d1df58282' and sha(H/'facts.json')=='186789a3a946e2d8fc7c677a75d10ce9f3142057543a7d0fa3d0eda2072fa7af';old=Path(ptr['new_primary_pending']);assert sha(old)=='5bb696846d017300b9e72052ee0cca1d21fe9243f19c24e8db964bfc03a44914';pending=js(old);records=list(pending['all_current_primary_source_and_true_consumer_configs']);known={v['path']:v for v in records}
note=H/'late_actual_notes/verified178_official_update_final_incremental_review_and_source_closure_20261117';note.mkdir(parents=True,exist_ok=False);snapshots={}
for n,v in W.items():
 raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];ast.parse(raw);p=R/'scripts/code_publication_windows'/n
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
 f=note/'primary_source_original_bytes'/(n+'.txt');f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw);rec=dict(path=str(p),relative=str(p.relative_to(R)),SHA256=v['SHA256'],bytes=len(raw),snapshot=str(f));snapshots[n]=rec
 if str(p) in known:assert known[str(p)]['SHA256']==v['SHA256']
 else:records.append(rec);known[str(p)]=rec
for n,v in Q.items():
 raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['SHA256'];f=note/'local_original_bytes'/(n+'.original');f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
for v in records:assert sha(v['path'])==v['SHA256'],v['path']
new=C/'pending_after_daily_20261117_verified178_final_official_incremental_all_primary_v12.json';pending.update(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(old),parent_pending_SHA256=sha(old),all_current_primary_source_and_true_consumer_configs=records,all_final_official_update_maintenance_sources_original_bytes_preserved=True,CodeDay17_already_pushed_no_second_commit=True);put(new,pending)
review=json.loads(base64.b64decode(Q['verified178_final_complete_incremental_review_20261117.json']['data']));official=json.loads(base64.b64decode(Q['910b_verified178_official_update_readonly_verified_20261117.json']['data']));assert official['verified'] and official['only_prompt_and_updated_at_changed'] and official['non_prompt_settings_and_all_keys_preserved'] and official['notification_policy_absent_and_not_reset'];assert not review['formal_NN_or_independent_verifier_active'] and not review['actual_project_procs'] and not review['actual_FLOCK']
for name,h in js(H/'facts.json')['canonical_SHA256'].items():assert sha(name)==h
closing_file=H/'closing_addendum.json';closing_raw=closing_file.read_bytes();(note/'closing_before.original').write_bytes(closing_raw);closing=json.loads(closing_raw)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),complete_incremental_review=review,official_update_only_via_official_tool_verified=official,new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),primary_source_count=len(records),new_late_primary_source_count=len(records)-len(js(old)['all_current_primary_source_and_true_consumer_configs']),all_source_snapshots=snapshots,no_model_report_weight_config_runtime_or_canonical_workbook_state_registry_write=True,no_new_matrix_deletion=True)
put(note/'actual.json',v);closing.update(UTC=v['UTC'],latest_actual_snapshot_is_facts=False,latest_actual_note=str(note/'actual.json'),latest_actual_note_SHA256=sha(note/'actual.json'),new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),current_primary_source_count=len(records),late_actual_notes=closing.get('late_actual_notes',[])+[dict(path=str(note/'actual.json'),SHA256=sha(note/'actual.json'))],official_910b_update_verified=True,official_TOML_after_SHA256=official['automation_after_SHA256'],official_prompt_after_SHA256=official['prompt_after_SHA256'],all_non_prompt_automation_settings_preserved=True,notification_policy_absence_preserved=True,no_actual_NN_or_NF_and_no_FLOCK=True)
assert closing_file.read_bytes()==closing_raw;tmp=closing_file.with_name('closing_addendum.final178.tmp');assert not tmp.exists();tmp.write_bytes(json.dumps(closing,ensure_ascii=False,indent=2).encode());os.replace(tmp,closing_file)
assert pf.read_bytes()==rawptr;newptr=dict(ptr);newptr.update(UTC=v['UTC'],closing_addendum_SHA256=sha(closing_file),new_primary_pending=str(new),new_primary_pending_SHA256=sha(new),latest_actual_note=str(note/'actual.json'),latest_actual_note_SHA256=sha(note/'actual.json'),current_primary_source_count=len(records));tmp=pf.with_name('A100_ALIGNMENT_HANDOFF_CURRENT.final178.tmp');assert not tmp.exists();tmp.write_bytes(json.dumps(newptr,ensure_ascii=False,indent=2).encode());os.replace(tmp,pf)
print(json.dumps(dict(passed=True,UTC=v['UTC'],note=str(note/'actual.json'),note_SHA256=sha(note/'actual.json'),closing_SHA256=sha(closing_file),pointer_SHA256=sha(pf),primary_source_count=len(records),new_pending=str(new),new_pending_SHA256=sha(new),official_TOML_after_SHA256=official['automation_after_SHA256'],all_non_prompt_settings_and_notification_absence_preserved=True,current_actual_NN_NF_FLOCK_empty=True,no_new_matrix_deleted=True),ensure_ascii=False))
'''.replace('__W__',repr(W)).replace('__Q__',repr(Q))
ast.parse(remote);(P/'archive_verified178_final_official_incremental_late_sources_20261117_remote.py').write_bytes(remote.encode());print(json.dumps(dict(prepared=True,remote_source_bytes=len(remote.encode()),late_useful_source_count=len(W),actual_original_packet_count=len(Q))))
