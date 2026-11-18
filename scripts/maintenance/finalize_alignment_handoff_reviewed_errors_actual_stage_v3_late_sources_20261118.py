"""Close official heartbeat metadata while retaining resolved historical failures."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,base64,contextlib,hashlib,io,json,os,shutil
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def main(plan_path,expected):
 assert sha(plan_path)==expected;plan=js(plan_path);R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=Path(plan['handoff_directory']);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'
 assert H.resolve().is_relative_to((D/'heartbeat_handoffs').resolve()) and sha(pointer)==plan['pointer_SHA256'];before=pointer.read_bytes();p=js(pointer);assert p['handoff_directory']==str(H)
 for field in ['handoff','handoff_path','handoff_txt']:assert p[field]==str(H/'handoff.txt')
 for field in ['facts','facts_path','facts_json']:assert p[field]==str(H/'facts.json')
 assert sha(H/'handoff.txt')==p['handoff_SHA256'] and sha(H/'facts.json')==p['facts_SHA256']
 late=H/'late_actual_notes/official_readback_and_current_stage_20261118';late.mkdir(exist_ok=False,parents=True);(late/'prior_pointer_original.json').write_bytes(before);(late/'prior_closing_original.json').write_bytes((H/'closing_addendum.json').read_bytes())
 for item in plan['local_original_artifacts']:
  raw=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['SHA256'];dest=late/'local_original_packets'/item['archive_name'];assert dest.resolve().is_relative_to(late.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 pending_path=Path(p['new_primary_pending']);assert sha(pending_path)==p['new_primary_pending_SHA256'];pending=js(pending_path)
 rows={row['path']:dict(row) for row in pending['all_current_primary_source_and_true_consumer_configs']}
 for row in rows.values():assert sha(row['path'])==row['SHA256'],row['path']
 for source in plan.get('additional_existing_primary_sources',[]):
  path=(R/source['relative']).resolve();assert path.is_relative_to(R.resolve()) and source['relative'].startswith('scripts/')
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==source['SHA256'];ast.parse(raw)
  snapshot=late/'primary_source_snapshots'/source['relative'];snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(raw)
  assert str(path) not in rows or rows[str(path)]['SHA256']==source['SHA256']
  rows[str(path)]=dict(path=str(path),relative=source['relative'],SHA256=source['SHA256'],bytes=len(raw),snapshot=str(snapshot),role=source['role'])
 new_pending=Path(plan['new_pending']);assert new_pending.parent==pending_path.parent and not new_pending.exists()
 pending.update(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(pending_path),parent_pending_SHA256=sha(pending_path),all_current_primary_source_and_true_consumer_configs=sorted(rows.values(),key=lambda r:r['path']),source_count=len(rows),official_readback_late_source_closure=True)
 save(new_pending,pending);p.update(new_primary_pending=str(new_pending),new_primary_pending_SHA256=sha(new_pending),current_primary_source_count=len(rows))
 for path,h in p['canonical_SHA256'].items():assert sha(path)==h,path
 for path,h in plan['reviewed_evidence_SHA256'].items():assert sha(path)==h,path
 sys.path[:0]=[str(R),str(R/'scripts')];import A100_alignment_incremental_monitor_20261103 as monitor
 assert sha(R/'scripts/A100_alignment_incremental_monitor_20261103.py')=='2103c7e9c86d1fc7928f1d00228ba90ffc93cdfce13bcb91d654cbea6ba4826d'
 with contextlib.redirect_stdout(io.StringIO()):monitor.main()
 latest=js(D/'incremental_monitor/latest.json');checkpoint=js(D/'incremental_monitor/checkpoint.json');save(late/'frozen_monitor_complete_latest.json',latest);save(late/'frozen_monitor_complete_checkpoint.json',checkpoint)
 assert not latest['source_errors']
 known_inspections=plan['reviewed_original_failed_jobs']
 for failure in latest['inspection_errors']:
  assert failure['job'] in known_inspections and failure['actual_returncode']==1,failure
  completion=Path(failure['job']).parent/'completion.json';assert sha(completion)==known_inspections[failure['job']],str(completion)
 known_logs=plan['reviewed_original_error_logs']
 for candidate in latest['new_log_error_candidates']:assert candidate['path'] in known_logs and sha(candidate['path'])==known_logs[candidate['path']],candidate
 for delta in latest['log_deltas']:assert delta.get('pending_bytes',0)==0 and not delta.get('oversized_or_incomplete_line',False),delta
 B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/plan['current_job_name'];C=B/plan['current_controller_name'];state=js(J/'state.json') if (J/'state.json').exists() else None;completion=js(J/'completion.json') if (J/'completion.json').exists() else None;controller=js(C/'state.json') if (C/'state.json').exists() else None;controller_completion=js(C/'actual_completion.json') if (C/'actual_completion.json').exists() else None
 actual=dict(UTC=datetime.now(timezone.utc).isoformat(),actual_project_processes=monitor.project_processes(),actual_original_kernel_FLOCK=monitor.runner_lock_owners(),actual_free_bytes=shutil.disk_usage(R).free,current_job=str(J),current_job_state=state,current_job_actual_completion=completion,current_controller_state=controller,current_controller_actual_completion=controller_completion,monitor_reports_model_or_verifier_process_active=latest['inference_or_verification_active'],current_NPU200_claim=False,canonical_counts=p['canonical_counts'],canonical_SHA256=p['canonical_SHA256'],all_monitor_source_errors=latest['source_errors'],all_monitor_inspection_errors=latest['inspection_errors'],all_monitor_new_log_error_candidates=latest['new_log_error_candidates'],reviewed_original_failure_contexts_preserved=True,new_primary_pending=str(new_pending),new_primary_pending_SHA256=sha(new_pending),current_primary_source_count=len(rows),reviewed_evidence_SHA256=plan['reviewed_evidence_SHA256'],official_readback=plan['official_readback'],no_matrix_deletion_this_cycle=True,Day18_already_pushed_commit='02df1887796f49c830c63db01f4dcdb2bb8e4298',no_duplicate_same_day_commit=True,no_old_failed_job_or_report_or_frozen_source_mutation=True)
 if completion and completion['actual_returncode']!=0:actual['current_unrecovered_run_error']=completion
 if controller_completion and controller_completion['actual_returncode']!=0:actual['current_controller_error_or_capacity_block']=controller_completion
 save(late/'actual.json',actual)
 closing=js(H/'closing_addendum.json');closing.update(UTC=actual['UTC'],latest_actual_note=str(late/'actual.json'),latest_actual_note_SHA256=sha(late/'actual.json'),official_readback=plan['official_readback'],latest_current_job_state=state,latest_current_job_completion=completion,latest_current_controller_completion=controller_completion,latest_pending=str(new_pending),latest_pending_SHA256=sha(new_pending),current_source_count=len(rows),monitor_all_new_errors_explicitly_context_reviewed_and_preserved=True)
 save(H/'closing_addendum.json',closing)
 for field in ['latest_actual_note','latest_actual_snapshot']:p[field]=str(late/'actual.json')
 for field in ['latest_actual_note_SHA256','latest_actual_snapshot_SHA256']:p[field]=sha(late/'actual.json')
 p.update(UTC=actual['UTC'],closing_addendum_SHA256=sha(H/'closing_addendum.json'),latest_official_automation_readback_SHA256=plan['official_readback_SHA256'],current_actual_job=str(J),current_actual_controller=str(C),current_actual_model_process_or_NF_active=latest['inference_or_verification_active'])
 for path,h in p['canonical_SHA256'].items():assert sha(path)==h,path
 for row in pending['all_current_primary_source_and_true_consumer_configs']:assert sha(row['path'])==row['SHA256'],row['path']
 assert pointer.read_bytes()==before;tmp=pointer.with_suffix('.reviewed_error_closure.tmp');assert not tmp.exists();save(tmp,p);os.replace(tmp,pointer)
 return dict(passed=True,UTC=actual['UTC'],handoff_directory=str(H),closing_addendum_SHA256=sha(H/'closing_addendum.json'),latest_actual_note=str(late/'actual.json'),latest_actual_note_SHA256=sha(late/'actual.json'),pointer_SHA256=sha(pointer),canonical_and_all_primary_source_SHA256_unchanged=True,known_historical_failure_records_not_erased=True,current_job_state=state,current_job_completion=completion,current_controller_completion=controller_completion,actual_original_FLOCK=actual['actual_original_kernel_FLOCK'],new_primary_pending=str(new_pending),new_primary_pending_SHA256=sha(new_pending),current_primary_source_count=len(rows))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);args=ap.parse_args();print(json.dumps(main(args.plan,args.plan_sha256),ensure_ascii=False))
