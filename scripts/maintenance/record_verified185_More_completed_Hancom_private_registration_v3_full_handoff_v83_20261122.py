"""Preserve exact new source/configuration consumers and current actual serial evidence."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pointer.read_bytes();assert hashlib.sha256(before).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(before);H=Path(ptr['handoff_directory']);closing=H/'closing_addendum.json';cb=closing.read_bytes();assert hashlib.sha256(cb).hexdigest()==q['expected_closing_SHA256'];parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_pending_SHA256']
 assert ptr['current_verified_count']==185 and all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
 rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert len(rows)==2842
 assert all(sha(n)==r['SHA256'] for n,r in rows.items()) and all(sha(R/n)==h for n,h in q['new_primary_expected_SHA256'].items())
 note=H/'late_actual_notes'/q['round_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(before);(note/'previous_closing_addendum.json').write_bytes(cb)
 for r in q['local_original_records']:
  b=base64.b64decode(r['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==r['SHA256'];p=note/'local_original_records'/r['name'];assert p.resolve().is_relative_to(note.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 paths=q['new_primary_paths']+[str(Path(__file__).relative_to(R))];jobs=[]
 for stem in ['HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v2_own_BF16_guard_real_FLOCK_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v3_original_registration_real_FLOCK_20261122','HancomHiLlava31_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v3_original_registration_real_FLOCK_20261122']:
  p=B/stem/'job.json'
  if not p.is_file():continue
  job=js(p);sources=job['spec']['source_sha256'];assert all(sha(R/n)==h for n,h in sources.items());paths.extend(sources);jobs.append(dict(job=str(p),SHA256=sha(p),source_SHA256=sources))
  cfg=Path(job['spec']['config'])
  for v in [cfg,cfg.with_stem(cfg.stem+'-smoke')]:
   if (R/v).is_file():paths.append(str(v))
 for n in sorted(set(paths)):
  p=R/n;assert p.is_file() and p.resolve().is_relative_to(R) and (n=='run.py' or n.startswith(('scripts/','configs/','universal_eval/','tests/')));h=sha(p)
  if str(p) in rows:assert rows[str(p)]['SHA256']==h,n
  else:
   dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),role='Own immutable Hancom v2/v3 exact source guard/original registration/full-NF/runtime YAML protocol and actual serial configuration consumer, diagnostic, maintenance or Windows review builder')
 reader=R/'scripts/maintenance/read_Hancom_private_original_registration_v3_actual_serial_v4_all_proc_cwd_20261122_remote.py';run=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert run.returncode==0,run.stderr;actual=json.loads(run.stdout);assert actual['card_returncode']==0 and not actual['frozen_monitor_summary']['source_errors']
 for n,h in ptr['canonical_SHA256'].items():assert sha(n)==h,n
 mon=actual['frozen_monitor_summary'];proc=actual['actual_proc_argv_cwd_PPid_startticks_FD_io'];closed=[];unknown=[];exact_jobs={j['job'][:-len('/job.json')] for j in jobs}
 for warning in mon.get('inspection_errors',[]):
  match=[p for p in proc if p['PID'] in {o['PID'] for o in warning.get('owners',[])}]
  ok=warning.get('reason')=='runner_lock_owner_not_in_project_process_snapshot' and len(match)==len(warning.get('owners',[])) and all(p['cwd']=='/root' and len(p['argv'])==4 and Path(p['argv'][2]).name=='A100_alignment_evidence_runner_20261103.py' and p['argv'][3] in exact_jobs and p['PPid'] in {c['PID'] for c in proc if any('continue_hancom_' in a for a in c['argv'])} for p in match)
  (closed if ok else unknown).append(warning)
 closed_log=[];unknown_log=[]
 for candidate in mon.get('new_log_error_candidates',[]):
  p=Path(candidate['path']);text=p.read_text() if p.is_file() else '';needle=candidate.get('line','');at=text.find(needle);context=text[max(0,at-250):at+len(needle)+250] if at>=0 else ''
  ok=('self.npu_exception =' in needle and 'SyntaxWarning' in context and '_error_code.py:103' in context and 'Traceback (most recent call last)' not in text)
  (closed_log if ok else unknown_log).append(dict(candidate=candidate,complete_exact_context=context))
 actual['specific_root_cwd_warning_closed_by_exact_current_job_PPid_startticks_kernel_EX']=closed;actual['unknown_inspection_errors']=unknown;actual['specific_syntax_warning_literal_error_candidates_closed']=closed_log;actual['unknown_new_log_error_candidates']=unknown_log
 snapshot=note/'More_completed_Hancom_private_v3_current_full_actual.json';put(snapshot,actual)
 model=[p for p in proc if 'run.py' in p['argv']];nf=[p for p in proc if any(Path(a).name.startswith('verify_') for a in p['argv'])];caps=actual['Hancom_all_capture_real_progress'];forwards=sum(c['total_committed_native_forwards'] for c in caps);current=caps[-1] if caps else None
 compact=dict(UTC=actual['UTC'],model_processes=model,full_numeric_verifiers=nf,actual_original_kernel_lock_owners=actual['lock_owners'],Hancom_current_capture=current,total_genuine_committed_Hancom_forwards=forwards,Hancom_current_natural_states={k:v['complete_value'] for k,v in actual['Hancom_private_v3_current_complete_records'].items() if k.endswith(('state.json','actual_completion.json','actual_failure.json'))},completed_previous_actual=actual['completed_previous_actual'],actual_free_bytes=actual['actual_original_disk_free_bytes'],closed_specific_root_cwd_warnings=closed,unknown_inspection_errors=unknown,closed_exact_syntax_warning_candidates=closed_log,unknown_new_log_error_candidates=unknown_log,full_actual_snapshot=str(snapshot),full_actual_snapshot_SHA256=sha(snapshot))
 full=q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False));fp=note/q['full_note_name'];fp.write_text(full,encoding='utf8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-22',Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',no_second_same_day_commit=True,next_daily_review='Day23 all exact source/header/registration repairs and full independent checks, real runtime config consumers, dependencies and Windows review sources; full diff/all outgoing/checks/reviewed publisher plan then one normal push'))
 facts=note/'facts.json';put(facts,dict(UTC=actual['UTC'],canonical_counts=dict(delivered=185,numeric_within=170,numeric_outside=15,complete_weights_pending=205,confirmed_missing=310,original_full_qualified=71),new_complete200_G_or_qualification_this_round=0,canonical_SHA256=ptr['canonical_SHA256'],parent_full_handoff_SHA256='9e564ea6ef78e868cdccecc57b74442de6f122614bb188a4234f9e46f7da6529',current_source_count=len(rows),primary_pending=str(pending),primary_pending_SHA256=sha(pending),current_full_actual_snapshot=str(snapshot),current_full_actual_snapshot_SHA256=sha(snapshot),new_runtime_jobs=jobs,no_existing_frontend200_constructor_or_passed_true3_replay=True,no_deletion_this_round=True,all_original_reports_sources_configs_logs_failed_and_necessary_full_values_preserved=True))
 updates=dict(UTC=datetime.now(timezone.utc).isoformat(),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_primary_source_count=len(rows),current_source_count=len(rows),latest_full_actual_summary=str(fp),latest_full_actual_summary_SHA256=sha(fp),latest_actual_note=str(fp),latest_actual_note_SHA256=sha(fp),latest_actual_snapshot=str(snapshot),latest_actual_snapshot_SHA256=sha(snapshot),latest_round_facts=str(facts),latest_round_facts_SHA256=sha(facts),current_actual_stage='More_genuine3_fullNF0_capacity_refused_Hancom_v1_v2_zeroNN_failure_original_registration_v3_actual_serial',current_formal_model_stage='Actual exact immutable Hancom v3 progress/processes/phase and current report/logs in snapshot; no loading or storage stage claimed as new genuine forwards',current_model_process_active=bool(model),current_full_numerical_verifier_active=bool(nf),current_formal_model_task_active=bool(model or nf),current_actual_model_process_or_NF_active=bool(model or nf),current_genuine_Hancom_committed_forwards=forwards,current_NN_forward_active_not_inferred_from_process=True,current_unknown_inspection_errors=unknown,current_unknown_log_error_candidates=unknown_log,current_stop_earliest_proven_Beijing='2026-11-22T10:56:52.252615+08:00' if not forwards else None,current_NN_or_NF_active=bool(nf),current_model_NN_or_NF_active=bool(nf),current_actual_model_NN_or_NF_active=bool(nf))
 assert pointer.read_bytes()==before and closing.read_bytes()==cb and all(sha(n)==h for n,h in ptr['canonical_SHA256'].items()) and all(sha(n)==z['SHA256'] for n,z in rows.items())
 co=json.loads(cb);co.update(updates);put(closing,co);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(closing);put(pointer,ptr)
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),pointer_SHA256=sha(pointer),closing_SHA256=sha(closing),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fp),full_actual_note_SHA256=sha(fp),facts=str(facts),facts_SHA256=sha(facts),actual_snapshot=str(snapshot),actual_snapshot_SHA256=sha(snapshot),Hancom_total_genuine_committed_forwards=forwards,current_model_processes=model,current_full_numeric_verifiers=nf,unknown_inspection_errors=unknown,unknown_log_error_candidates=unknown_log,no_active_source_job_config_deadline_signal_mutation=True,no_canonical_mutation=True,no_deletion=True,no_second_Day22_commit=True);put(note/'round_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
