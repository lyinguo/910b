"""Preserve complete immutable model evidence and every useful new primary source."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
js=lambda p:json.loads(Path(p).read_bytes())
def put(p,v):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pointer.read_bytes();assert hashlib.sha256(before).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(before);H=Path(ptr['handoff_directory']);closing=H/'closing_addendum.json';cb=closing.read_bytes();assert hashlib.sha256(cb).hexdigest()==q['expected_closing_SHA256'];parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_pending_SHA256']
 assert ptr['current_verified_count']==185 and all(sha(n)==h for n,h in ptr['canonical_SHA256'].items())
 rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert len(rows)==2815 and all(sha(n)==r['SHA256'] for n,r in rows.items())
 assert all(sha(R/n)==h for n,h in q['new_primary_expected_SHA256'].items())
 note=H/'late_actual_notes'/q['round_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(before);(note/'previous_closing_addendum.json').write_bytes(cb)
 for rec in q['local_original_records']:
  b=base64.b64decode(rec['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==rec['SHA256'];p=note/'local_original_records'/rec['name'];assert p.resolve().is_relative_to(note.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 paths=list(q['new_primary_paths'])+[str(Path(__file__).relative_to(R))];jobs=[]
 for stem in ['LlavaMore_own_original_FP16_discriminative_letter_true3_v1_real_FLOCK_20261122','LlavaMore_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122','HancomHiLlava31_own_original_Transformers_FP16_2048_true3_v1_real_FLOCK_20261122','HancomHiLlava31_own_completed3_full_capacity_approved_NEWfresh3_fixed200_serial_v1_real_FLOCK_20261122']:
  p=B/stem/'job.json'
  if not p.is_file():continue
  job=js(p);sources=job['spec']['source_sha256'];assert all(sha(R/n)==h for n,h in sources.items());paths.extend(sources);jobs.append(dict(job=str(p),SHA256=sha(p),source_SHA256=sources))
  cfg=Path(job['spec']['config'])
  for variant in [cfg,cfg.with_stem(cfg.stem+'-smoke')]:
   if (R/variant).is_file():paths.append(str(variant))
 for n in sorted(set(paths)):
  p=R/n;assert p.is_file() and p.resolve().is_relative_to(R) and (n=='run.py' or n.startswith(('scripts/','configs/','universal_eval/','tests/')));h=sha(p)
  if str(p) in rows:assert rows[str(p)]['SHA256']==h,n
  else:
   dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),role='Own Hancom native original FP16 graph/full independent numeric verifier/real serial config consumer, frontend and CPU provenance, exact current model/evidence maintenance or Windows builder')
 reader=R/'scripts/maintenance/read_More_and_Hancom_own_original_native_serial_actual_v4_20261122_remote.py';r=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert r.returncode==0,r.stderr;actual=json.loads(r.stdout);base=actual['frozen_monitor_and_actual_processes'];assert base['all_current_canonical_fresh_SHA256']==ptr['canonical_SHA256'];assert base['current_Xtuner_same_serial_actual']['npu_smi']['returncode']==0
 mon=next(v for k,v in base['complete_monitor_latest_checkpoint'].items() if k.endswith('/latest.json'));assert not mon['source_errors']
 owners=base['actual_original_lock_owner_proc'];exact_jobs={j['job'][:-len('/job.json')] for j in jobs};closed=[];unknown=[]
 for warning in mon.get('inspection_errors',[]):
  matching=[o for o in owners if o.get('owner',{}).get('PID') in {w['PID'] for w in warning.get('owners',[])}]
  ok=warning.get('reason')=='runner_lock_owner_not_in_project_process_snapshot' and len(matching)==len(warning.get('owners',[])) and all(o.get('cwd')=='/root' and len(o.get('argv',[]))==4 and Path(o['argv'][2]).name=='A100_alignment_evidence_runner_20261103.py' and o['argv'][3] in exact_jobs for o in matching)
  (closed if ok else unknown).append(warning)
 model=[p for p in base['current_Xtuner_same_serial_actual']['actual_processes'] if 'run.py' in p['argv']];nf=[p for p in base['current_Xtuner_same_serial_actual']['actual_processes'] if any(Path(a).name.startswith('verify_') for a in p['argv'])]
 actual['current_specific_root_cwd_owner_warnings_closed_by_exact_job_proc']=closed;actual['current_unknown_inspection_errors']=unknown;actual['all_active_model_source_job_config_deadline_signal_untouched']=True;actual['all185_canonical_unchanged']=True
 snapshot=note/'Xtuner_completed3_More_current_Hancom_source_ready_full_actual.json';put(snapshot,actual)
 more_progress=[]
 for cap in actual['More_captures']:
  value=cap['files'].get('probability_progress.jsonl',{}).get('complete_value')
  progress=[value] if isinstance(value,dict) else [json.loads(z) for z in (value or '').splitlines() if z.strip()]
  more_progress.append(dict(home=cap['home'],committed_probability_forwards=progress,NPZ=cap['NPZ']))
 states={k:v['complete_value'] if v else None for k,v in actual['More_current_attempt_complete'].items() if k.endswith(('state.json','completion.json','actual_failure.json'))}
 hancom={k:v['complete_value'] for k,v in actual['Hancom_own_original_native_serial_actual'].items() if k.endswith(('state.json','readiness.json','actual_completion.json','actual_failure.json'))}
 compact=dict(UTC=actual['UTC'],model_processes=model,full_numeric_verifiers=nf,actual_original_kernel_lock_owners=owners,More_current_committed_probability_progress=more_progress,More_natural_states=states,Hancom_source_ready_and_queue_actual=hancom,actual_free_bytes=base['current_Xtuner_same_serial_actual']['free_bytes'],unknown_inspection_errors=unknown,new_log_error_candidates=mon.get('new_log_error_candidates',[]),specific_root_cwd_warning_closed=closed,full_actual_snapshot=str(snapshot),full_actual_snapshot_SHA256=sha(snapshot))
 full=q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False));fp=note/q['full_note_name'];fp.write_text(full,encoding='utf8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=datetime.now(timezone.utc).isoformat(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-22',Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',no_second_same_day_commit=True,next_daily_review='Day23 all useful new own native/fullNF/actual consumers/source provenance/readers/Windows builders and tests; fresh SHA/full diff/all outgoing/checks/reviewed publisher plan then one normal push'))
 facts=note/'facts.json';put(facts,dict(UTC=actual['UTC'],canonical_counts=dict(delivered=185,numeric_within=170,numeric_outside=15,complete_weights_pending=205,confirmed_missing=310,original_full_qualified=71),new_complete200_G_or_qualification_this_round=0,canonical_SHA256=ptr['canonical_SHA256'],parent_complete_handoff_SHA256='5e1dacec7b6921278fcefb761e38df7f15416d54bdcf18bb65dc2aaa00890e01',current_source_count=len(rows),primary_pending=str(pending),primary_pending_SHA256=sha(pending),current_full_actual_snapshot=str(snapshot),current_full_actual_snapshot_SHA256=sha(snapshot),new_runtime_jobs=jobs,all_old_proofs_reused_no_old_CPU200_or_big_weight_rescans=True,no_deletion_this_round=True,all_original_reports_sources_configs_logs_and_necessary_full_values_preserved=True))
 updates=dict(UTC=datetime.now(timezone.utc).isoformat(),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_primary_source_count=len(rows),current_source_count=len(rows),latest_full_actual_summary=str(fp),latest_full_actual_summary_SHA256=sha(fp),latest_actual_note=str(fp),latest_actual_note_SHA256=sha(fp),latest_actual_snapshot=str(snapshot),latest_actual_snapshot_SHA256=sha(snapshot),latest_round_facts=str(facts),latest_round_facts_SHA256=sha(facts),current_actual_stage='Xtuner_genuine3_fullNF_passed_200_capacity_refused_More_current_fullNF_Hancom_own_source_ready_serial',current_formal_model_stage='More genuine3/3 committed original probability forwards completed; full independent verifier actual in snapshot; own next Hancom awaits natural previous controller/global EX release',current_model_process_active=bool(model),current_full_numerical_verifier_active=bool(nf),current_formal_model_task_active=bool(model or nf),current_model_NN_or_NF_active=bool(nf),current_actual_model_NN_or_NF_active=bool(nf),current_NN_or_NF_active=bool(nf),current_actual_model_process_or_NF_active=bool(model or nf),current_next_original_model='HancomInSpaceAI/HiLLaVA-llama_3_1-8b',current_next_original_model_source_ready=True,current_unknown_inspection_errors=unknown,current_stop_earliest_proven_Beijing=None if model or nf else None)
 assert pointer.read_bytes()==before and closing.read_bytes()==cb and all(sha(n)==h for n,h in ptr['canonical_SHA256'].items()) and all(sha(n)==z['SHA256'] for n,z in rows.items())
 co=json.loads(cb);co.update(updates);put(closing,co);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(closing);put(pointer,ptr)
 result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),pointer_SHA256=sha(pointer),closing_SHA256=sha(closing),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fp),full_actual_note_SHA256=sha(fp),facts=str(facts),facts_SHA256=sha(facts),actual_snapshot=str(snapshot),actual_snapshot_SHA256=sha(snapshot),More_committed_forwards=sum(len(p['committed_probability_forwards']) for p in more_progress),current_model_processes=model,current_full_numeric_verifiers=nf,Hancom_source_ready=True,unknown_inspection_errors=unknown,new_log_error_candidates=mon.get('new_log_error_candidates',[]),no_active_source_job_config_deadline_signal_mutation=True,no_canonical_mutation=True,no_deletion=True,no_second_Day22_commit=True);put(note/'round_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
