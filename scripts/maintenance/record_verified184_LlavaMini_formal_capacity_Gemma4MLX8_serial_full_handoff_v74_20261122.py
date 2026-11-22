"""Incremental exact source/receipt handoff; no model, workbook, policy or runtime change."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,x):
 p=Path(p);temp=p.with_name(p.name+'.tmp.'+str(os.getpid()));temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,p)
def main():
 plan=Path(sys.argv[1]);assert sha(plan)==sys.argv[2];q=js(plan);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';before=pointer.read_bytes();assert hashlib.sha256(before).hexdigest()==q['expected_pointer_SHA256'];ptr=json.loads(before)
 H=Path(ptr['handoff_directory']);parent=Path(ptr['new_primary_pending']);assert sha(parent)==ptr['new_primary_pending_SHA256']==q['expected_pending_SHA256'];canonical=dict(ptr['canonical_SHA256']);assert all(sha(p)==h for p,h in canonical.items())
 assert sha(R/'configs/publication/code_only.json')=='39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5'
 assert sha(C/'daily_20261122_actual_reviewed_commit_remote_receipts/daily_20261122_receipt.json')=='32ff728e32482f9009e8ef1fe9e22f5e836613a48edd8181fef01fa847ac1115'
 rows={r['path']:r for r in js(parent)['all_current_primary_source_and_true_consumer_configs']};assert all(sha(p)==r['SHA256'] for p,r in rows.items())
 note=H/'late_actual_notes'/q['round_name'];note.mkdir(exist_ok=False);(note/'previous_pointer.json').write_bytes(before);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 for row in q['local_original_records']:
  raw=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==row['SHA256'];path=note/'local_original_records'/row['name'];assert path.resolve().is_relative_to(note.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
 primary=list(q['new_primary_paths'])+[str(Path(__file__).resolve().relative_to(R))];job_paths=q['current_model_job_paths']
 for rel in job_paths:
  p=B/rel/'job.json'
  if p.is_file():primary+=list(js(p)['spec']['source_sha256'])
 for folder,pattern in [('Gemma4ForConditionalGeneration','unsloth--gemma-4-E4B-it-MLX-8bit-*-20261122*.yaml'),('protocols','gemma4_e4b_MLX8_*_20261122.json')]:
  primary += [str(p.relative_to(R)) for p in (R/'configs'/folder).glob(pattern)]
 for name in sorted(set(primary)):
  p=R/name;assert p.is_file() and p.resolve().is_relative_to(R) and (name=='run.py' or name.startswith(('scripts/','configs/','universal_eval/','tests/')));digest=sha(p)
  if str(p) in rows:assert rows[str(p)]['SHA256']==digest,name
  else:
   dest=note/'primary_source_snapshots'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=name,SHA256=digest,bytes=p.stat().st_size,snapshot=str(dest),role='Own original protocol adaptation, complete independent evidence, true config consumer, serial recovery/status/Windows control or preservation source')
 reader=R/'scripts/maintenance/read_Gemma4MLX8_formal_NF2_serial_recovery_v4_committed_trace_20261122_remote.py';result=subprocess.run(['python3','-B',str(reader)],cwd=R,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert result.returncode==0,result.stderr.decode(errors='replace');actual=json.loads(result.stdout)
 state=js(D/'state.json');counts={n:sum(r['status']==n for r in state['models'].values()) for n in ['done','needs_adaptation','skipped_no_complete_weights']};assert counts==dict(done=184,needs_adaptation=206,skipped_no_complete_weights=310)
 sources_to_check=[]
 for rel in job_paths:
  p=B/rel/'job.json'
  if p.is_file():sources_to_check.append((str(p),js(p)['spec']['source_sha256']))
 assert all(sha(R/n)==h for job,values in sources_to_check for n,h in values.items())
 model_processes=[p for p in actual['processes'] if any(Path(a).name=='run.py' for a in p['argv'])];verifiers=[p for p in actual['processes'] if any(Path(a).name.startswith('verify_gemma4_e4b_MLX8_') for a in p['argv'])]
 verified_caps=[]
 for name in ['Gemma4E4BMLX8_completed_genuine3_formal_NF2_actual_profiler_schema_then_capacity_serial_v2_20261122/own_completed3_all_ZIP_all200_shape_capacity.json','LlavaMini_completed_formal_single_JSON_NF_acceptance_only_capacity_resume_v4_20261122/own_passed3_complete_all200_shape_capacity.json']:
  p=B/name
  if p.is_file():
   cap=js(p);verified_caps.append(dict(path=str(p),SHA256=sha(p),empirical_estimate_bytes=cap['empirical_estimate_bytes'],actual_free_bytes=cap['actual_free_bytes'],capacity_passed=cap['capacity_passed'],complete_ZIP_parts=len(cap.get('all_complete_ZIP_parts',[]))))
 model_forward_proven=any(c.get('records') and any(c['path'].endswith('_'+str(p['PID'])) for p in model_processes) for c in actual['captures'])
 actual.update(canonical_SHA256=canonical,canonical_counts=counts,model_process_active=bool(model_processes),current_model_has_committed_actual_NPU_forwards=model_forward_proven,formal_full_numerical_verifier_active=bool(verifiers),model_process_does_not_imply_continuous_NPU_forward_submission=True,capacity_summaries=verified_caps,Code_Day22_unique_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',current_source_and_real_config_count=len(rows),all_current_frozen_job_sources_unchanged=True)
 put(note/'current_serial_real_process_acceptance_capacity_actual.json',actual)
 compact=dict(UTC=actual['UTC'],full_actual_snapshot=str(note/'current_serial_real_process_acceptance_capacity_actual.json'),full_actual_snapshot_SHA256=sha(note/'current_serial_real_process_acceptance_capacity_actual.json'),canonical_counts=counts,capacity_summaries=verified_caps,runner_lock_owners=actual['runner_lock_owners'],actual_model_processes=[dict(PID=p['PID'],PPid=p['PPid'],cwd=p['cwd'],argv=p['argv']) for p in actual['processes']],capture_progress=[dict(path=c['path'],committed_samples=len(c['records']),committed_native_forwards=sum(r['forwards'] or 0 for r in c['records']),error_indices=[r['index'] for r in c['records'] if r['error']]) for c in actual['captures']],actual_job_states=[dict(job=j['job'],state=json.loads(f['text'])) for j in actual['jobs'] for f in j['files'] if Path(f['path']).name=='state.json' and 'text' in f])
 full=q['full_handoff_template'].replace('CURRENT_SOURCE_COUNT',str(len(rows))).replace('CURRENT_ACTUAL_JSON',json.dumps(compact,ensure_ascii=False));fullpath=note/q['full_note_name'];fullpath.write_text(full,encoding='utf-8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=now(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-22',Code_Day_commit='43a149bb69510288451c2348159d6e7baa0d8d8e',no_second_same_day_commit=True,next_daily_review='Day23 all new useful source/config/test/Windows consumer/dependencies, fresh SHA/full diff/all outgoing/appropriate validation and actual publisher reviewed planSHA; no empty commit'))
 active=bool(model_forward_proven or verifiers)
 updates=dict(UTC=now(),new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_source_count=len(rows),current_primary_source_count=len(rows),latest_actual_note=str(fullpath),latest_actual_note_SHA256=sha(fullpath),latest_full_actual_summary=str(fullpath),latest_full_actual_summary_SHA256=sha(fullpath),latest_actual_snapshot=str(note/'current_serial_real_process_acceptance_capacity_actual.json'),latest_actual_snapshot_SHA256=sha(note/'current_serial_real_process_acceptance_capacity_actual.json'),current_actual_stage='verified184_LlavaMini_same3_formal_acceptance_capacity_refused_Gemma4MLX8_own_BF16_serial_actual_priority',current_formal_model_stage='Gemma4MLX8 genuine3/24 formally NF2 passed; same approved NEWfresh3/fixed200 actual source/job/trace/exit wins',current_verified_count=184,numeric_within5pct=169,numeric_outside5pct=15,complete_weights_pending_formal=206,confirmed_missing=310,complete_original_protocol_qualified=70,current_model_process_active=bool(model_processes),current_full_numerical_verifier_active=bool(verifiers),current_model_NN_or_NF_active=active,current_actual_model_NN_or_NF_active=active,current_NN_or_NF_active=active,current_actual_model_process_or_NF_active=active,model_process_active_does_not_claim_continuous_NPU_forwards=True,current_stop_earliest_proven_Beijing=None if active else 'Unknown: use latest actual verifier/controller exit, no accumulation across recovery',last_formal_recovery_Beijing='2026-11-22T02:40:05+08:00',current_retention_ledger_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'))
 assert pointer.read_bytes()==before and all(sha(p)==h for p,h in canonical.items()) and all(sha(p)==r['SHA256'] for p,r in rows.items())
 closing=js(H/'closing_addendum.json');closing.update(updates);put(H/'closing_addendum.json',closing);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 receipt=dict(passed=True,UTC=now(),pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fullpath),full_actual_note_SHA256=sha(fullpath),actual_snapshot=str(note/'current_serial_real_process_acceptance_capacity_actual.json'),current_model_process_active=bool(model_processes),current_full_numerical_verifier_active=bool(verifiers),all_model_job_sources_canonical_workbooks_state_reg_policies_retention_unchanged=True,no_new_deletion_or_second_Day22_commit=True)
 put(note/'round_preservation_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
