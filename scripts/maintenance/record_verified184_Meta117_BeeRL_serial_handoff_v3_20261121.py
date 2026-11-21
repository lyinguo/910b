"""Incrementally preserve real serial NN/NF/publication phases without altering any live job."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def stage(p):
 p=Path(p);return {n:dict(SHA256=sha(p/n),value=js(p/n)) for n in ['state.json','actual_completion.json','formal_acceptance.json','dual_publication_completed.json','full_target_dual_publication_audit.json','publication_receipt.json','capacity_review.json','completed_storage_proof_reuse.json'] if (p/n).is_file()}
def main():
 q=js(sys.argv[1]);assert sha(sys.argv[1])==sys.argv[2];pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';raw=pointer.read_bytes();ptr=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==q['expected_pointer_SHA256']
 H=Path(ptr['handoff_directory']);parent=Path(ptr['new_primary_pending']);assert sha(parent)==q['expected_pending_SHA256'];rows={x['path']:x for x in js(parent)['all_current_primary_source_and_true_consumer_configs']}
 note=H/'late_actual_notes'/q['round_name'];assert not note.exists();note.mkdir();(note/'previous_pointer.json').write_bytes(raw);(note/'previous_closing_addendum.json').write_bytes((H/'closing_addendum.json').read_bytes())
 primary=list(q['new_primary_paths'])+[str(Path(__file__).resolve().relative_to(R))]
 bj=B/'BeeSFT_completed_true3_full_LZMA_approved_NEWfresh3_fixed200_after_Ovis_dual_serial_v2_POSIX_real_FLOCK_20261121/job.json'
 if bj.is_file():primary+=list(js(bj)['spec']['source_sha256'])
 for n in sorted(set(primary)):
  p=R/n;assert p.is_relative_to(R) and n.startswith(('scripts/','configs/','universal_eval/'));h=sha(p)
  if str(p) in rows:assert rows[str(p)]['SHA256']==h
  else:
   dest=note/'primary_source_snapshots'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());rows[str(p)]=dict(path=str(p),relative=n,SHA256=h,bytes=p.stat().st_size,snapshot=str(dest),role='actual serial acceptance/publication/next-resource continuation source or true consumer config')
 for x in q['local_original_records']:
  b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'];p=note/'local_original_records'/x['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 sys.path[:0]=[str(R),str(R/'scripts')]
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 canon={p:sha(p) for p in ptr['canonical_SHA256']};pub=B/'OvisU1_original_BF16_EOSNone512_fullNF2_verified184_blank_G_publication_20261121'
 changed={p:(h,canon[p]) for p,h in ptr['canonical_SHA256'].items() if canon[p]!=h}
 if changed:
  audit=js(pub/'full_target_dual_publication_audit.json');receipt=js(pub/'publication_receipt.json');assert audit['passed'] and js(pub/'dual_publication_completed.json')['completed']
  for p,(old,new) in changed.items():
   if p==str(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'):
    ret=B/'verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121'
    done=js(ret/'actual_completion.json')
    assert old=='6f4dcf0be1dad1b0e4a1e8ec53725d76d6abff0965f5fdf87a9428f772b7179c' and new=='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd'
    assert sha(ret/'actual_completion.json')=='c2ed3e30c9f71b4695ea3dfa35630834f4b9752fc58e13f84c1d64a900c63d3c' and sha(ret/'ledger_before.original')==old
    assert done['actual_returncode']==0 and done['removed_files']==117 and done['all_protected_SHA256_unchanged'] and not done['protected_errors']
    assert sha(done['manifest'])==done['manifest_SHA256']=='363e58f7b5e5463c9761df7b11c6780aa9ff0774f89a0f650e440c53c4facc59'
   else:assert receipt['canonical_before_sha256'].get(p)==old and audit['canonical_SHA256'].get(p)==new
 state=js(D/'state.json');counts={k:sum(v['status']==k for v in state['models'].values()) for k in ['done','needs_adaptation','skipped_no_complete_weights']};assert counts in [dict(done=183,needs_adaptation=207,skipped_no_complete_weights=310),dict(done=184,needs_adaptation=206,skipped_no_complete_weights=310)]
 stages={
 'BeeRL_storage_fullNF_capacity':stage(B/'BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121'),
 'Meta_fullNF4':stage(B/'MetaMllama_completed_genuine200_formal_NF4_original_empty_invalid_schema_v2_actual_file_SHA_after_Ovis_serial_no_NN_replay_20261121'),
 'Ovis_fullNF2':stage(B/'OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121'),
 'Ovis_conditional_publication':stage(B/'OvisU1_fullNF2_qualified184_reviewed_dual_controller_20261121'),
 'Ovis_dual_publication':stage(pub),
 'Bee_capacity_v3':stage(B/'BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121'),
 'Bee_capacity_wait_v4':stage(B/'BeeSFT_completed_storage_NF_capacity_v4_actual_original_disk_waiter_serial_20261121'),
 'Meta117_retention':stage(B/'verified184_Meta_fullNF4_correct_nonexample117_v3_actual_groups_Bee_capacity_recovery_20261121'),
 'Bee_serial_v2':stage(B/'BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121')}
 procs=project_processes();owners=runner_lock_owners();actual=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
   if any(x in ' '.join(argv).lower() for x in ['verify_ovisu1_completed200','continue_ovisu1','continue_beesft','continue_beerl','bee8brl_own','bee-8b-rl','bee8bsft_own','bee-8b-sft','a100_alignment_evidence_runner_20261103.py','publish_ovisu1','verify_bee8bsft']):
    stat=(p/'stat').read_text();actual.append(dict(PID=int(p.name),argv=argv,cwd=os.readlink(p/'cwd'),stat=stat,io=(p/'io').read_text()))
  except (OSError,UnicodeError):pass
 bee_home=B/'BeeSFT_completed_true3_full_LZMA_NF_capacity_after_Ovis_verified184_dual_serial_v2_POSIX_20261121'
 bee_nf=(bee_home/'storage_fullNF_stderr.log').exists() and not (bee_home/'complete_storage_fullNF_proof.json').exists() and not (bee_home/'actual_completion.json').exists() and any(x['PID']==stages['Bee_serial_v2'].get('state.json',{}).get('value',{}).get('PID') for x in owners)
 rl_home=B/'BeeRL_completed_true3_full_LZMA_NF_capacity_serial_v1_20261121'
 rl_nf=(rl_home/'storage_fullNF_stderr.log').exists() and not (rl_home/'complete_storage_fullNF_proof.json').exists() and not (rl_home/'actual_completion.json').exists() and any(x['PID']==stages['BeeRL_storage_fullNF_capacity'].get('state.json',{}).get('value',{}).get('PID') for x in owners)
 active=bee_nf or rl_nf or any(any('verify_ovisu1_completed200' in a or ('verify_bee8bsft_' in a or 'verify_bee8brl_' in a) or (a.endswith('run.py')) for a in x['argv']) for x in actual)
 snap=dict(UTC=now(),canonical_counts=counts,canonical_SHA256=canon,actual_project_processes=procs,actual_bound_processes=actual,original_FLOCK=owners,stages=stages,current_formal_NN_or_fullNF_active=active,free_bytes=shutil_disk_free(),no_signal_hot_edit_NN_replay_or_deletion_in_this_recorder=True)
 put(note/'current_serial_fullNF_and_publication_actual.json',snap)
 full=q['full_handoff_template'].replace('CURRENT_ACTUAL_JSON',json.dumps(snap,ensure_ascii=False)).replace('CURRENT_SOURCE_COUNT',str(len(rows)))
 fp=note/q['full_note_name'];fp.write_text(full,encoding='utf8')
 pending=C/q['pending_name'];assert not pending.exists();put(pending,dict(UTC=now(),parent_pending=str(parent),parent_pending_SHA256=sha(parent),source_count=len(rows),all_current_primary_source_and_true_consumer_configs=[rows[k] for k in sorted(rows)],Code_Day_already_pushed='2026-11-21',Code_Day_commit='189b3146a82adf0dc9e50f12691007ff31de6b6b',no_second_same_day_commit=True,next_daily_review='Day22 all useful new code/config consumers/dependencies/freshSHA/full diff/outgoing/appropriate tests/actual publisher reviewed planSHA'))
 updates=dict(UTC=now(),canonical_SHA256=canon,new_primary_pending=str(pending),new_primary_pending_SHA256=sha(pending),latest_pending=str(pending),latest_pending_SHA256=sha(pending),current_source_count=len(rows),current_primary_source_count=len(rows),latest_actual_note=str(fp),latest_actual_note_SHA256=sha(fp),latest_full_actual_summary=str(fp),latest_full_actual_summary_SHA256=sha(fp),latest_actual_snapshot=str(note/'current_serial_fullNF_and_publication_actual.json'),latest_actual_snapshot_SHA256=sha(note/'current_serial_fullNF_and_publication_actual.json'),current_actual_stage='verified184_Ovis_delivered_Meta117_retention_complete_BeeSFT_capacity_wait_BeeRL_storage_prepare_or_fullNF_see_actual',current_formal_model_stage='BeeSFT full storage NF passed but capacity wait; BeeRL complete lossless copies and own fullNF/capacity in same serial controller; exact live NF flag in snapshot',current_actual_model_NN_or_NF_active=active,current_model_NN_or_NF_active=active,current_NN_or_NF_active=active,current_actual_model_process_or_NF_active=active,current_Ovis_progress=dict(genuine_NPU_samples=200,genuine_NPU_forwards=102400,original_model_returncode=0,original_verifier_returncode=1,new_formal_acceptance_home=str(B/'OvisU1_completed_genuine200_formal_NF2_original_raw_fallback_metadata_after_Meta_serial_no_NN_replay_20261121')),canonical_counts=counts,current_verified_count=counts['done'],numeric_within5pct=169,numeric_outside5pct=15,complete_weights_pending_formal=206,confirmed_missing=310,complete_original_protocol_qualified=70,current_stop_earliest_proven_Beijing=None if active else '2026-11-21T20:35:24.837093+08:00',current_Bee_wait_home=str(B/'BeeSFT_completed_storage_NF_capacity_v4_actual_original_disk_waiter_serial_20261121'),current_retention_ledger_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'))
 assert pointer.read_bytes()==raw and all(sha(p)==h for p,h in canon.items());close=js(H/'closing_addendum.json');close.update(updates);put(H/'closing_addendum.json',close);ptr.update(updates);ptr['closing_addendum_SHA256']=sha(H/'closing_addendum.json');put(pointer,ptr)
 result=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),pending=str(pending),pending_SHA256=sha(pending),source_count=len(rows),full_actual_note=str(fp),full_actual_note_SHA256=sha(fp),actual_snapshot=str(note/'current_serial_fullNF_and_publication_actual.json'),canonical_counts=counts,canonical_SHA256=canon,no_original_job_or_live_source_changed=True);put(note/'round_preservation_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
def shutil_disk_free():
 import shutil
 return shutil.disk_usage(R).free
if __name__=='__main__':main()
