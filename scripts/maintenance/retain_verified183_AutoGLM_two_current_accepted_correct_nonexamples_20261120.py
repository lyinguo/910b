"""Exact approved correct-only NPZ retention, with all wrong/groups/records kept."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,fcntl,hashlib,json,os,re,shutil,stat,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';S=R/'workspace/diagnostics/model_runs';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'verified183_current_accepted_correct_nonexample_metadata_reaudit_continue_GLM41Thinking_20261120';CFG=R/'configs/retention/verified183_AutoGLM_two_current_accepted_correct_nonexamples_20261120.json';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
MIDS={'zai-org/AutoGLM-Phone-9B','zai-org/AutoGLM-Phone-9B-Multilingual'}
from maintenance import retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119 as base
from maintenance.audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119 import preserve_and_remove_indices,full_head
sha,js,put=base.sha,base.js,base.put
def now():return datetime.now(timezone.utc).isoformat()
def quiet(owned=False):
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import core_unchanged
 core_unchanged();others=[v for v in project_processes() if v['PID']!=os.getpid()];assert not others,others
 owners=runner_lock_owners();assert (bool(owners) and all(v['PID']==os.getpid() for v in owners)) if owned else not owners,owners
def row_metadata(row,scopes):
 mid=row['model_id'];assert mid in MIDS and mid in scopes;scope=scopes[mid];home=Path(row['home']);p=Path(row['path']);i=row['index']
 assert home==Path(scope['home']) and home.is_relative_to(S) and p.parent==home and p.name=='sample_%05d.npz'%i and row['logical_indices']==[i] and 0<=i<200
 selection=scope['selection'];assert i in selection['correct'] and i in selection['remove'] and i not in selection['keep'] and i not in selection['incorrect']
 assert re.fullmatch('[0-9a-f]{64}',row['SHA256']) and row['proof']==scope['proof'] and row['proof_SHA256']==scope['proof_SHA256']
 return True
def exact_file(row,scopes):
 row_metadata(row,scopes);p=Path(row['path']);home=Path(row['home']);st=p.lstat()
 assert p.resolve()==p and home.resolve()==home and not p.is_symlink() and not home.is_symlink() and stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_dev==R.stat().st_dev
 assert st.st_size==row['bytes'] and sha(p)==row['SHA256'];assert sha(row['proof'])==row['proof_SHA256']
def tests():
 mid='zai-org/AutoGLM-Phone-9B';home=str(S/'owned_fixture_no_file_operation');selection=dict(correct=[12],remove=[12],keep=[0,18],incorrect=[18]);scope=dict(home=home,selection=selection,proof='/fixture/proof.json',proof_SHA256='a'*64)
 row=dict(model_id=mid,home=home,path=home+'/sample_00012.npz',index=12,logical_indices=[12],SHA256='b'*64,proof=scope['proof'],proof_SHA256='a'*64)
 assert row_metadata(row,{mid:scope});negative=[dict(index=0,logical_indices=[0],path=home+'/sample_00000.npz'),dict(index=18,logical_indices=[18],path=home+'/sample_00018.npz'),dict(path=home+'/sample_00013.npz'),dict(logical_indices=[12,18]),dict(model_id='foreign/model'),dict(path=str(S/'outside/sample_00012.npz')),dict(proof='/foreign/proof.json')]
 for changes in negative:
  try:row_metadata(dict(row,**changes),{mid:scope})
  except AssertionError:continue
  raise AssertionError(('Unsafe metadata accepted',changes))
 return dict(passed=True,one_positive_and_seven_negative_path_owner_wrong_example_group_proof_checks=True,no_file_mutation=True)
def plan():
 quiet();cfg=js(CFG);assert tests()['passed'];assert sha(LED)==cfg['ledger_before_SHA256']=='06f870990721bdb4e58a409f842980b7b4578eddc23367263d041c052f8e43ce'
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json')=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7'
 assert sha(H/'discovery.json')==cfg['discovery_SHA256'] and sha(H/'foreign_FD_metadata_review.json')==cfg['foreign_review_SHA256'];disc=js(H/'discovery.json');review=js(H/'foreign_FD_metadata_review.json')
 assert review['passed'] and not review['foreign_references_preserved'] and review['selected_count']==271 and review['selected_bytes']==26607613493
 for n,h in cfg['canonical_SHA256'].items():assert sha(n)==h,n
 audit=Path(cfg['verified183_full_target_audit']);assert sha(audit)==cfg['verified183_full_target_audit_SHA256'];assert js(audit)['passed'] and js(audit)['qualified_delivered']==69
 state,reg=js(D/'state.json'),js(D/'registry.json');assert sum(e['status']=='done' for e in state['models'].values())==183
 scopes={};accepted={r['model_id']:dict(proof=r['proof'],report_dir=r['report_dir']) for r in disc['all_selected_proof_metadata']}
 for old in disc['scopes']:
  mid=old['model_id'];assert mid in MIDS;sp=reg[mid];pf=Path(old['proof']);proof=js(pf);folder=Path(old['report_dir'])
  assert sp['accepted_formal200_proof']==str(pf) and sp['accepted_formal200_proof_sha256']==old['proof_SHA256']==sha(pf)
  assert proof['passed'] and proof['actual_NPU_inference'] and proof['sample_count']==200 and not proof['historical_record_reuse'] and not proof['error_indices'] and proof['original_protocol_final_qualification_complete'] and proof['formal_alignment_within_5pct'] and full_head(proof)
  assert proof['all_own_source_actual_NPU_SHA256_verified'] and proof['all_native_KV_values_losslessly_reconstructed'] and proof['all_original_active_and_unused_source_bytes_verified']
  assert state['models'][mid]['original_protocol_final_qualification_complete'] and state['models'][mid]['report_dir']==str(folder) and sp['accepted_formal200_report']==str(folder)
  assert {p.name for p in folder.iterdir()}==set(proof['report_sha256'])=={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
  for n,h in proof['report_sha256'].items():assert sha(folder/n)==h
  report=js(folder/'eval_report.json');special=set().union(*(set(proof[k]) for k in ['error_indices','empty_indices','invalid_indices','truncated_indices']))
  selection=preserve_and_remove_indices(report['detailed_results'],report['pipeline_config']['algorithm']['batch_size'],special)
  assert selection==old['all_wrong_invalid_empty_truncated_and_whole_groups_preserved'];scopes[mid]=dict(home=old['home'],report_dir=str(folder),proof=str(pf),proof_SHA256=sha(pf),selection=selection)
 assert set(scopes)==MIDS
 selected={r['path']:dict(r,index=r['logical_indices'][0]) for r in review['selected']}
 helpers=base.old_helpers(cfg);foreign,metadata=base.sources_and_dependencies(reg,accepted,selected,helpers);assert not foreign
 for row in selected.values():exact_file(row,scopes)
 protected=set(metadata)|{Path(n) for n in cfg['canonical_SHA256']}|{CFG,H/'discovery.json',H/'foreign_FD_metadata_review.json',audit,D/'A100_ALIGNMENT_HANDOFF_CURRENT.json'}
 for e in state['models'].values():
  if e['status']=='done':protected.update(Path(e['report_dir'])/n for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'])
 for sc in scopes.values():protected.update(p for p in Path(sc['home']).rglob('*') if p.is_file() and str(p) not in selected)
 own3=B/'GLM41Thinking_own_original_VLLM_BF16_8192_true3_v1_real_FLOCK_20261120/fresh3/verification_stdout.json';assert sha(own3)=='48d8a3ad392c380c134d1185a800e25039689d4cf92dd820bb460db8a1ca9478'
 q=js(own3);assert q['passed'] and q['sample_count']==3 and q['actual_forward_count']==696;protected.add(own3)
 protected.update(Path(n) for n in q['external_evidence_sha256'] if Path(n).is_relative_to(S) and Path(n).is_file())
 protected.update(Path(q['report_dir'])/n for n in q['report_sha256'])
 pointer=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json');pending=Path(pointer['new_primary_pending']);assert sha(pending)==pointer['new_primary_pending_SHA256']
 protected.update(Path(v['path']) for v in js(pending)['all_current_primary_source_and_true_consumer_configs'])
 protected.add(Path(__file__).resolve());assert not {str(p) for p in protected}&set(selected)
 hashes={}
 for i,p in enumerate(sorted(protected)):
  hashes[str(p)]=sha(p)
  if i%200==0:put(H/'retention_progress.json',dict(UTC=now(),phase='hash_all_protected_reports_sources_configs_logs_examples',checked=i+1,total=len(protected)))
 quiet();assert all(sha(n)==h for n,h in cfg['canonical_SHA256'].items());assert not (H/'exact271_correct_nonexample_plan.json').exists();(H/'ledger_before_exact271.original').write_bytes(LED.read_bytes())
 value=dict(UTC=now(),config_SHA256=sha(CFG),canonical_SHA256=cfg['canonical_SHA256'],ledger_before_SHA256=sha(LED),scopes=scopes,selected=list(selected.values()),protected_SHA256=hashes,protected_count=len(hashes),remove_count=len(selected),remove_bytes=sum(v['bytes'] for v in selected.values()),remove_allocated_bytes=sum(v['allocated_bytes'] for v in selected.values()),free_before_bytes=shutil.disk_usage(R).free,user_authorization=cfg['user_authorization'],whole_original_groups_all_wrong_invalid_truncated_small_failed_unverified_preserved=True,no_delete_yet=True,guard_tests=tests())
 assert value['remove_count']==271 and value['remove_bytes']==26607613493;put(H/'exact271_correct_nonexample_plan.json',value)
 return dict(passed=True,plan_SHA256=sha(H/'exact271_correct_nonexample_plan.json'),remove_count=271,remove_bytes=value['remove_bytes'],protected_count=len(hashes),free_before_bytes=value['free_before_bytes'],projected_free_bytes=value['free_before_bytes']+value['remove_allocated_bytes'],no_delete_yet=True)
def apply(reviewed):
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 quiet();cfg=js(CFG);path=H/'exact271_correct_nonexample_plan.json';assert sha(path)==reviewed;p=js(path);assert sha(CFG)==p['config_SHA256'] and sha(LED)==p['ledger_before_SHA256'];assert not (H/'retention_actual_completion.json').exists()
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');quiet(True)
  for i,(n,h) in enumerate(p['protected_SHA256'].items()):
   assert sha(n)==h,('Protected changed before delete',n)
   if i%200==0:put(H/'retention_progress.json',dict(UTC=now(),phase='fresh_complete_protected_SHA_before_delete',checked=i+1,total=p['protected_count']))
  for row in p['selected']:exact_file(row,p['scopes'])
  helpers=base.old_helpers(cfg);reg=js(D/'registry.json');disc=js(H/'discovery.json');accepted={r['model_id']:dict(proof=r['proof'],report_dir=r['report_dir']) for r in disc['all_selected_proof_metadata']};foreign,_=base.sources_and_dependencies(reg,accepted,{r['path']:r for r in p['selected']},helpers);assert not foreign
  quiet(True);assert all(sha(n)==h for n,h in p['canonical_SHA256'].items());manifest=H/'exact_removed_correct_nonexample_NPZ_SHA256.jsonl';assert not manifest.exists();removed=[];error=None;free=shutil.disk_usage(R).free
  try:
   with manifest.open('x',encoding='utf8') as out:
    for row in p['selected']:
     exact_file(row,p['scopes']);Path(row['path']).unlink();removed.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
  except BaseException:error=traceback.format_exc()
  changed=[n for n,h in p['protected_SHA256'].items() if sha(n)!=h];ok=not error and not changed and len(removed)==p['remove_count']
  result=dict(UTC=now(),actual_returncode=0 if ok else 1,plan_SHA256=reviewed,removed_files=len(removed),removed_bytes=sum(r['bytes'] for r in removed),removed_allocated_bytes=sum(r['allocated_bytes'] for r in removed),manifest=str(manifest),manifest_SHA256=sha(manifest),protected_files=p['protected_count'],all_protected_SHA256_unchanged=not changed,protected_errors=changed,error=error,free_before_bytes=free,free_after_bytes=shutil.disk_usage(R).free,all_five_reports_sources_configs_logs_and_necessary_full_groups_all_wrong_invalid_truncated_small_failed_unverified_values_preserved=True,canonical_scores_both_books_state_registry_binding_unchanged=True,exact_pruned_arrays_now_unavailable_original_proof_bytes_unchanged=True)
  put(H/'retention_actual_completion.json',result);assert sha(LED)==p['ledger_before_SHA256'];ledger=js(LED);ledger['events'].append(dict(UTC=result['UTC'],authorization=cfg['user_authorization'],receipt=str(H/'retention_actual_completion.json'),receipt_SHA256=sha(H/'retention_actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=len(removed),removed_bytes=result['removed_bytes'],all_original_records_kept=True,all_incorrect_and_required_groups_preserved=True,intentional_nonexample_matrix_unavailability=True,actual_returncode=result['actual_returncode']));put(LED,ledger)
  assert all(sha(n)==h for n,h in p['canonical_SHA256'].items() if n!=str(LED));assert ok
  return dict(actual=result,actual_SHA256=sha(H/'retention_actual_completion.json'),ledger_SHA256=sha(LED))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','apply','test']);ap.add_argument('plan_SHA256',nargs='?');a=ap.parse_args();print(json.dumps(tests() if a.mode=='test' else plan() if a.mode=='plan' else apply(a.plan_SHA256),ensure_ascii=False))
