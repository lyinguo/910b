"""Exact user-authorized pruning after full numerical measurement acceptance.

Baseline-gap qualification stays false. All wrong cases and original full
first/middle/last outer groups remain; only listed correct nonexample NPZ go.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,fcntl,shutil,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';S=R/'workspace/diagnostics/model_runs'
sys.path[:0]=[str(R),str(R/'scripts')]
from maintenance import retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119 as base
from maintenance import audit_Huihui_completed200_baselinegap_all_targets_before_selective_retention_20261119 as audit
H=audit.H;CFG=R/'configs/retention/Huihui_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.json';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
sha=base.sha;js=base.js;put=base.put;now=base.now
def guard(owned=False):
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import core_unchanged
 core_unchanged();assert not [p for p in project_processes() if p['PID']!=os.getpid()]
 owners=runner_lock_owners();assert not owners if not owned else owners and all(int(x['PID'])==os.getpid() for x in owners),owners
def plan(reviewed_audit_SHA256):
 guard();cfg=js(CFG);assert sha(H/'measurement_target_audit.json')==reviewed_audit_SHA256==cfg['measurement_target_audit_SHA256'];a=js(H/'measurement_target_audit.json');helper=base.old_helpers(cfg)
 assert a['passed'] and not a['original_protocol_final_qualification_complete'] and not a['G_publication_permitted'] and a['no_G_state_registry_binding_or_workbook_write'] and a['sample_count']==200 and a['actual_forward_count']==4325
 assert len(a['selection']['incorrect'])==61 and set(a['selection']['incorrect'])<=set(a['selection']['keep']) and not set(a['selection']['remove'])&set(a['selection']['keep'])
 pf=js(a['formal_proof']);audit.measurement_contract(pf);assert sha(a['formal_proof'])==a['formal_proof_SHA256'] and all(sha(n)==h for n,h in a['canonical_SHA256'].items())
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json')=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7' and sha(LED)=='5e875cc0d8a6b70416ab254db8c542c01f641b3ebee9199009a722478f72bc32'
 assert not (H/'retention_plan.json').exists()
 state=js(D/'state.json');reg=js(D/'registry.json');assert state['models'][audit.MID]['status']=='needs_adaptation' and audit.MID not in reg
 candidates={x['path']:x for x in a['all200_archive_original_metadata'] if x['index'] in a['selection']['remove']}
 old=js(B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119/discovery.json');accepted={x['model_id']:x for x in old['all_selected_proof_metadata']};accepted[audit.MID]=dict(model_id=audit.MID,proof=a['formal_proof'],report_dir=a['report_dir'])
 foreign,metadata=base.sources_and_dependencies(reg,accepted,candidates,helper)
 for n in foreign:candidates.pop(n,None)
 assert candidates and len(candidates)<=105 and sum(x['bytes'] for x in candidates.values())<=cfg['max_remove_bytes']
 protected=set(metadata)|{Path(n) for n in a['canonical_SHA256']}|{CFG,H/'measurement_target_audit.json',H/'local_current_snapshot.json'}
 for x in state['models'].values():
  if x['status']=='done':protected.update(Path(x['report_dir'])/n for n in helper.FIVE)
 for home in [Path(a['home']),Path(a['report_dir'])]:protected.update(p for p in home.rglob('*') if p.is_file() and str(p) not in candidates)
 for proof in [a['original_true3_proof'],a['fresh3_proof']]:
  protected.add(Path(proof));p=js(proof)
  protected.update(Path(p['report_dir'])/n for n in helper.FIVE)
  for n in p['external_evidence_sha256']:
   f=Path(n)
   if f.is_relative_to(S) and f.is_file():protected.add(f)
 for root in [R/'scripts',R/'configs',R/'universal_eval']:
  protected.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.ir'])
 assert not set(candidates)&{str(p) for p in protected}
 hashes={}
 for i,p in enumerate(sorted(protected)):
  hashes[str(p)]=sha(p)
  if i%200==0:put(H/'retention_progress.json',dict(UTC=now(),phase='hashing_all_protected_reports_sources_logs_wrong_controls_examples',checked=i+1,total=len(protected)))
 for row in candidates.values():
  p=Path(row['path']);assert helper.removable_file(p,Path(a['home'])) and sha(p)==row['SHA256'] and row['index'] in a['selection']['correct'] and row['index'] not in a['selection']['keep']
 (H/'ledger_before_selective_retention.original').write_bytes(LED.read_bytes());guard()
 p=dict(UTC=now(),config_SHA256=sha(CFG),audit_SHA256=reviewed_audit_SHA256,ledger_before_SHA256=sha(LED),canonical_before_SHA256=a['canonical_SHA256'],remove=list(candidates.values()),remove_count=len(candidates),remove_bytes=sum(x['bytes'] for x in candidates.values()),remove_allocated_bytes=sum(x['allocated_bytes'] for x in candidates.values()),foreign_or_open_references_preserved=foreign,protected_SHA256=hashes,protected_count=len(hashes),keep_indices=a['selection']['keep'],incorrect_indices=a['selection']['incorrect'],all61_wrong_controls_and_full_original_examples_preserved=True,all_small_and_failure_values_preserved=True,qualification_stays_false_and_all_canonical_scores_unchanged=True,policy_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json'),free_before_bytes=shutil.disk_usage(R).free)
 put(H/'retention_plan.json',p)
 return dict(plan_SHA256=sha(H/'retention_plan.json'),remove_count=p['remove_count'],remove_bytes=p['remove_bytes'],remove_allocated_bytes=p['remove_allocated_bytes'],protected_count=p['protected_count'],preserved_wrong_count=61,preserved_numeric_cases=len(p['keep_indices']),foreign_files=len(foreign),no_deletion_yet=True)
def apply(reviewed_plan_SHA256):
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 guard();cfg=js(CFG);helper=base.old_helpers(cfg);assert sha(H/'retention_plan.json')==reviewed_plan_SHA256;p=js(H/'retention_plan.json')
 assert sha(CFG)==p['config_SHA256'] and sha(H/'measurement_target_audit.json')==p['audit_SHA256'] and sha(LED)==p['ledger_before_SHA256'] and not (H/'retention_actual_completion.json').exists()
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');guard(True)
  for i,(n,h) in enumerate(p['protected_SHA256'].items()):
   assert sha(n)==h,('Protected changed before deletion',n)
   if i%200==0:put(H/'retention_progress.json',dict(UTC=now(),phase='fresh_full_protected_SHA_before_any_deletion',checked=i+1,total=p['protected_count']))
  for row in p['remove']:
   f=Path(row['path']);home=Path(row['home']);assert home==Path(js(H/'measurement_target_audit.json')['home']) and home.is_relative_to(S) and helper.removable_file(f,home)
   assert row['index'] not in p['keep_indices'] and row['index'] not in p['incorrect_indices'] and str(f) not in p['protected_SHA256'] and f.stat().st_size==row['bytes'] and sha(f)==row['SHA256'] and sha(row['proof'])==row['proof_SHA256']
  assert all(sha(n)==h for n,h in p['canonical_before_SHA256'].items());guard(True)
  manifest=H/'removed_exact_verified_correct_nonexample_NPZ_original_SHA256.jsonl';assert not manifest.exists();removed=[];failure=None;free=shutil.disk_usage(R).free
  try:
   with manifest.open('x',encoding='utf8') as out:
    for row in p['remove']:
     f=Path(row['path']);assert helper.removable_file(f,Path(row['home'])) and sha(f)==row['SHA256'];f.unlink();removed.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
  except BaseException:failure=traceback.format_exc()
  bad=[n for n,h in p['protected_SHA256'].items() if sha(n)!=h];ok=not failure and not bad and len(removed)==p['remove_count']
  actual=dict(UTC=now(),actual_returncode=0 if ok else 1,plan_SHA256=reviewed_plan_SHA256,audit_SHA256=p['audit_SHA256'],removed_files=len(removed),removed_bytes=sum(x['bytes'] for x in removed),removed_allocated_bytes=sum(x['allocated_bytes'] for x in removed),manifest=str(manifest),manifest_SHA256=sha(manifest),protected_files=p['protected_count'],all_protected_SHA256_unchanged=not bad,protected_errors=bad,error=failure,free_before_bytes=free,free_after_bytes=shutil.disk_usage(R).free,all61_incorrect_debug_control_cases_and_full_original_examples_retained=True,all_five_all_records_sources_configs_snapshots_logs_retained=True,no_small_failed_incomplete_unverified_arrays_deleted=True,all_original_proofs_unchanged_exact_pruned_arrays_no_longer_available=True,canonical_scores_state_registry_binding_and_both_workbooks_unchanged=True,measurement_numerically_accepted_but_not_qualified_or_published=True)
  put(H/'retention_actual_completion.json',actual);assert sha(LED)==p['ledger_before_SHA256'];ledger=js(LED)
  ledger['events'].append(dict(UTC=actual['UTC'],authorization=cfg['latest_actual_user_authorization'],receipt=str(H/'retention_actual_completion.json'),receipt_SHA256=sha(H/'retention_actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=len(removed),removed_bytes=actual['removed_bytes'],all_original_records_kept=True,intentional_nonexample_matrix_unavailability=True,all_incorrect_cases_preserved=True,numerical_measurement_not_G_qualification=True,actual_returncode=actual['actual_returncode']))
  put(LED,ledger);assert all(sha(n)==h for n,h in p['canonical_before_SHA256'].items() if n!=str(LED));assert ok
  return dict(actual=actual,actual_SHA256=sha(H/'retention_actual_completion.json'),ledger_SHA256=sha(LED))
if __name__=='__main__':print(json.dumps(plan(sys.argv[2]) if sys.argv[1]=='plan' else apply(sys.argv[2]),ensure_ascii=False))
