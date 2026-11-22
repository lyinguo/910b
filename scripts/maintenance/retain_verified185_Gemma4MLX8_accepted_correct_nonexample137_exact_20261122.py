"""Exact post-fullNF/dual-audit correct nonexample retention. Preserve all originals otherwise."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,stat,fcntl,shutil,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';S=R/'workspace/diagnostics/model_runs';MID='unsloth/gemma-4-E4B-it-MLX-8bit';H=B/'verified185_Gemma4MLX8_fullNF2_dual_audit_correct_nonexample137_exact_retention_20261122';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json';PUB=B/'Gemma4E4BMLX8_original_Transformers_BF16_verified185_blank_G_publication_20261122';HOME=S/'Gemma4E4BMLX8_own_original_BF16_20261122_024718_052733_1692381';J=B/'Gemma4E4BMLX8_completed_genuine3_fullNF2_approved_NEWfresh3_fixed200_serial_v2_real_FLOCK_20261122'
sys.path[:0]=[str(R),str(R/'scripts')]
from maintenance import retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119 as base
sha,js,put=base.sha,base.js,base.put
FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
now=lambda:datetime.now(timezone.utc).isoformat()
def quiet(owned=False):
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
 core_unchanged();allowed=[]
 for p in project_processes():
  if int(p['PID'])==os.getpid():continue
  watcher=R/'scripts/maintenance/watch_batch_910b_immediate_disk_recovery_cleanup_v1_20261120.py';bad=R/'scripts/continue_BeeSFT_existing_true3_full_LZMA_NF_capacity_after_Ovis_dual_serial_20261121.py'
  if p['cwd']==str(R) and str(watcher) in p['argv'] and sha(watcher)=='32bd7209129b3cf22bac0056584847262e9793b3fab6b52dfaccc7b3de801401':allowed.append(p);continue
  if p['cwd']==str(R) and str(bad) in p['argv'] and sha(bad)=='16241ed4e429572f4d1fedc49d31bb89f79e9603c1ee357507a51739e2970678':
   q=js(R/'configs/queues/BeeSFT_completed_true3_complete_LZMA_after_Ovis_dual_serial_20261121.json');assert not Path(q['previous_dual_publication_home']).is_absolute() and not Path(q['previous_dual_publication_home']).exists();allowed.append(p);continue
  raise AssertionError(('Unknown or active project process: retain all files',p))
 owners=runner_lock_owners()
 if owned:kernel_lock_owned(os.getpid(),D/'runner.lock');assert owners and all(int(x['PID'])==os.getpid() for x in owners)
 else:assert not owners
 return allowed
def exact(row):
 p=Path(row['path']);assert p==HOME/('sample_%05d.npz'%row['index']) and p.resolve()==p and HOME.resolve()==HOME and HOME.is_relative_to(S)
 st=p.lstat();assert stat.S_ISREG(st.st_mode) and not p.is_symlink() and st.st_nlink==1 and st.st_size==row['bytes'];assert sha(p)==row['SHA256']
def plan():
 quiet();H.mkdir(exist_ok=False)
 assert sha(LED)=='b3969dd7799de7a9d9a5cfa2621681b931c734c4c6f1233734dc3f803a80fbdd' and sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json')=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7'
 audit_path=PUB/'full_target_dual_publication_audit.json';assert sha(audit_path)=='cf97094bacfa17f27c531431c7486594e0a07b57f18c02fce19e93a6751f08f7';audit=js(audit_path);assert audit['passed'] and audit['qualified_delivered']==71 and audit['numeric_within5pct']==170
 canonical=dict(audit['canonical_SHA256']);assert all(sha(n)==h for n,h in canonical.items())
 state,reg=js(D/'state.json'),js(D/'registry.json');assert state['models'][MID]['status']=='done' and state['models'][MID]['score']==.84 and reg[MID]['accepted_formal200_proof_sha256']=='3cf18242f6f6276cfe552ccbb5d2bfd54ea56ad0863785b21097e26404bbad8d'
 job=js(J/'job.json');done=js(J/'completion.json');assert done['actual_returncode']==done['model_returncode']==done['verifier_returncode']==0 and [x['sample_count'] for x in done['accepted']]==[3,200]
 proof_path=Path(done['accepted'][-1]['proof']);assert sha(proof_path)=='3cf18242f6f6276cfe552ccbb5d2bfd54ea56ad0863785b21097e26404bbad8d';pf=js(proof_path);assert pf['passed'] and pf['sample_count']==200 and pf['actual_forward_count']==13383 and pf['original_protocol_final_qualification_complete'] and not any(pf[k] for k in ['error_indices','invalid_indices','empty_indices','truncated_indices'])
 folder=Path(pf['report_dir']);assert {p.name for p in folder.iterdir()}==FIVE and all(sha(folder/n)==h for n,h in pf['report_sha256'].items())
 report=js(folder/'eval_report.json');assert report['metrics']=={'Accuracy':.84} and report['statistics']['total_samples']==200
 outputs=[json.loads(s) for s in (folder/'sample_outputs.jsonl').read_text().splitlines()];wrong={r['index'] for r in outputs if str(r['prediction']).strip()!=str(r['reference']).strip()};assert len(wrong)==32
 keep=wrong|set(range(10))|set(range(90,110))|set(range(190,200));assert len(keep)==63
 line=report['operator_lineage']['actual_execution'];trace=[json.loads(s) for s in Path(line['sample_trace']).read_text().splitlines()];assert [r['index'] for r in trace]==list(range(200))
 candidates={}
 for e in trace:
  i=e['index'];p=Path(e['actual_execution_file']);assert p==HOME/('sample_%05d.json'%i) and sha(p)==e['actual_execution_SHA256'];v=js(p);assert v['index']==i and v['error'] is None and not v['original_budget_hit']
  npz=Path(v['archive']);assert npz==HOME/('sample_%05d.npz'%i) and npz.stat().st_size==v['archive_bytes']
  if i not in keep:
   assert outputs[i]['error'] is None and str(outputs[i]['prediction']).strip()==str(outputs[i]['reference']).strip();st=npz.stat();row=dict(model_id=MID,index=i,logical_indices=[i],home=str(HOME),path=str(npz),bytes=st.st_size,allocated_bytes=st.st_blocks*512,SHA256=v['archive_SHA256'],proof=str(proof_path),proof_SHA256=sha(proof_path));exact(row);candidates[str(npz)]=row
 assert len(candidates)==137
 cfg=js(R/'configs/retention/verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json');helpers=base.old_helpers(cfg)
 accepted={m:dict(proof=v['accepted_formal200_proof'],report_dir=v['accepted_formal200_report']) for m,v in reg.items() if isinstance(v,dict) and v.get('accepted_formal200_proof')}
 foreign,metadata=base.sources_and_dependencies(reg,accepted,candidates,helpers)
 for n in foreign:candidates.pop(n,None)
 assert candidates and len(candidates)<=137
 protected=set(metadata)|{Path(n) for n in canonical}|{audit_path,LED,D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json',proof_path}
 for entry in state['models'].values():
  if entry['status']=='done':
   if entry.get('active_report_files'):protected.update(Path(v['path']) for v in entry['active_report_files'].values())
   else:protected.update(Path(entry['report_dir'])/n for n in FIVE)
 for home in [HOME,J,Path(job['spec']['alignment_approval_review']).parent,PUB]:protected.update(p for p in home.rglob('*') if p.is_file() and str(p) not in candidates)
 approval=js(job['spec']['alignment_approval_review']);controls=[js(approval['true3_proof'])]+[js(x['proof']) for x in done['accepted'] if x['sample_count']==3]
 for p in controls:
  assert p['passed'] and p['sample_count']==3;protected.update(Path(n) for n in p['external_evidence_sha256'] if Path(n).is_relative_to(S) and Path(n).is_file());protected.update(Path(p['report_dir'])/n for n in FIVE)
 for root in [R/'scripts',R/'configs',R/'universal_eval']:protected.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.ir'])
 assert not set(candidates)&{str(p) for p in protected};hashes={}
 for i,p in enumerate(sorted(protected)):
  hashes[str(p)]=sha(p)
  if i%200==0:put(H/'progress.json',dict(UTC=now(),phase='hash_all_protected_reports_sources_configs_logs_wrong_examples_controls',checked=i+1,total=len(protected)))
 quiet();(H/'ledger_before.original').write_bytes(LED.read_bytes())
 p=dict(UTC=now(),canonical_SHA256=canonical,ledger_before_SHA256=sha(LED),full_target_audit_SHA256=sha(audit_path),proof_SHA256=sha(proof_path),all32_wrong_indices=sorted(wrong),complete_original_outer10_example_indices=sorted(set(range(10))|set(range(90,110))|set(range(190,200))),keep_indices=sorted(keep),remove=list(candidates.values()),remove_count=len(candidates),remove_bytes=sum(x['bytes'] for x in candidates.values()),remove_allocated_bytes=sum(x['allocated_bytes'] for x in candidates.values()),foreign_references_preserved=foreign,protected_SHA256=hashes,protected_count=len(hashes),free_before_bytes=shutil.disk_usage(R).free,all_full5_records_source_configs_snapshots_logs_wrong_small_failed_unverified_values_preserved=True,no_deletion_yet=True)
 put(H/'plan.json',p);return dict(plan_SHA256=sha(H/'plan.json'),remove_count=p['remove_count'],remove_bytes=p['remove_bytes'],protected_count=len(hashes),foreign_references=len(foreign),projected_free_bytes=p['free_before_bytes']+p['remove_allocated_bytes'],no_deletion_yet=True)
def apply(reviewed_plan_SHA256):
 quiet();assert sha(H/'plan.json')==reviewed_plan_SHA256;p=js(H/'plan.json');assert not (H/'actual_completion.json').exists();assert sha(LED)==p['ledger_before_SHA256']
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');quiet(True)
  for n,h in p['protected_SHA256'].items():assert sha(n)==h,('Protected file changed; no deletion',n)
  for row in p['remove']:assert row['index'] not in p['keep_indices'] and row['path'] not in p['protected_SHA256'];exact(row)
  assert all(sha(n)==h for n,h in p['canonical_SHA256'].items());quiet(True)
  reg=js(D/'registry.json');accepted={m:dict(proof=v['accepted_formal200_proof'],report_dir=v['accepted_formal200_report']) for m,v in reg.items() if isinstance(v,dict) and v.get('accepted_formal200_proof')};cfg=js(R/'configs/retention/verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json');helpers=base.old_helpers(cfg)
  foreign,_=base.sources_and_dependencies(reg,accepted,{r['path']:r for r in p['remove']},helpers);assert not foreign,('New foreign/FD references: retain all',foreign)
  manifest=H/'removed_exact_accepted_correct_nonexample_NPZ_original_SHA256.jsonl';assert not manifest.exists();free=shutil.disk_usage(R).free;removed=[];failure=None
  try:
   with manifest.open('x',encoding='utf-8') as file:
    for row in p['remove']:
     path=Path(row['path']);exact(row);path.unlink();removed.append(row);file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush()
  except BaseException:failure=traceback.format_exc()
  bad=[n for n,h in p['protected_SHA256'].items() if n!=str(LED) and sha(n)!=h];ok=not failure and not bad and len(removed)==p['remove_count'];actual=dict(UTC=now(),actual_returncode=0 if ok else 1,plan_SHA256=reviewed_plan_SHA256,removed_files=len(removed),removed_bytes=sum(v['bytes'] for v in removed),removed_allocated_bytes=sum(v['allocated_bytes'] for v in removed),manifest=str(manifest),manifest_SHA256=sha(manifest),protected_files=p['protected_count'],all_protected_SHA256_unchanged=not bad,error=failure,protected_errors=bad,free_before_bytes=free,free_after_bytes=shutil.disk_usage(R).free,all32_wrong_and63_required_complete_numeric_examples_small_failed_unverified_retained=True,all_five_records_source_configs_snapshots_logs_original_proof_bytes_retained=True,pruned137_or_exact_foreign_reduced_set_no_longer_all_array_reverifiable=True,canonical_scores_state_registry_and_both_workbooks_unchanged=True)
  put(H/'actual_completion.json',actual);assert sha(LED)==p['ledger_before_SHA256'];ledger=js(LED);ledger['events'].append(dict(UTC=actual['UTC'],authorization='User repeated instruction: remove accepted correct nonexample intermediate matrices after full audit; keep full reports/source/config/logs and necessary complete numeric examples; continue serial immediately',receipt=str(H/'actual_completion.json'),receipt_SHA256=sha(H/'actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=len(removed),removed_bytes=actual['removed_bytes'],all_original_records_kept=True,intentional_nonexample_matrix_unavailability=True,actual_returncode=actual['actual_returncode']));put(LED,ledger)
  assert all(sha(n)==h for n,h in p['canonical_SHA256'].items()) and ok
  return dict(actual=actual,actual_SHA256=sha(H/'actual_completion.json'),ledger_SHA256=sha(LED))
if __name__=='__main__':
 assert len(sys.argv) in [2,3]
 result=plan() if sys.argv[1]=='plan' and len(sys.argv)==2 else apply(sys.argv[2]) if sys.argv[1]=='apply' and len(sys.argv)==3 else (_ for _ in ()).throw(ValueError('Require plan or reviewed apply SHA'))
 print(json.dumps(result,ensure_ascii=False))
