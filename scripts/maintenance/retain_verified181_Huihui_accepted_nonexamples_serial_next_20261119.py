"""Exact current-accepted nonexample retention for the user's immediate serial continuation.
The previous immutable retention helper supplies only pure path/index functions.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,stat,shutil,fcntl,re,importlib.util,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';S=R/'workspace/diagnostics/model_runs'
H=B/'authorized_verified181_Huihui_only_accepted_nonexample_serial_next_20261119';CFG=R/'configs/retention/verified181_Huihui_only_accepted_nonexample_serial_next_20261119.json';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
sys.path[:0]=[str(R),str(R/'scripts')]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(p,value):
 p=Path(p);tmp=p.with_name(p.name+'.owned.'+str(os.getpid())+'.tmp');tmp.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode());os.replace(tmp,p)
def old_helpers(cfg):
 source=R/'scripts/retain_verified179_accepted_current_nonexample_matrices_user_request_20261117.py';assert sha(source)==cfg['prior_helper_SHA256']
 s=importlib.util.spec_from_file_location('unaltered_retention_pure_helpers',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def guard_idle(cfg,owned_lock=False):
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 from eager_corrective_control_safety_2203 import proc,core_unchanged
 core_unchanged();owners=runner_lock_owners()
 assert not owners if not owned_lock else all(int(x['PID'])==os.getpid() for x in owners),owners
 allowed={}
 for item in cfg['known_no_NN_waiters']:
  source=R/item['relative'];assert sha(source)==item['source_SHA256']
  if not (Path('/proc')/str(item['PID'])).is_dir():continue
  v=proc(item['PID']);assert v['start_ticks']==item['start_ticks'] and v['cwd']==item['cwd'] and v['argv']==['python3','-B',str(source)],v
  state=js(item['state']);assert state['phase'] in item['allowed_idle_phases'] and not state.get('actual_NN_NF') and not state.get('continuation_holds_original_runner_lock'),state
  allowed[item['PID']]=v
 unknown=[v for v in project_processes() if int(v['PID']) not in [os.getpid(),*allowed]]
 assert not unknown,('Unrecognized project process, do not ignore it',unknown)
 return dict(known_bound_idle_waiters=allowed,original_lock_owners=owners)
def is_real_dependency_reference(owner,candidate_owner,metadata_path,current_acceptance_proof,current_five):
 # Own acceptance inventory and full reports are immutable records to retain;
 # listing their accepted arrays is not a second task dependency.
 # Every other metadata path/owner keeps the original conservative protection.
 return not (owner==candidate_owner and str(metadata_path) in {str(current_acceptance_proof),*map(str,current_five)})

def sources_and_dependencies(reg,accepted,candidates,helpers):
 foreign={};metadata=set()
 def mark(path,reason):foreign.setdefault(path,[]).append(reason)
 for mid,row in accepted.items():
  pf=Path(row['proof']);metadata.add(pf);proof=js(pf)
  for n in helpers.paths_in(proof):
   if n in candidates and candidates[n]['model_id']!=mid:mark(n,'another_current_accepted_model:'+mid)
  for k,v in proof.items():
   if k!='external_evidence_sha256' and k.endswith(('evidence_sha256','evidence_SHA256')):
    for n in helpers.paths_in(v):
     if n in candidates:mark(n,'additional_static_evidence_map:'+mid+':'+k)
 for mid,sp in reg.items():
  if not isinstance(sp,dict):continue
  files={R/n for n in sp.get('source_sha256',{}) if not Path(n).is_absolute() and Path(n).suffix in ['.json','.yaml','.yml']}
  for n in helpers.paths_in(sp):
   p=Path(n)
   if p.is_file() and p.suffix=='.json' and p.stat().st_size<=8*1024**2:files.add(p)
  for p in files:
   if not p.is_file():continue
   metadata.add(p)
   if p.suffix!='.json':continue
   try:content=js(p)
   except (ValueError,UnicodeDecodeError):continue
   for n in helpers.paths_in(content):
    if n in candidates:
     own=accepted.get(mid);own_proof=Path(own['proof']) if own else None
     own_five=[Path(own['report_dir'])/name for name in helpers.FIVE] if own else []
     if is_real_dependency_reference(mid,candidates[n]['model_id'],p,own_proof,own_five):
      mark(n,'current_registry_static_dependency:'+mid+':'+str(p))
 for jp in B.glob('*/job.json'):
  metadata.add(jp)
  for n in helpers.paths_in(js(jp)):
   if n in candidates:mark(n,'private_job_reference:'+str(jp))
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:n=os.readlink(fd)
    except OSError:continue
    if n in candidates:mark(n,'open_FD_PID:'+proc.name)
  except OSError:pass
 return foreign,metadata
def plan(discovery_SHA256):
 cfg=js(CFG);helpers=old_helpers(cfg);guard_idle(cfg);assert sha(H/'discovery.json')==discovery_SHA256==cfg['discovery_SHA256'];disc=js(H/'discovery.json')
 assert sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json')=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7'
 assert cfg['keep_original_logical_indices']==sorted(helpers.KEEP) and cfg['allowed_model_ids']==[s['model_id'] for s in disc['scopes']]
 assert not (H/'plan_owner_reference_v2.json').exists()
 state=js(D/'state.json');reg=js(D/'registry.json');assert sum(x['status']=='done' for x in state['models'].values())==181
 assert all(sha(p)==h for p,h in disc['canonical_SHA256'].items())
 accepted={v['model_id']:v for v in disc['all_selected_proof_metadata']};scopes=disc['scopes'];candidates={r['path']:r for s in scopes for r in s['remove']}
 for s in scopes:
  assert s['model_id'] in cfg['allowed_model_ids'] and sha(s['proof'])==s['proof_SHA256'];pf=js(s['proof']);assert pf['passed'] and pf['sample_count']==200 and not pf.get('historical_record_reuse') and not pf.get('error_indices')
  assert state['models'][s['model_id']]['report_dir']==s['report_dir'] and set(pf['report_sha256'])==helpers.FIVE
  assert all(sha(Path(s['report_dir'])/n)==h for n,h in pf['report_sha256'].items())
 foreign,metadata=sources_and_dependencies(reg,accepted,candidates,helpers)
 for n in foreign:candidates.pop(n,None)
 assert candidates and len(candidates)<=cfg['max_remove_files'] and sum(r['bytes'] for r in candidates.values())<=cfg['max_remove_bytes']
 protected=set(metadata)|{Path(n) for n in disc['canonical_SHA256']}|{CFG,H/'discovery.json'}
 for e in state['models'].values():
  if e['status']=='done':protected.update(Path(e['report_dir'])/n for n in helpers.FIVE)
 for s in scopes:protected.update(p for p in Path(s['home']).rglob('*') if p.is_file() and str(p) not in candidates)
 for root in [R/'scripts',R/'configs',R/'universal_eval']:
  protected.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.ir'])
 assert not set(candidates)&{str(p) for p in protected}
 protected_SHA={};sizes=0
 for i,p in enumerate(sorted(protected)):
  protected_SHA[str(p)]=sha(p);sizes+=p.stat().st_size
  if i%200==0:put(H/'progress.json',dict(UTC=now(),phase='hashing_all_protected_reports_sources_logs_examples',checked=i+1,total=len(protected),bytes_checked=sizes))
 oldledger=LED.read_bytes();(H/'ledger_before_owner_reference_v2.original').write_bytes(oldledger)
 value=dict(UTC=now(),config_SHA256=sha(CFG),discovery_SHA256=discovery_SHA256,authorization=cfg['latest_actual_user_authorization'],policy_SHA256=sha(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json'),ledger_before_SHA256=sha(LED),canonical_before_SHA256=disc['canonical_SHA256'],scopes=scopes,remove=list(candidates.values()),foreign_or_open_reference_reasons_preserved=foreign,protected_SHA256=protected_SHA,protected_count=len(protected_SHA),protected_bytes=sizes,remove_count=len(candidates),remove_bytes=sum(v['bytes'] for v in candidates.values()),remove_allocated_bytes=sum(v['allocated_bytes'] for v in candidates.values()),free_before_bytes=shutil.disk_usage(R).free,all_reports_sources_configs_logs_needed_examples_retained=True,all_small_failed_incomplete_unverified_values_preserved=True,whole_keep_groups_original0_9_90_109_190_199_preserved=True,no_other_disk_or_capacity_guard_change=True)
 put(H/'plan_owner_reference_v2.json',value)
 return dict(plan_SHA256=sha(H/'plan_owner_reference_v2.json'),remove_count=value['remove_count'],remove_bytes=value['remove_bytes'],remove_allocated_bytes=value['remove_allocated_bytes'],protected_count=value['protected_count'],protected_bytes=sizes,foreign_or_open_files_preserved=len(foreign),foreign_models_summary={mid:sum(row['model_id']==mid for name,row in {r['path']:r for s in scopes for r in s['remove']}.items() if name in foreign) for mid in cfg['allowed_model_ids']},scopes=[dict(model_id=s['model_id'],remove_count=sum(v['model_id']==s['model_id'] for v in candidates.values()),remove_bytes=sum(v['bytes'] for v in candidates.values() if v['model_id']==s['model_id'])) for s in scopes],free_before_bytes=value['free_before_bytes'],projected_free_after_bytes=value['free_before_bytes']+value['remove_allocated_bytes'],no_deletion_yet=True)
def apply(reviewed_plan_SHA256):
 cfg=js(CFG);helpers=old_helpers(cfg);guard_idle(cfg);assert sha(H/'plan_owner_reference_v2.json')==reviewed_plan_SHA256;p=js(H/'plan_owner_reference_v2.json')
 assert sha(CFG)==p['config_SHA256'] and sha(LED)==p['ledger_before_SHA256'];assert not (H/'actual_completion.json').exists()
 from eager_corrective_control_safety_2203 import kernel_lock_owned
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');guard_idle(cfg,owned_lock=True)
  for i,(n,h) in enumerate(p['protected_SHA256'].items()):
   assert sha(n)==h,('Protected changed before any removal',n)
   if i%200==0:put(H/'progress.json',dict(UTC=now(),phase='fresh_protected_SHA_checks_before_any_deletion',checked=i+1,total=p['protected_count']))
  for i,row in enumerate(p['remove']):
   f=Path(row['path']);home=Path(row['home']);assert home.is_relative_to(S) and helpers.removable_file(f,home) and f.stat().st_size==row['bytes']
   assert not set(row['logical_indices'])&helpers.KEEP and str(f) not in p['protected_SHA256'];assert sha(row['proof'])==row['proof_SHA256'] and sha(f)==row['SHA256'],str(f)
   if i%100==0:put(H/'progress.json',dict(UTC=now(),phase='fresh_every_exact_candidate_complete_SHA_before_any_deletion',checked=i+1,total=p['remove_count']))
  assert all(sha(n)==h for n,h in p['canonical_before_SHA256'].items());guard_idle(cfg,owned_lock=True)
  manifest=H/'removed_exact_accepted_nonexample_NPZ_original_SHA256.jsonl';assert not manifest.exists();removed=[];failure=None;free=shutil.disk_usage(R).free
  try:
   with manifest.open('x',encoding='utf8') as out:
    for i,row in enumerate(p['remove']):
     f=Path(row['path']);assert helpers.removable_file(f,Path(row['home'])) and f.stat().st_size==row['bytes'];f.unlink();removed.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
     if i%100==0:put(H/'progress.json',dict(UTC=now(),phase='precise_verified_nonexample_NPZ_removal',removed=i+1,total=p['remove_count']))
  except BaseException:failure=traceback.format_exc()
  bad=[n for n,h in p['protected_SHA256'].items() if sha(n)!=h];ok=not failure and not bad and len(removed)==p['remove_count']
  actual=dict(UTC=now(),actual_returncode=0 if ok else 1,plan_SHA256=reviewed_plan_SHA256,removed_files=len(removed),removed_bytes=sum(v['bytes'] for v in removed),removed_allocated_bytes=sum(v['allocated_bytes'] for v in removed),manifest=str(manifest),manifest_SHA256=sha(manifest),protected_files=p['protected_count'],all_protected_SHA256_unchanged=not bad,protected_errors=bad,error=failure,free_before_bytes=free,free_after_bytes=shutil.disk_usage(R).free,all_five_and_all_records_sources_configs_snapshots_logs_and_required_full_examples_retained=True,no_small_failed_incomplete_unverified_arrays_deleted=True,all_original_proofs_unchanged_exact_pruned_arrays_no_longer_available=True,no_model_or_CPU200_replay=True,canonical_scores_state_registry_and_workbooks_unchanged=True)
  put(H/'actual_completion.json',actual)
  assert sha(LED)==p['ledger_before_SHA256'];ledger=js(LED);ledger['events'].append(dict(UTC=actual['UTC'],authorization=cfg['latest_actual_user_authorization'],receipt=str(H/'actual_completion.json'),receipt_SHA256=sha(H/'actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=len(removed),removed_bytes=actual['removed_bytes'],all_original_records_kept=True,intentional_nonexample_matrix_unavailability=True,actual_returncode=actual['actual_returncode']))
  put(LED,ledger);assert all(sha(n)==h for n,h in p['canonical_before_SHA256'].items() if n!=str(LED));assert ok
  return dict(actual=actual,actual_SHA256=sha(H/'actual_completion.json'),ledger_SHA256=sha(LED))
