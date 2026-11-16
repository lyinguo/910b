"""Explicit user retention consent; selected complete, verified runs only.

The exact approved input binds eleven owned runs, original first/middle/last
groups, accepted proofs, and reports. It never prunes a live/failed/incomplete
run, weights, datasets, code, configs, logs, reports, or another persistent disk.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,shutil,re,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';sys.path[:0]=[str(R),str(R/'scripts')]
H=B/'authorized_eleven_complete_verified_runs_nonexample_retention_for_RedHatE2B_capacity_20261116';POL=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'
ALLOWED={'zelk12/MT-Gen1-gemma-3-12B', 'google/gemma-4-E2B-it', 'unsloth/medgemma-1.5-4b-it', 'lapa-llm/lapa-12b-pt', 'google/pix2struct-textcaps-base', 'heretic-org/gemma-3-4b-it-heretic', 'google/paligemma-3b-pt-448', 'RedHatAI/gemma-4-E4B-it', 'google/gemma-4-E4B-it', 'google/paligemma2-3b-ft-docci-448', 'coder3101/gemma-4-E2B-it-qat-q4_0-unquantized-heretic'}
FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'}
def now():return datetime.now(timezone.utc).isoformat()
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(n,v):
 p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def idle():
 from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
 assert not runner_lock_owners() and not [p for p in project_processes() if p['role'] in ['model','alignment_runner','CPU_verifier_or_audit','CPU_verifier','verifier','independent_verifier']]
def classify(f,home,kind,mid):
 rel=f.relative_to(home)
 if kind=='batch10_grouped':
  if len(rel.parts)!=3 or rel.parts[0]!='groups' or not re.fullmatch(r'\d{3}',rel.parts[1]) or rel.parts[2]!='full_vocab_KV_inputs_visual.npz':return None
  i=int(rel.parts[1]);assert 0<=i<20;return i,i in {0,9,19}
 if kind=='sample_serial':
  if len(rel.parts)!=1 or not re.fullmatch(r'sample_\d{5}\.npz',f.name):return None
  i=int(f.stem.split('_')[-1]);assert 0<=i<200
  keep=(set(range(4))|set(range(96,104))|set(range(196,200))) if mid.startswith('google/gemma-4-') else (set(range(10))|set(range(90,100))|set(range(190,200)))
  return i,i in keep
 if kind=='grouped_serial_one_based':
  if len(rel.parts)!=3 or rel.parts[0]!='groups' or not re.fullmatch(r'\d{4}',rel.parts[1]) or rel.parts[2]!='actual_visual_positions_masks_full_vocab.npz':return None
  i=int(rel.parts[1]);assert 1<=i<=200;return i,i in (set(range(1,11))|set(range(90,111))|set(range(190,201)))
 assert kind in ['full_vocab_and_visual','visual_features','native_samples']
 if len(rel.parts)!=2 or rel.parts[0]!=kind or not re.fullmatch(r'\d+\.npz',f.name):return None
 i=int(f.stem);assert 0<=i<200;return i,i in (set(range(10))|set(range(90,110))|set(range(190,200)))
def historical_removed():
 known={}
 for event in js(LED)['events']:
  name=event.get('manifest') or event.get('removed_manifest')
  if not name or not Path(name).is_file():continue
  f=Path(name);digest=event.get('manifest_SHA256') or event.get('removed_manifest_SHA256')
  if digest:assert sha(f)==digest
  for line in f.read_text().splitlines():
   v=json.loads(line)
   if v.get('path') and v.get('SHA256'):known[v['path']]=v['SHA256']
 return known
def plan(input_path,input_sha):
 assert sha(input_path)==input_sha;v=js(input_path);assert set(s['model_id'] for s in v['scopes'])==ALLOWED and len(v['scopes'])==len(ALLOWED)
 assert v['only_after_RedHatE4B_full200_and_dual_publication_completed'] and v['no_live_or_failed_or_small_sample_arrays_in_scope']
 binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert binding['real_record_count']>=174 and sha('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')==binding['local_workbook_sha256'];H.mkdir(exist_ok=False)
 state=js(D/'state.json');reg=js(D/'registry.json');old_removed=historical_removed();removed=[];kept=[];protected=set();scope_records=[];scope_owners={}
 critical={D/n for n in ['state.json','registry.json','PAUSE','POLICY.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json']};critical.add(Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx'));protected.update(critical)
 for mid,e in state['models'].items():
  if e.get('status')=='done':protected.update(Path(e['report_dir'])/n for n in FIVE)
 for s in v['scopes']:
  mid=s['model_id'];e=state['models'][mid];rr=reg[mid];pf=Path(s['proof']);home=Path(s['home']).resolve();assert e['status']=='done' and rr['accepted_formal200_proof']==str(pf) and rr['accepted_formal200_proof_sha256']==s['proof_SHA256']==sha(pf)
  assert home.is_relative_to(R/'workspace/diagnostics/model_runs') and not home.is_symlink();p=js(pf);assert p['passed'] and p['sample_count']==200 and not p.get('error_indices');assert set(p['report_sha256'])==FIVE;protected.add(pf)
  for n,h in p['report_sha256'].items():f=Path(e['report_dir'])/n;assert sha(f)==h;protected.add(f)
  observed=set();missing=[]
  for name,h in p['external_evidence_sha256'].items():
   f=Path(name)
   if f.suffix!='.npz' or not f.is_relative_to(home):continue
   c=classify(f,home,s['kind'],mid)
   if c is None:continue
   index,keep=c;observed.add(index)
   if not f.exists():assert not keep and old_removed.get(name)==h,('Unexplained missing protected evidence',name);missing.append(name);continue
   assert not f.is_symlink() and f.resolve().is_relative_to(home) and f.stat().st_nlink==1,name
   row=dict(path=name,SHA256=h,bytes=f.stat().st_size,model_id=mid,home=str(home),kind=s['kind'],index=index,proof=str(pf),proof_SHA256=s['proof_SHA256'])
   if keep:kept.append(row);protected.add(f)
   else:removed.append(row)
  assert observed==(set(range(1,201)) if s['kind']=='grouped_serial_one_based' else set(range(20 if s['kind']=='batch10_grouped' else 200))),(mid,observed)
  scope_records.append(dict(s,previously_authorized_missing_nonexamples=missing,original_first_middle_last_groups_and_small_samples_preserved=True));scope_owners[str(home)]=mid
 # Preserve any exact array consumed by a different active accepted model.
 foreign=set();candidate_paths={x['path'] for x in removed};owner_by_file={x['path']:x['model_id'] for x in removed}
 for mid,rr in reg.items():
  if not isinstance(rr,dict) or not rr.get('accepted_formal200_proof'):continue
  f=Path(rr['accepted_formal200_proof']);assert sha(f)==rr['accepted_formal200_proof_sha256'];pp=js(f)
  for name in pp.get('external_evidence_sha256',{}):
   if name in candidate_paths:
    owner=owner_by_file[name]
    if owner!=mid:foreign.add(name)
 retained_foreign=[x for x in removed if x['path'] in foreign];removed=[x for x in removed if x['path'] not in foreign];protected.update(Path(x['path']) for x in retained_foreign)
 candidates={x['path'] for x in removed};assert len(candidates)==len(removed) and 0<len(removed)<1600 and sum(x['bytes'] for x in removed)<65*1024**3
 for s in v['scopes']:
  protected.update(p for p in Path(s['home']).rglob('*') if p.is_file() and str(p) not in candidates)
 for folder in [R/'universal_eval',R/'scripts',R/'configs']:protected.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json','.ir'])
 assert not candidates.intersection(map(str,protected));before={str(f):sha(f) for f in sorted(protected)};ledger=LED.read_bytes();(H/'ledger_before_original_bytes.json').write_bytes(ledger)
 data=dict(UTC=now(),reviewed_input=str(input_path),reviewed_input_SHA256=input_sha,authorization=js(POL)['authorization'],authorization_SHA256=js(POL)['authorization_SHA256'],policy_SHA256=sha(POL),ledger_before_SHA256=hashlib.sha256(ledger).hexdigest(),scopes=scope_records,remove=removed,keep=kept,foreign_current_model_arrays_retained=retained_foreign,remove_count=len(removed),remove_bytes=sum(x['bytes'] for x in removed),protected_SHA256=before,protected_count=len(before),free_before_bytes=shutil.disk_usage(D).free,critical_shared_paths=sorted(map(str,critical)),all_full_five_reports_all_records_source_YAML_snapshots_logs_preserved=True,no_current_failed_or_small_sample_arrays_in_scope=True,no_G_score_state_registry_or_original_report_proof_change=True,not_a_worst_case_capacity_guarantee=True)
 put('plan.json',data);print(json.dumps(dict(plan=str(H/'plan.json'),plan_SHA256=sha(H/'plan.json'),remove_count=data['remove_count'],remove_bytes=data['remove_bytes'],protected_count=data['protected_count'],free_before_bytes=data['free_before_bytes'],retained_original_examples=len(kept),foreign_arrays_kept=len(retained_foreign))))
def apply(plan_sha):
 import fcntl
 assert sha(H/'plan.json')==plan_sha;p=js(H/'plan.json');assert sha(LED)==p['ledger_before_SHA256'];idle()
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,h in p['protected_SHA256'].items():assert sha(name)==h,name
  for i,x in enumerate(p['remove']):
   f=Path(x['path']);home=Path(x['home']).resolve();assert f.resolve().is_relative_to(home) and home.is_relative_to(R/'workspace/diagnostics/model_runs') and not f.is_symlink() and f.suffix=='.npz' and f.stat().st_nlink==1 and f.stat().st_size==x['bytes'] and sha(f)==x['SHA256'],str(f)
   assert classify(f,home,x['kind'],x['model_id'])==(x['index'],False)
   if i%25==0:put('progress.json',dict(UTC=now(),phase='full_original_SHA_before_any_removal',checked=i+1,total=p['remove_count']))
  manifest=H/'removed_exact_original_paths_SHA256.jsonl';assert not manifest.exists();removed=[];failure=None
  try:
   with manifest.open('x',encoding='utf-8') as out:
    for i,x in enumerate(p['remove']):
     f=Path(x['path']);assert f.resolve().is_relative_to(Path(x['home']).resolve()) and not f.is_symlink() and f.stat().st_size==x['bytes'];f.unlink();removed.append(x);out.write(json.dumps(x,ensure_ascii=False)+'\n');out.flush()
     if i%25==0:put('progress.json',dict(UTC=now(),phase='removing_exact_preverified_nonexample_NPZ_only',removed=i+1,total=p['remove_count']))
  except BaseException:failure=traceback.format_exc()
  assert sha(LED)==p['ledger_before_SHA256']
  post_errors=[]
  for name,h in p['protected_SHA256'].items():
   if sha(name)!=h:post_errors.append(name)
  ok=failure is None and not post_errors and len(removed)==p['remove_count'] and all(not Path(x['path']).exists() for x in removed)
  actual=dict(UTC=now(),actual_returncode=0 if ok else 1,plan_SHA256=plan_sha,removed_files=len(removed),removed_bytes=sum(x['bytes'] for x in removed),removed_manifest=str(manifest),removed_manifest_SHA256=sha(manifest),protected_files=p['protected_count'],all_protected_SHA256_unchanged=not post_errors,post_protection_errors=post_errors,error=failure,free_before_bytes=p['free_before_bytes'],free_after_bytes=shutil.disk_usage(D).free,retained_original_examples=len(p['keep']),foreign_current_model_arrays_retained=len(p['foreign_current_model_arrays_retained']),all_full_five_reports_all_records_source_YAML_snapshots_logs_preserved=True,old_proofs_unchanged_and_pruned_arrays_explicitly_no_longer_available=True,no_NN_or_canonical_score_write=True);put('actual_completion.json',actual)
  ledger=js(LED);ledger['events'].append(dict(UTC=actual['UTC'],authorization=p['authorization'],authorization_SHA256=p['authorization_SHA256'],receipt=str(H/'actual_completion.json'),receipt_SHA256=sha(H/'actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=actual['removed_files'],removed_bytes=actual['removed_bytes'],all_original_records_kept=True,intentional_nonexample_matrix_unavailability=True,actual_returncode=actual['actual_returncode']));tmp=LED.with_name(LED.name+'.tmp.'+str(os.getpid()));tmp.write_bytes(json.dumps(ledger,ensure_ascii=False,indent=2).encode());os.replace(tmp,LED)
  print(json.dumps(dict(actual=actual,actual_SHA256=sha(H/'actual_completion.json'),ledger_SHA256=sha(LED))));assert ok
def main():
 idle();assert sha(POL)=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7';policy=js(POL);assert policy['explicit_user_consent'] and policy['permanently_preserve_full_five_and_all_records_source_YAML_logs'];assert sha(policy['authorization'])==policy['authorization_SHA256']=='06eba6a16df1559793352231cc9527955f5f13d58eb5e96929a110695afa9ee3'
 if sys.argv[1]=='plan':plan(Path(sys.argv[2]),sys.argv[3])
 else:assert sys.argv[1]=='apply';apply(sys.argv[2])
if __name__=='__main__':main()
