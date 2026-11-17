"""Prune only exact NPZ files from current, independently accepted complete200.

All reports, records, source/configuration, logs, original small samples and
whole first/middle/last logical groups remain. No recursive deletion occurs.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import os,json,re,hashlib,stat,shutil,fcntl,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';S=R/'workspace/diagnostics/model_runs';H=B/'authorized_verified179_accepted_current_nonexample_retention_user_continue_Prithiv_20261117';POL=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json';LED=D/'A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json';FIVE={'eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json'};KEEP=set(range(10))|set(range(90,110))|set(range(190,200))
sys.path[:0]=[str(R),str(R/'scripts')]
from A100_alignment_incremental_monitor_20261103 import project_processes,runner_lock_owners
from eager_corrective_control_safety_2203 import core_unchanged,kernel_lock_owned
def now():return datetime.now(timezone.utc).isoformat()
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,v):
 p=H/name;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes(json.dumps(v,ensure_ascii=False,indent=2).encode());os.replace(t,p)
def idle():
 assert not runner_lock_owners() and not project_processes(),'Any other project Python process or original lock is active'
def paths_in(v):
 if isinstance(v,str):
  if v.startswith('/'):yield v
 elif isinstance(v,list):
  for x in v:yield from paths_in(x)
 elif isinstance(v,dict):
  for k,x in v.items():
   if isinstance(k,str) and k.startswith('/'):yield k
   yield from paths_in(x)
def proof_for(mid,e,reg):
 p=reg[mid].get('accepted_formal200_proof')
 if p:
  assert sha(p)==reg[mid]['accepted_formal200_proof_sha256'];return Path(p)
 a=[x for x in e.get('attempts',[]) if x.get('count')==200 and x.get('report_dir')==e['report_dir'] and isinstance(x.get('independent_verification'),str) and Path(x['independent_verification']).is_file()]
 return Path(a[-1]['independent_verification']) if a else None
def own_home(line):
 found=set()
 for key in ['evidence_root','per_sample_evidence','sample_trace','weight_evidence','weight_loading_evidence','full_numeric_group_index','final_trace_manifest','runtime_evidence','actual_batch_manifest','execution_trace','trace_path']:
  x=line.get(key)
  if isinstance(x,str) and Path(x).is_absolute() and Path(x).is_relative_to(S):found.add(S/Path(x).relative_to(S).parts[0])
 return next(iter(found)) if len(found)==1 else None
def sample_group_index(line):
 name=line.get('per_sample_evidence') or line.get('sample_trace')
 if not isinstance(name,str) or not Path(name).is_file():return {},None
 rows=[json.loads(x) for x in Path(name).read_text().splitlines()];groups=defaultdict(set)
 if len(rows)!=200 or any(not isinstance(x.get('index'),int) for x in rows) or sorted(x['index'] for x in rows)!=list(range(200)):return {},Path(name)
 for x in rows:
  g=x.get('group',x.get('actual_group'))
  if isinstance(g,int):groups[g].add(x['index'])
 return dict(groups),Path(name)
def indices_for(relative,group_map):
 p=Path(relative);parts=p.parts
 if len(parts)==1:
  if re.fullmatch(r'sample_\d{5}\.npz',p.name):i=int(p.stem[7:]);return {i} if 0<=i<200 else None
  if re.fullmatch(r'full_native_group_\d{2}\.npz',p.name):g=int(p.stem.rsplit('_',1)[1]);return set(range(g*10,g*10+10)) if 0<=g<20 else None
  if re.fullmatch(r'group_\d{2}_arrays\.npz',p.name):g=int(p.stem.split('_')[1]);return set(range(g*10,g*10+10)) if 0<=g<20 else None
 if len(parts)==2 and parts[0] in ['visual_features','native_samples','full_vocab_and_visual','samples']:
  m=re.fullmatch(r'(\d+)(?:_(?:visual|full_logits))?\.npz',p.name)
  if m:
   i=int(m[1]);return {i} if 0<=i<200 else None
 if len(parts)==2 and parts[0]=='generation_groups':
  m=re.fullmatch(r'(\d+)_(?:visual|full_logits)\.npz',p.name)
  if m:return group_map.get(int(m[1]))
 if len(parts)==3 and parts[0]=='groups' and parts[1].isdigit():
  if p.name in ['actual_visual_positions_cache_full_vocab.npz','actual_visual_positions_masks_full_vocab.npz']:return group_map.get(int(parts[1]))
  if p.name=='full_vocab_KV_inputs_visual.npz':
   g=int(parts[1]);return set(range(g*10,g*10+10)) if 0<=g<20 else None
 return None
def removable_file(p,home):
 s=p.lstat();return stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_nlink==1 and s.st_dev==R.stat().st_dev and p.suffix=='.npz' and p.resolve().is_relative_to(home.resolve()) and not home.is_symlink()
def discover(config):
 state=js(D/'state.json');reg=js(D/'registry.json');assert len([e for e in state['models'].values() if e['status']=='done'])==179
 result=[];skips=[];accepted={}
 for mid,e in state['models'].items():
  if e['status']!='done':continue
  pf=proof_for(mid,e,reg)
  if pf is None:continue
  proof=js(pf);accepted[mid]=(pf,proof)
  if mid not in config['allowed_model_ids'] or not proof.get('passed') or proof.get('sample_count')!=200 or proof.get('historical_record_reuse') or proof.get('error_indices'):continue
  folder=Path(e['report_dir']);assert proof.get('report_dir',str(folder))==str(folder) and set(proof['report_sha256'])==FIVE
  changed=[n for n,h in proof['report_sha256'].items() if sha(folder/n)!=h]
  if changed:skips.append(dict(model_id=mid,reason='Current report bytes differ from selected acceptance proof',files=changed));continue
  line=js(folder/'operator_lineage.json').get('actual_execution',{});home=own_home(line)
  if home is None or home.is_symlink():continue
  groups,trace=sample_group_index(line);files=[];keep=[];unknown=[];missing=[];allidx=set()
  for name,h in proof.get('external_evidence_sha256',{}).items():
   p=Path(name)
   if p.suffix!='.npz' or not p.is_relative_to(home):continue
   indices=indices_for(p.relative_to(home),groups)
   if indices is None:unknown.append(name);continue
   assert indices and indices.issubset(set(range(200)));allidx.update(indices)
   if not p.exists():missing.append(name);continue
   assert isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h)
   row=dict(path=name,SHA256=h,bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512,model_id=mid,home=str(home),logical_indices=sorted(indices),relative=str(p.relative_to(home)),proof=str(pf),proof_SHA256=sha(pf))
   if indices&KEEP or not removable_file(p,home):keep.append(row)
   else:files.append(row)
  if missing:
   skips.append(dict(model_id=mid,reason='Conservative scope exclusion: original series already has absent matrices',missing_count=len(missing)));continue
  if allidx!=set(range(200)):
   skips.append(dict(model_id=mid,reason='Full original logical200 archive mapping not closed',observed=sorted(allidx)));continue
  if files:result.append(dict(model_id=mid,home=str(home),report_dir=str(folder),proof=str(pf),proof_SHA256=sha(pf),trace=str(trace) if trace else None,group_index_map={str(k):sorted(v) for k,v in groups.items()},remove=files,keep=keep,unknown_or_nonseries_NPZ_preserved=unknown,bytes=sum(v['bytes'] for v in files),allocated_bytes=sum(v['allocated_bytes'] for v in files)))
 return sorted(result,key=lambda x:x['allocated_bytes'],reverse=True),skips,accepted
def main():
 cfgpath=R/'configs/retention/verified179_accepted_nonexample_matrices_for_Prithiv_user_request_20261117.json';cfg=js(cfgpath);assert cfg['preserve_all_five_all_records_sources_configs_snapshots_logs'] and cfg['only_current_accepted_complete200'] and cfg['keep_original_logical_indices']==sorted(KEEP)
 assert sha(POL)=='66857aae2ab0836a7639a09d7d09a32db01808b3184fb052f027da4ab6b88ac7';core_unchanged()
 if sys.argv[1] in ['discover','rediscover']:
  idle();scopes,skips,_=discover(cfg)
  if sys.argv[1]=='rediscover':
   assert sha(H/'discovery.json')==sys.argv[2] and not (H/'plan.json').exists();old=H/('discovery.original.'+sys.argv[2]+'.json');assert not old.exists();old.write_bytes((H/'discovery.json').read_bytes())
  else:assert not H.exists();H.mkdir();save('user_authorization.json',cfg['latest_actual_user_authorization'])
  save('discovery.json',dict(UTC=now(),config=str(cfgpath),config_SHA256=sha(cfgpath),scopes=scopes,skipped=skips,no_deletion=True));print(json.dumps(dict(discovery_SHA256=sha(H/'discovery.json'),total_candidate_bytes=sum(x['bytes'] for x in scopes),total_allocated_bytes=sum(x['allocated_bytes'] for x in scopes),candidate_count=sum(len(x['remove']) for x in scopes),scopes=[dict(model_id=x['model_id'],remove_files=len(x['remove']),remove_bytes=x['bytes'],keep_files=len(x['keep']),home=x['home']) for x in scopes],no_deletion=True),ensure_ascii=False));return
 if sys.argv[1]=='plan':
  idle();assert sha(H/'discovery.json')==sys.argv[2];scopes,skips,accepted=discover(cfg);assert [x['model_id'] for x in scopes]==[x['model_id'] for x in js(H/'discovery.json')['scopes']]
  candidates={x['path']:x for s in scopes for x in s['remove']};foreign=set();metadata=set();unresolved=[]
  # Every other accepted model and every static numerical-evidence map retains its exact references.
  for mid,(pf,proof) in accepted.items():
   metadata.add(pf)
   for name in paths_in(proof):
    if name in candidates and candidates[name]['model_id']!=mid:foreign.add(name)
   for key,value in proof.items():
    if key!='external_evidence_sha256' and key.endswith(('evidence_sha256','evidence_SHA256')):
     foreign.update(n for n in paths_in(value) if n in candidates)
  reg=js(D/'registry.json')
  for mid,sp in reg.items():
   if not isinstance(sp,dict):continue
   files={R/n for n in sp.get('source_sha256',{}) if not Path(n).is_absolute() and Path(n).suffix in ['.json','.yaml','.yml']}
   for name in paths_in(sp):
    p=Path(name)
    if p.is_file() and p.suffix=='.json' and p.stat().st_size<=8*1024**2:files.add(p)
   for p in files:
    if not p.is_file():continue
    metadata.add(p)
    if p.suffix!='.json':continue
    raw=p.read_bytes()
    try:content=json.loads(raw)
    except Exception:continue
    foreign.update(n for n in paths_in(content) if n in candidates)
  # Protect any static source/approval path used by any existing private job, including pending jobs.
  for p in B.glob('*/job.json'):
   metadata.add(p);j=js(p);mid=j.get('model_id')
   for name in paths_in(j):
    if name in candidates:foreign.add(name)
  # A file currently open by any process is always preserved.
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:
    for fd in (proc/'fd').iterdir():
     try:name=os.readlink(fd)
     except OSError:continue
     if name in candidates:foreign.add(name)
   except OSError:continue
  for name in foreign:candidates.pop(name,None)
  assert candidates and len(candidates)<=cfg['max_remove_files'] and sum(x['bytes'] for x in candidates.values())<=cfg['max_remove_bytes']
  protected=set(metadata);critical=[D/n for n in ['state.json','registry.json','PAUSE','POLICY.json','A100_ALIGNMENT_LOCAL_WORKBOOK.json','A100_ALIGNMENT_PUBLICATION_POLICY.json','A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json']]+[Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')];protected.update(critical)
  state=js(D/'state.json')
  for e in state['models'].values():
   if e['status']=='done':protected.update(Path(e['report_dir'])/n for n in FIVE)
  for sc in scopes:protected.update(p for p in Path(sc['home']).rglob('*') if p.is_file() and str(p) not in candidates)
  for root in [R/'scripts',R/'configs',R/'universal_eval']:
   protected.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.mjs','.ir'])
  assert not set(candidates)&{str(p) for p in protected};oldledger=LED.read_bytes();(H/'ledger_before.original').write_bytes(oldledger)
  # Full immutable-byte checks are performed once, not another CPU200/model replay.
  before={}
  for i,p in enumerate(sorted(protected)):
   before[str(p)]=sha(p)
   if i%200==0:save('progress.json',dict(UTC=now(),phase='hashing_protected_full_reports_sources_logs_examples',checked=i+1,total=len(protected)))
  plan=dict(UTC=now(),discovery_SHA256=sys.argv[2],config_SHA256=sha(cfgpath),authorization_SHA256=sha(H/'user_authorization.json'),policy_SHA256=sha(POL),ledger_before_SHA256=sha(LED),canonical_before_SHA256={str(p):sha(p) for p in critical},scopes=scopes,remove=list(candidates.values()),foreign_or_open_file_paths_preserved=sorted(foreign),protected_SHA256=before,protected_count=len(before),remove_count=len(candidates),remove_bytes=sum(x['bytes'] for x in candidates.values()),remove_allocated_bytes=sum(x['allocated_bytes'] for x in candidates.values()),free_before_bytes=shutil.disk_usage(R).free,all_reports_records_sources_configs_snapshots_logs_and_needed_examples_preserved=True,no_failed_incomplete_or_small_sample_run_selected=True,all_original_proofs_and_reports_preserved_not_rewritten=True,no_unresolved_scope_or_path_accepted=True)
  save('plan.json',plan);print(json.dumps(dict(plan_SHA256=sha(H/'plan.json'),remove_count=plan['remove_count'],remove_bytes=plan['remove_bytes'],remove_allocated_bytes=plan['remove_allocated_bytes'],free_before_bytes=plan['free_before_bytes'],protected_count=len(before),foreign_or_open_files_preserved=len(foreign),scopes=[dict(model_id=s['model_id'],candidate_count=sum(x['model_id']==s['model_id'] for x in candidates.values())) for s in scopes]),ensure_ascii=False));return
 assert sys.argv[1]=='apply';idle();assert sha(H/'plan.json')==sys.argv[2];p=js(H/'plan.json');assert sha(LED)==p['ledger_before_SHA256'] and sha(cfgpath)==p['config_SHA256'] and sha(POL)==p['policy_SHA256']
 with (D/'runner.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kernel_lock_owned(os.getpid(),D/'runner.lock');core_unchanged()
  for i,(name,h) in enumerate(p['protected_SHA256'].items()):
   assert sha(name)==h,('Protected evidence changed before removal',name)
   if i%200==0:save('progress.json',dict(UTC=now(),phase='rechecking_protected_full_reports_sources_logs_examples',checked=i+1,total=p['protected_count']))
  for i,x in enumerate(p['remove']):
   f=Path(x['path']);home=Path(x['home']);assert home.is_relative_to(S) and removable_file(f,home) and f.stat().st_size==x['bytes'] and sha(f)==x['SHA256']
   assert not set(x['logical_indices'])&KEEP and str(f) not in p['protected_SHA256'];assert sha(x['proof'])==x['proof_SHA256']
   if i%100==0:save('progress.json',dict(UTC=now(),phase='checking_every_exact_candidate_original_SHA_before_any_delete',checked=i+1,total=p['remove_count']))
  assert sha(LED)==p['ledger_before_SHA256'];manifest=H/'removed_exact_NPZ_paths_original_SHA256.jsonl';assert not manifest.exists();removed=[];failure=None
  try:
   with manifest.open('x',encoding='utf8') as out:
    for i,x in enumerate(p['remove']):
     f=Path(x['path']);assert removable_file(f,Path(x['home'])) and f.stat().st_size==x['bytes'];f.unlink();removed.append(x);out.write(json.dumps(x,ensure_ascii=False)+'\n');out.flush()
     if i%100==0:save('progress.json',dict(UTC=now(),phase='removing_only_preverified_exact_nonexample_NPZ',removed=i+1,total=p['remove_count']))
  except BaseException:failure=traceback.format_exc()
  bad=[name for name,h in p['protected_SHA256'].items() if sha(name)!=h];ok=not failure and not bad and len(removed)==p['remove_count']
  actual=dict(UTC=now(),actual_returncode=0 if ok else 1,plan_SHA256=sys.argv[2],removed_files=len(removed),removed_bytes=sum(x['bytes'] for x in removed),removed_manifest=str(manifest),removed_manifest_SHA256=sha(manifest),protected_files=p['protected_count'],all_protected_SHA256_unchanged=not bad,post_protection_errors=bad,error=failure,free_before_bytes=p['free_before_bytes'],free_after_bytes=shutil.disk_usage(R).free,all_five_reports_all_records_source_config_snapshots_logs_and_first_middle_last_groups_retained=True,no_failed_incomplete_or_small_sample_arrays_deleted=True,all_original_proofs_unchanged_but_exact_pruned_matrices_no_longer_available=True,canonical_score_state_registry_and_workbooks_unchanged=True,no_model_or_CPU200_rerun=True);save('actual_completion.json',actual)
  ledger=js(LED);assert sha(LED)==p['ledger_before_SHA256'];ledger['events'].append(dict(UTC=actual['UTC'],authorization=str(H/'user_authorization.json'),authorization_SHA256=p['authorization_SHA256'],receipt=str(H/'actual_completion.json'),receipt_SHA256=sha(H/'actual_completion.json'),manifest=str(manifest),manifest_SHA256=sha(manifest),removed_files=len(removed),removed_bytes=actual['removed_bytes'],all_original_records_kept=True,intentional_nonexample_matrix_unavailability=True,actual_returncode=actual['actual_returncode']));tmp=LED.with_name(LED.name+'.user_cleanup.tmp');assert not tmp.exists();tmp.write_bytes(json.dumps(ledger,ensure_ascii=False,indent=2).encode());os.replace(tmp,LED);print(json.dumps(dict(actual=actual,actual_SHA256=sha(H/'actual_completion.json'),ledger_SHA256=sha(LED)),ensure_ascii=False));assert ok
if __name__=='__main__':main()
