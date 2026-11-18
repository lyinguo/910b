"""Collect SHA-pinned primary code and real configuration consumers, never run artifacts."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,base64,hashlib,json

R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';C=D/'code_publication_source_layout_20261114'
sys.path[:0]=[str(R),str(R/'scripts')]
import publish_code_daily as pub

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def collect(pending,expected,home,extra):
 assert pending.resolve().is_relative_to(C.resolve()) and sha(pending)==expected
 assert home.resolve().is_relative_to(C.resolve()) and not home.exists()
 policy=js(R/'configs/publication/code_only.json')
 assert policy['repository']=='https://github.com/lyinguo/910b.git' and policy['branch']=='codex/910b-source-layout-20261114'
 keep_local={'.gitattributes','.gitignore','README.md','configs/publication/code_only.json'}
 paths=set(policy['files'])-keep_local;reasons={p:{'Existing reviewed publication baseline'} for p in paths};declared=js(pending)['all_current_primary_source_and_true_consumer_configs']
 def add(rel,reason):
  p=Path(rel)
  if p.is_absolute():rel=str(p.relative_to(R))
  q=R/rel
  assert (rel=='run.py' or rel.startswith(('scripts/','universal_eval/','configs/','tests/'))) and q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(R.resolve()),rel
  assert q.suffix.lower() in {'.py','.yaml','.yml','.json','.md','.ir','.toml','.txt','.mjs'},rel
  paths.add(rel);reasons.setdefault(rel,set()).add(reason)
 for v in declared:
  rel=v.get('relative') or str(Path(v['path']).relative_to(R));assert sha(R/rel)==v['SHA256'],rel
  add(rel,'Explicit SHA-pinned primary inventory '+pending.name)
 for rel in extra:add(rel,'Explicit current-day useful implementation or actual inspection/dependency')
 add(str(Path(__file__).resolve().relative_to(R)),'Actual primary source publication collector')
 while True:
  before=len(paths)
  for rel in list(paths):
   if not rel.endswith('.py'):continue
   tree=ast.parse((R/rel).read_bytes(),filename=rel);package=list(Path(rel).with_suffix('').parts[:-1])
   for n in ast.walk(tree):
    modules=[]
    if isinstance(n,ast.Import):modules=[a.name for a in n.names]
    elif isinstance(n,ast.ImportFrom):
     prefix=package[:len(package)-n.level+1] if n.level else []
     base=prefix+(n.module.split('.') if n.module else []);modules=['.'.join(base)]
     if n.module is None:modules+=['.'.join(base+[a.name]) for a in n.names]
    for module in modules:
     if not module:continue
     q=module.replace('.','/')
     for suffix in [q+'.py',q+'/__init__.py']:
      for candidate in [suffix,'scripts/'+suffix]:
       if candidate.startswith(('scripts/','universal_eval/','tests/')) and (R/candidate).is_file():add(candidate,'Actual local Python dependency of '+rel)
  if len(paths)==before:break
 texts={rel:(R/rel).read_text(encoding='utf-8-sig') for rel in paths if Path(rel).suffix.lower() in {'.py','.yaml','.yml'}}
 records=[];consumers={};unconsumed=[];opaque=[]
 for rel in sorted(paths):
  p=R/rel;raw=p.read_bytes();entry=dict(policy['files'].get(rel,{}))
  if not entry:
   role='runtime_config' if p.suffix.lower() in {'.json','.yaml','.yml'} else 'runtime_code' if rel.startswith('universal_eval/') or '/continue_' in rel else 'verification_code'
   entry=dict(role=role,reason='; '.join(sorted(reasons[rel])))
  pub.validate_path(rel,entry['role'])
  if p.suffix=='.mjs':entry['reviewed_javascript_source_SHA256']=sorted(set(entry.get('reviewed_javascript_source_SHA256',[])+[sha(p)]))
  if p.suffix=='.ir':entry['weight_free_operator_graph_SHA256']=sorted(set(entry.get('weight_free_operator_graph_SHA256',[])+[sha(p)]))
  if p.suffix=='.json' and rel not in policy['files']:
   actual=[n for n,t in texts.items() if rel in t or p.name in t]
   if not actual:unconsumed.append(rel);continue
   consumers[rel]=actual;entry['reason']='Runtime configuration actually consumed by '+', '.join(actual[:5])
  try:pub.validate_content(rel,raw,entry)
  except ValueError as err:
   if 'Embedded payload requires separate review:' not in str(err):raise
   opaque.append(dict(path=rel,SHA256=sha(p),bytes=len(raw),reason='Opaque deployment payload requires a source extraction review; never silently omit useful code'))
   continue
  if rel.endswith('.py'):ast.parse(raw,filename=rel)
  records.append(dict(path=rel,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),entry=entry,base64=base64.b64encode(raw).decode()))
 for v in declared:
  rel=v.get('relative') or str(Path(v['path']).relative_to(R));assert sha(R/rel)==v['SHA256'],'Concurrent source mutation: '+rel
 protected_job=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118/job.json'
 protected=js(protected_job)['protected_canonical_hashes']
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),policy_before=policy,policy_before_SHA256=sha(R/'configs/publication/code_only.json'),records=records,record_count=len(records),pending_manifests=[dict(path=str(pending),SHA256=expected,declared_source_count=len(declared))],new_JSON_consumers=consumers,unconsumed_JSON_excluded=unconsumed,opaque_payload_scripts_excluded_with_exact_reason=opaque,no_weights_data_arrays_reports_logs_or_credentials=True,local_maintenance_files_to_retain=sorted(keep_local),active_original_GLM_job_protected_paths=list(protected),collection_does_not_mutate_any_model_canonical_policy_or_R_source=True)
 home.mkdir()
 (home/'collection_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pending',type=Path,required=True);ap.add_argument('--pending-sha',required=True);ap.add_argument('--review-home',type=Path,required=True);ap.add_argument('--include',action='append',default=[]);a=ap.parse_args();collect(a.pending,a.pending_sha,a.review_home,a.include)
