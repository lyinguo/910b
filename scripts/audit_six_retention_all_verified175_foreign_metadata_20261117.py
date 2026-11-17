import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'authorized_six_complete_verified_runs_nonexample_retention_v3_shared_examples_protected_for_CoderE4_capacity_20261117';PLAN_SHA=sys.argv[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def strings(v):
 if isinstance(v,str):yield v
 elif isinstance(v,dict):
  for k,x in v.items():yield str(k);yield from strings(x)
 elif isinstance(v,list):
  for x in v:yield from strings(x)
plan=js(H/'plan.json');assert sha(H/'plan.json')==PLAN_SHA;owned={x['path']:x['model_id'] for x in plan['remove']};homeowners={s['home']:s['model_id'] for s in plan['scopes']};state=js(D/'state.json');reg=js(D/'registry.json');matches=[];docs=[];seen=set()
for mid,e in state['models'].items():
 if e.get('status')!='done':continue
 paths=[Path(e['report_dir'])/n for n in ['eval_report.json','operator_lineage.json']];rr=reg.get(mid,{})
 if rr.get('accepted_formal200_proof'):paths.append(Path(rr['accepted_formal200_proof']))
 if rr.get('reuse',{}).get('proof'):paths.append(Path(rr['reuse']['proof']))
 if e.get('active_config'):paths.append(Path(e['active_config']))
 if e.get('attempts'):
  for a in reversed(e['attempts']):
   name=a.get('independent_verification')
   if name:paths.append(Path(name));break
 todo=[(p,0) for p in paths if p.is_file()]
 while todo:
  f,depth=todo.pop();key=(mid,str(f))
  if key in seen:continue
  seen.add(key);text=f.read_text();docs.append(dict(model_id=mid,path=str(f),SHA256=sha(f)))
  try:v=json.loads(text)
  except Exception:
   import yaml
   v=yaml.safe_load(text)
  for s in strings(v):
   if s in owned and owned[s]!=mid:matches.append(dict(model_id=mid,document=str(f),candidate=s,owner=owned[s],kind='exact_foreign_candidate'))
   for home,owner in homeowners.items():
    if owner==mid or home not in s:continue
    p=Path(s)
    try:existing_absolute_file=len(s)<4096 and p.is_absolute() and p.is_file()
    except OSError:existing_absolute_file=False
    if existing_absolute_file and str(p) in plan['protected_SHA256']:
     if p.suffix in ['.json','.jsonl'] and depth<2:
      if p.suffix=='.json':todo.append((p,depth+1))
      else:
       for line in p.read_text().splitlines():
        for name in strings(json.loads(line)):
         if name in owned and owned[name]!=mid:matches.append(dict(model_id=mid,document=str(p),candidate=name,owner=owned[name],kind='foreign_protected_manifest_references_candidate'))
    elif s not in owned:matches.append(dict(model_id=mid,document=str(f),home=home,reference=s,kind='unresolved_foreign_owned_home_reference'))
assert sha(H/'plan.json')==PLAN_SHA
result=dict(passed=not matches,UTC=datetime.now(timezone.utc).isoformat(),plan_SHA256=PLAN_SHA,current_completed_models_checked=sum(e.get('status')=='done' for e in state['models'].values()),active_report_lineage_config_and_registered_or_legacy_proof_metadata_checked=docs,foreign_candidate_or_unresolved_home_references=matches,no_matrix_read_reverified_or_deleted=True)
(H/'all_current_foreign_evidence_metadata_review.json').write_bytes(json.dumps(result,ensure_ascii=False,indent=2).encode());print(json.dumps(dict(passed=result['passed'],plan_SHA256=PLAN_SHA,models_checked=result['current_completed_models_checked'],metadata_documents=len(docs),matches=matches,review_SHA256=sha(H/'all_current_foreign_evidence_metadata_review.json')),ensure_ascii=False));assert not matches
