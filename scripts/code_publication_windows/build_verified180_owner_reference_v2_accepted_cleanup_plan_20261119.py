from pathlib import Path
import ast,base64,hashlib,json
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
base=(P/'retain_verified180_current_accepted_nonexample_user_serial_20261119.py').read_text()
base=base.replace('verified180_accepted_nonexample_user_serial_continue_20261119.json','verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json')
base=base.replace("H/'plan.json'","H/'plan_owner_reference_v2.json'").replace("H/'ledger_before.original'","H/'ledger_before_owner_reference_v2.original'")
needle="def sources_and_dependencies(reg,accepted,candidates,helpers):"
replacement="""def is_real_dependency_reference(owner,candidate_owner,metadata_path,current_acceptance_proof,current_five):
 # Own acceptance inventory and full reports are immutable records to retain;
 # listing their accepted arrays is not a second task dependency.
 # Every other metadata path/owner keeps the original conservative protection.
 return not (owner==candidate_owner and str(metadata_path) in {str(current_acceptance_proof),*map(str,current_five)})

def sources_and_dependencies(reg,accepted,candidates,helpers):"""
assert base.count(needle)==1;base=base.replace(needle,replacement)
needle="if n in candidates:mark(n,'current_registry_static_dependency:'+mid+':'+str(p))"
replacement="""if n in candidates:
     own=accepted.get(mid);own_proof=Path(own['proof']) if own else None
     own_five=[Path(own['report_dir'])/name for name in helpers.FIVE] if own else []
     if is_real_dependency_reference(mid,candidates[n]['model_id'],p,own_proof,own_five):
      mark(n,'current_registry_static_dependency:'+mid+':'+str(p))"""
assert base.count(needle)==1;base=base.replace(needle,replacement);ast.parse(base)
name='retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119.py';(P/name).write_text(base,encoding='utf8',newline='\n')
cfg=json.loads((P/'verified180_accepted_nonexample_user_serial_continue_20261119.json').read_bytes())
cfg.update(version=2,own_current_acceptance_inventory_and_standard_five_not_a_foreign_consumer=True,all_actual_foreign_metadata_static_configuration_private_jobs_and_open_FDs_protected=True,preserve_own_acceptance_report_bytes_and_record_exact_pruned_paths_in_ledger=True)
cfgname='verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json';raw=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode();(P/cfgname).write_bytes(raw)
# Pure dependency classification checks: no model, arrays, kernel or acceptance replay.
ns={};tree=ast.parse(base);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='is_real_dependency_reference');exec(compile(ast.Module(body=[f],type_ignores=[]),'<actual_dependency_classifier>','exec'),ns);test=ns['is_real_dependency_reference']
checks=[('same current own proof',False,test('A','A','/own/proof','/own/proof',['/own/report'])),('same own standard report',False,test('A','A','/own/report','/own/proof',['/own/report'])),('different owner still protects',True,test('B','A','/own/proof','/own/proof',['/own/report'])),('same owner real config protects',True,test('A','A','/own/runtime_config','/own/proof',['/own/report'])),('own previous proof remains protected',True,test('A','A','/old/proof','/own/proof',['/own/report'])),('unregistered source still protected',True,test('A','A','/other/proof',None,[])),('matching basename cannot impersonate own proof',True,test('A','A','/other/proof','/own/proof',['/own/report']))]
assert all(want==got for _,want,got in checks)
(P/'verified180_dependency_owner_classifier_seven_scope_checks_20261119.json').write_text(json.dumps(dict(passed=True,checks=[dict(name=n,expected=w,actual=g) for n,w,g in checks],model_forwards=0),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sources=[]
for n,r in [(name,'scripts/maintenance/'+name),(cfgname,'configs/retention/'+cfgname)]:
 b=(P/n).read_bytes();sources.append(dict(relative=r,SHA256=sha(b),base64=base64.b64encode(b).decode()))
data=dict(sources=sources,discovery_SHA256=cfg['discovery_SHA256'])
remote='DATA='+repr(data)+'\n'+'''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,hashlib,json,importlib,traceback
R=Path('/root/projects/comperror')
for x in DATA['sources']:
 p=R/x['relative'];b=base64.b64decode(x['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==x['SHA256'];assert p.resolve().is_relative_to(R)
 ast.parse(b) if p.suffix=='.py' else json.loads(b)
 if p.exists():assert p.read_bytes()==b
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
sys.path[:0]=[str(R),str(R/'scripts')]
m=importlib.import_module('maintenance.retain_verified180_current_accepted_nonexample_owner_reference_v2_user_serial_20261119')
try:result=m.plan(DATA['discovery_SHA256'])
except Exception:
 (m.H/'plan_owner_reference_v2_failure_preserved.txt').write_text(traceback.format_exc(),encoding='utf8');raise
print(json.dumps(result,ensure_ascii=False))
'''
ast.parse(remote);(P/'deploy_plan_verified180_owner_reference_v2_user_accepted_cleanup_20261119_remote.py').write_text(remote,encoding='utf8',newline='\n')
print(json.dumps(dict(sources=[{k:v for k,v in x.items() if k!='base64'} for x in sources],pure_scope_checks=7,model_forwards=0,only_dependency_classification_and_private_paths_changed=True),ensure_ascii=False))
