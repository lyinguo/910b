import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,base64,ast
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';C=D/'code_publication_source_layout_20261114'
sys.path.insert(0,str(R/'scripts'))
import publish_code_daily as pub
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
policy=js(R/'configs/publication/code_only.json');assert policy['repository']=='https://github.com/lyinguo/910b.git' and policy['branch']=='codex/910b-source-layout-20261114'
local_only={'.gitattributes','.gitignore','README.md','configs/publication/code_only.json'}
paths=set(policy['files'])-local_only;reasons={p:set(['Existing reviewed publication baseline']) for p in paths};records=[];outside=[]
def add(p,why):
 if p.startswith(str(R)+'/'):p=p[len(str(R))+1:]
 if not p.startswith(('scripts/','universal_eval/','configs/','tests/')) or not (R/p).is_file():
  outside.append(dict(path=p,reason=why));return
 if Path(p).suffix not in ['.py','.yaml','.yml','.json','.md','.ir','.toml','.txt','.mjs']:return
 paths.add(p);reasons.setdefault(p,set()).add(why)
pending_paths=[C/'pending_after_daily_20261117_verified180_Prithiv_Ovis_Phi35_official_full_closure_all_primary_v22.json']
assert sha(pending_paths[0])=='cd0df225923793ec9b42da1339edc9cbfd9d80a4a374dcf0b273df6fa428157d'
for p in pending_paths:
 v=js(p);why='Explicit primary implementation inventory '+p.name
 for key in ['source_records','entries','explicit_source_manifest','all_current_primary_source_and_true_consumer_configs']:
  for rec in v.get(key,[]):
   if isinstance(rec,dict) and isinstance(rec.get('path') or rec.get('relative'),str):add(rec.get('path') or rec['relative'],why)
 for key in ['reviewed_new_code_and_configs']:
  for rel in v.get(key,{}):add(rel,why)
 for rel in v.get('useful_paths_for_next_daily_review',[]):add(rel,why)
for rel in ['tests/test_code_publication_javascript.py','scripts/code_publication_windows/read_verified180_complete_required_handoff_latest_actual_0020_20261118_remote.py','scripts/code_publication_windows/read_Phi35_current_original_interface_and_own_meta_source_no_weights_reread_0020_20261118_remote.py']:
 add(rel,'Day18 actual publisher boundary test or explicit own source/state reader')
# These sources were deployed after the last frozen pending inventory.
for rel in ['scripts/publish_RedHatGemma4E2B_verified175_single_lock_dual_transaction_v1_exact_original_reference_20261116.py','scripts/run_RedHatGemma4E2B_verified175_dual_publication_actual_completion_v1_exact_LF_20261116.py','scripts/tests/test_RedHatAI_gemma_4_E2B_it_verified175_original_writer_isolated_20261116.py','scripts/prepare_gemma4unified12bit_own_source_assets_META_no_learned_CPU_20261116.py','scripts/prepare_gemma4unified12bit_own_frontend200_and_exact_META_names_20261116.py','universal_eval/model/reference_sources/gemma4_unified_owned_checkpoint_layout.py']:
 add(rel,'Later deployed own Unified preflight or fully completed verified175 publication source')
latest=js(D/'incremental_monitor/latest.json')
for name in latest.get('new_or_modified_source_config_paths',[]):add(name,'Actual frozen incremental monitor source delta')
for pat in ['scripts/*minicpmo26*.py','universal_eval/model/minicpmo26_own_original_20261116/*.py','universal_eval/model/adapters/minicpmo26_own_*.py','configs/MiniCPMO/*MiniCPM-o-2_6-own-original*20261116*.yaml','configs/protocols/minicpmo26_own_original*.json','scripts/*agentcpmgui*.py','universal_eval/model/adapters/agentcpmgui_own_*.py','universal_eval/model/agentcpmgui_own_original_20261116/*.py','configs/MiniCPMV/*AgentCPM-GUI-own-original*20261116*.yaml','configs/protocols/agentcpmgui_own_original*.json']:
 for p in R.glob(pat):add(str(p.relative_to(R)),'New own AgentCPMGUI workflow and exact dependency')
dynamic_config_consumers={
 'configs/queues/Eagle2_1B_Eagle2_2B_available_own_20261115.json':['scripts/continue_eagle2_2b_after_eagle2_1b_original_release_v1.py'],
 'configs/queues/available_own_InternVL2_Docopilot_20261115.json':['scripts/continue_available_own_models_immediately_20261115.py']}
for rel,consumers in dynamic_config_consumers.items():
 if (R/rel).is_file():
  add(rel,'Queue configuration passed to its real controller')
  for consumer in consumers:add(consumer,'Actual dynamic queue configuration consumer')
# Preserve configuration-backed root namespace modules reached from actual local imports.
while True:
 before=len(paths)
 for rel in list(paths):
  if not rel.endswith('.py'):continue
  try:tree=ast.parse((R/rel).read_bytes())
  except (SyntaxError,UnicodeError):raise
  package=list(Path(rel).with_suffix('').parts[:-1])
  for n in ast.walk(tree):
   modules=[]
   if isinstance(n,ast.Import):modules=[a.name for a in n.names]
   elif isinstance(n,ast.ImportFrom):
    parts=(package[:len(package)-n.level+1]+(n.module.split('.') if n.module else [])) if n.level else (n.module.split('.') if n.module else [])
    modules=['.'.join(parts)]
    if not n.module:modules+=['.'.join(parts+[a.name]) for a in n.names]
   for module in modules:
    if module.startswith(('universal_eval.','scripts.','tests.')):
     q=module.replace('.','/')
     for pp in [q+'.py',q+'/__init__.py']:
      if (R/pp).is_file():add(pp,'Actual local Python import from '+rel)
 if len(paths)==before:break
texts={rel:(R/rel).read_text(encoding='utf-8-sig') for rel in paths if Path(rel).suffix in ['.py','.yaml','.yml']}
json_consumers={};unconsumed=[];opaque_payloads=[]
for rel in sorted(paths):
 p=R/rel;raw=p.read_bytes()
 role=policy['files'].get(rel,{}).get('role')
 if role is None:role='runtime_config' if p.suffix in ['.json','.yaml','.yml'] else 'runtime_code' if rel.startswith('universal_eval/') or '/continue_' in rel else 'verification_code'
 pub.validate_path(rel,role)
 entry=dict(role=role,reason='; '.join(sorted(reasons.get(rel,{'Explicit consumed source'}))))
 if p.suffix=='.mjs':
  entry['reviewed_javascript_source_SHA256']=[sha(p)]
 if p.suffix=='.ir':
  text=raw.decode();assert text.lstrip().startswith('graph(')
  import re
  assert not any('prim::Constant' in l and re.search(r'Tensor|BFloat|Half\(|value=\{',l) for l in text.splitlines())
  entry['weight_free_operator_graph_SHA256']=[sha(p)]
 if p.suffix=='.json' and rel not in policy['files']:
  consumers=[n for n,text in texts.items() if rel in text or p.name in text]+dynamic_config_consumers.get(rel,[])
  if not consumers:
   unconsumed.append(rel);continue
  json_consumers[rel]=consumers;entry['reason']='Runtime configuration consumed by '+', '.join(consumers[:4])
 try:pub.validate_content(rel,raw,entry)
 except ValueError as error:
  if not ('Embedded payload requires separate review:' in str(error) or ('Binary/oversized content:' in str(error) and rel.endswith('_remote.py'))):raise
  assert rel not in policy['files'],'An existing published runtime must never be silently excluded'
  assert not any(v.get('verifier')==rel for v in js(D/'registry.json').values()),'A live model verifier cannot be omitted'
  opaque_payloads.append(dict(path=rel,SHA256=sha(p),bytes=len(raw),reason='Historical generated deployment/preparation payload requires source extraction review; original remains preserved, not included in this code-only commit',large_literal_lengths=[len(n.value) for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Constant) and isinstance(n.value,(str,bytes)) and len(n.value)>32768]))
  continue
 assert not p.is_symlink() and len(raw)<2*1024**2
 records.append(dict(path=rel,SHA256=sha(p),bytes=len(raw),entry=entry,base64=base64.b64encode(raw).decode()))
result=dict(UTC=datetime.now(timezone.utc).isoformat(),policy_before_SHA256=sha(R/'configs/publication/code_only.json'),policy_before=policy,records=records,record_count=len(records),new_JSON_consumers=json_consumers,unconsumed_JSON_excluded=unconsumed,opaque_payload_scripts_excluded_with_exact_reason=opaque_payloads,outside_primary_source_references_excluded=outside,pending_manifests=[dict(path=str(p),SHA256=sha(p)) for p in pending_paths],no_weights_data_arrays_reports_logs_or_credentials=True,no_R_canonical_or_active_job_source_change=True,local_maintenance_files_to_retain=sorted(local_only))
H=C/'daily_20261118_explicit_primary_source_review_v1';H.mkdir(exist_ok=False)
(H/'collection_manifest.json').write_text(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

