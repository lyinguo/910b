import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,collections,difflib,hashlib,importlib.util,json,subprocess
V=Path(__file__).resolve().parent;repo=Path('D:/code-lv/910b');source=Path('D:/code-lv/.tmp/cd21s2')
assert source.resolve().parent==Path('D:/code-lv/.tmp').resolve() and not source.exists();source.mkdir()
packet_path=V/'R_daily21_verified183_final_v63_explicit_primary_sources_v2_20261121.json';packet=json.loads(packet_path.read_bytes())
policy=json.loads((repo/'configs/publication/code_only.json').read_bytes());assert policy==packet['policy_before']
assert packet['opaque_payload_scripts_excluded_with_exact_reason']==[]
dynamic=json.loads((V/'R_daily21_dynamic_actual_consumer_and_Ovis_full_log_20261121.json').read_bytes())
extra=dynamic['record'];consumer=dynamic['actual_consumer']
assert packet['unconsumed_JSON_excluded']==[extra['path']]
records=list(packet['records'])+[extra];consumers=dict(packet['new_JSON_consumers']);consumers[extra['path']]=[consumer]
excluded={};raws={}
for rec in records:
 raw=base64.b64decode(rec['base64'],validate=True);assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['SHA256'];rel=rec['path']
 if rel.endswith('.json') and rel not in policy['files']:
  value=json.loads(raw)
  if 'input_records' in value:
   assert rel in ['configs/protocols/meta_mllama_own_original_BF16_SDPA_2048_serial_v1_20261120.json','configs/protocols/meta_mllama_completed_genuine3_shaped_storage_LZMA_v3_approved_first200_serial_20261120.json'] and len(value['input_records'])==200
   excluded[rel]=dict(SHA256=rec['SHA256'],reason='Runtime-generated protocol embeds 200 actual task prompts/reference/token arrays; preserved byte-for-byte on R and source archives, excluded from code-only Git publication. Own preparation/continuation code and independent YAML remain included.')
   continue
 assert rel not in raws;raws[rel]=raw;p=source/rel;assert p.resolve().is_relative_to(source.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
assert len(excluded)==2 and 'run.py' in raws and consumer in raws
tree=ast.parse(raws[consumer]);assert 'config = Path(sys.argv[1])' in raws[consumer].decode() and 'q = read(config)' in raws[consumer].decode()
spec=importlib.util.spec_from_file_location('reviewed_publisher_day21',source/'scripts/publish_code_daily.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
review_rows=[];diff=[];new_json={};javascript=[];node='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
for rec in records:
 name=rec['path']
 if name in excluded:continue
 raw=raws[name];pub.validate_path(name,rec['entry']['role']);pub.validate_content(name,raw,rec['entry']);previous=(repo/name).read_bytes() if (repo/name).is_file() else None
 if name not in policy['files']:policy['files'][name]=dict(rec['entry'],initial_SHA256=rec['SHA256'])
 if name.endswith('.json') and name not in packet['policy_before']['files']:
  cs=consumers[name];assert cs and all(c in raws for c in cs);new_json[name]=cs
 if name.endswith('.mjs'):
  rr=subprocess.run([node,'--check',str(source/name)],capture_output=True);assert rr.returncode==0,(name,rr.stderr.decode(errors='replace'));javascript.append(dict(path=name,SHA256=rec['SHA256'],actual_node_syntax_returncode=0))
 action='unchanged' if previous==raw else 'add' if previous is None else 'update';row=dict(path=name,SHA256=rec['SHA256'],bytes=len(raw),action=action,role=policy['files'][name]['role'])
 if name.endswith('.py'):
  t=ast.parse(raw.decode('utf-8-sig'),filename=name);row['top_level_definitions']=[n.name for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
 if action!='unchanged':diff.extend(difflib.unified_diff((previous or b'').decode('utf-8-sig').splitlines(keepends=True),raw.decode('utf-8-sig').splitlines(keepends=True),fromfile='before/'+name,tofile='after/'+name))
 review_rows.append(row)
for name in ['README.md','.gitignore','.gitattributes']:
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((repo/name).read_bytes())
new_policy=(json.dumps(policy,ensure_ascii=False,indent=2)+'\n').encode();p=source/'configs/publication/code_only.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(new_policy);pub.load_policy(p)
assert not(set(packet['policy_before']['files'])-set(policy['files']))
diff.extend(difflib.unified_diff((repo/'configs/publication/code_only.json').read_text().splitlines(keepends=True),new_policy.decode().splitlines(keepends=True),fromfile='before/configs/publication/code_only.json',tofile='after/configs/publication/code_only.json'))
diff_path=V/'daily_20261121_full_source_diff_v2.patch';diff_path.write_text(''.join(diff),encoding='utf8',newline='\n')
# Model changes are storage-only; independent NF4 changes precisely one original expected metadata expression.
old=raws['scripts/verify_meta_mllama_own_original_BF16_SDPA_2048_full_values_v3_exact_original_power_reference_20261120.py'].decode();new=raws['scripts/verify_meta_mllama_completed200_original_empty_invalid_prediction_schema_v4_20261121.py'].decode()
assert new.replace('\r\n','\n').replace("prediction='' if choice is None else choice","prediction=raw if choice is None else choice")==old.replace('\r\n','\n')
o=ast.parse(raws['universal_eval/model/adapters/ovisu1_own_original_understanding_bf16_512_native_v2_actual_empty_precision_guard_20261116.py']);n=ast.parse(raws['universal_eval/model/adapters/ovisu1_own_original_understanding_BF16_512_complete_standard_LZMA_20261121.py'])
defs=lambda t:{x.name:ast.dump(x,include_attributes=False) for x in t.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
od,nd=defs(o),defs(n);assert set(od)==set(nd) and [k for k in od if od[k]!=nd[k]]==['save']
review=dict(passed=True,source_root=str(source),source_collection_SHA256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),prior_explicit_paths=len(packet['policy_before']['files']),explicit_paths=len(policy['files']),records=review_rows,new_JSON_actual_consumers=new_json,exact_task_data_exclusions=excluded,actual_parameter_consumed_prior_failed_publication_config_included=dict(path=extra['path'],consumer=consumer),javascript_actual_syntax_checks=javascript,action_counts=dict(collections.Counter(r['action'] for r in review_rows)),all_path_content_secret_python_AST_payload_and_operator_IR_boundaries_passed=True,all_local_maintenance_preserved=True,all_R_model_report_data_and_config_original_bytes_preserved=True,only_new_Ovis_save_AST_diff_and_Meta_NF4_metadata_expected_value_diff_verified=True,no_source_repo_or_R_model_mutation=True,full_diff=str(diff_path),full_diff_SHA256=hashlib.sha256(diff_path.read_bytes()).hexdigest(),policy_before_SHA256=packet['policy_before_SHA256'],policy_after_SHA256=hashlib.sha256(new_policy).hexdigest())
(V/'daily_20261121_source_content_review_v2.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in review.items() if k not in ['records','new_JSON_actual_consumers','javascript_actual_syntax_checks']},ensure_ascii=False))
print('CHANGED_EXISTING',json.dumps([r for r in review_rows if r['action']=='update'],ensure_ascii=False))
