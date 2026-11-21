"""Freeze explicitly collected original code, review full diffs and consumers."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,base64,collections,difflib,hashlib,importlib.util,json,subprocess
V=Path(__file__).resolve().parent;repo=Path('D:/code-lv/910b');source=Path('D:/code-lv/.tmp/cd22s1')
assert source.resolve().parent==Path('D:/code-lv/.tmp').resolve() and not source.exists();source.mkdir()
packet_path=V/'R_daily22_verified184_explicit_primary_sources_v2_LlavaMini_cutoff_20261122.json';packet=json.loads(packet_path.read_bytes())
policy=json.loads((repo/'configs/publication/code_only.json').read_bytes());assert policy==packet['policy_before'] and packet['policy_before_SHA256']=='c8a705028535eace9acb88fb60fbe45bf6afa3b974a9a3566201d44698786f05'
assert packet['opaque_payload_scripts_excluded_with_exact_reason']==[] and packet['unconsumed_JSON_excluded']==[]
excluded={};raws={}
for rec in packet['records']:
    raw=base64.b64decode(rec['base64'],validate=True);assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['SHA256'];name=rec['path']
    if name.endswith('.json') and name not in policy['files'] and 'input_records' in json.loads(raw):
        assert name in ['configs/protocols/meta_mllama_own_original_BF16_SDPA_2048_serial_v1_20261120.json','configs/protocols/meta_mllama_completed_genuine3_shaped_storage_LZMA_v3_approved_first200_serial_20261120.json'] and len(json.loads(raw)['input_records'])==200
        excluded[name]=dict(SHA256=rec['SHA256'],reason='Contains 200 actual task prompts/references/token arrays; full original remains on R and evidence. Only exact code-only Git exclusion; own source generators and independent YAML retained.');continue
    assert name not in raws;raws[name]=raw;path=source/name;assert path.resolve().is_relative_to(source.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
assert len(excluded)==2 and 'run.py' in raws
spec=importlib.util.spec_from_file_location('reviewed_publisher_day22',source/'scripts/publish_code_daily.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
rows=[];diff=[];new_json={};javascript=[]
for rec in packet['records']:
    name=rec['path']
    if name in excluded:continue
    raw=raws[name];pub.validate_path(name,rec['entry']['role']);pub.validate_content(name,raw,rec['entry']);before=(repo/name).read_bytes() if (repo/name).is_file() else None
    if name not in policy['files']:policy['files'][name]=dict(rec['entry'],initial_SHA256=rec['SHA256'])
    if name.endswith('.json') and name not in packet['policy_before']['files']:
        consumers=packet['new_JSON_consumers'][name];assert consumers and all(c in raws for c in consumers);new_json[name]=consumers
    if name.endswith('.mjs'):
        result=subprocess.run(['C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe','--check',str(source/name)],capture_output=True);assert result.returncode==0,(name,result.stderr.decode(errors='replace'));javascript.append(dict(path=name,SHA256=rec['SHA256'],actual_node_syntax_returncode=0))
    action='unchanged' if before==raw else 'add' if before is None else 'update';row=dict(path=name,SHA256=rec['SHA256'],bytes=len(raw),action=action,role=policy['files'][name]['role'])
    if name.endswith('.py'):row['top_level_definitions']=[n.name for n in ast.parse(raw.decode('utf-8-sig'),filename=name).body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
    if action!='unchanged':diff.extend(difflib.unified_diff((before or b'').decode('utf-8-sig').splitlines(keepends=True),raw.decode('utf-8-sig').splitlines(keepends=True),fromfile='before/'+name,tofile='after/'+name))
    rows.append(row)
for name in ['README.md','.gitignore','.gitattributes']:(source/name).write_bytes((repo/name).read_bytes())
new_policy=(json.dumps(policy,ensure_ascii=False,indent=2)+'\n').encode();path=source/'configs/publication/code_only.json';path.write_bytes(new_policy);pub.load_policy(path)
assert not(set(packet['policy_before']['files'])-set(policy['files']))
diff.extend(difflib.unified_diff((repo/'configs/publication/code_only.json').read_text().splitlines(keepends=True),new_policy.decode().splitlines(keepends=True),fromfile='before/configs/publication/code_only.json',tofile='after/configs/publication/code_only.json'))
diff_path=V/'daily_20261122_full_source_diff.patch';diff_path.write_text(''.join(diff),encoding='utf8',newline='\n')
review=dict(passed=True,source_root=str(source),source_collection_SHA256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),prior_explicit_paths=len(packet['policy_before']['files']),explicit_paths=len(policy['files']),records=rows,new_JSON_actual_consumers=new_json,exact_task_data_exclusions=excluded,javascript_actual_syntax_checks=javascript,action_counts=dict(collections.Counter(r['action'] for r in rows)),all_path_content_secret_python_AST_payload_and_operator_IR_boundaries_passed=True,all_local_maintenance_preserved=True,no_source_repo_or_R_model_mutation=True,full_diff=str(diff_path),full_diff_SHA256=hashlib.sha256(diff_path.read_bytes()).hexdigest(),policy_before_SHA256=packet['policy_before_SHA256'],policy_after_SHA256=hashlib.sha256(new_policy).hexdigest())
(V/'daily_20261122_source_content_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in review.items() if k not in ['records','new_JSON_actual_consumers','javascript_actual_syntax_checks']},ensure_ascii=False))
print('CHANGED_EXISTING',json.dumps([r for r in rows if r['action']=='update'],ensure_ascii=False))
