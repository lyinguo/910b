"""Review exact plan, complete new code diff/consumers and outgoing history."""
from pathlib import Path
from datetime import datetime,timezone
import ast,collections,hashlib,importlib.util,json,os,subprocess
V=Path(__file__).resolve().parent;source=Path('D:/code-lv/.tmp/cd22s1');repo=Path('D:/code-lv/910b')
review=json.loads((V/'daily_20261122_source_content_review.json').read_bytes());local=json.loads((V/'daily_20261122_snapshot_tests_actual.json').read_bytes());remote_tests=json.loads((V/'R_daily22_actual_R_source_unit_checks_20261122.json').read_bytes());special=json.loads((V/'R_LlavaMini_original_private_cache_complete_observer_units_no_NN_20261122.json').read_bytes())
assert review['passed'] and local['passed'] and local['tests_run']==19 and remote_tests['passed'] and remote_tests['tests_run']==23 and special['passed'] and special['tests_run']==3
assert len(review['javascript_actual_syntax_checks'])==23 and {r['path'] for r in review['javascript_actual_syntax_checks']}=={r['path'] for r in review['records'] if r['path'].endswith('.mjs')}
assert hashlib.sha256(Path(review['full_diff']).read_bytes()).hexdigest()==review['full_diff_SHA256']
spec=importlib.util.spec_from_file_location('reviewed_publisher_day22',source/'scripts/publish_code_daily.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+env['PATH'];env.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.sslBackend',GIT_CONFIG_VALUE_0='openssl')
def git(*args):
    result=subprocess.run(['git','-c','credential.interactive=never',*args],cwd=repo,env=env,capture_output=True);assert result.returncode==0,result.stderr.decode(errors='replace')[-1800:];return result.stdout
assert git('status','--porcelain')==b'' and git('branch','--show-current').decode().strip()=='codex/910b-source-layout-20261114' and git('remote','get-url','origin').decode().strip()=='https://github.com/lyinguo/910b.git'
head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').decode().split()[0];assert head==remote=='189b3146a82adf0dc9e50f12691007ff31de6b6b'
assert 'Code-Day: 2026-11-22' not in git('log','-200','--format=%B%x00').decode()
result=subprocess.run([os.sys.executable,'-X','utf8','-B',str(source/'scripts/publish_code_daily.py'),'plan','--source-root',str(source),'--repo-root',str(repo),'--policy',str(source/'configs/publication/code_only.json')],cwd=repo,env=env,capture_output=True)
(V/'daily_20261122_plan.stdout.json').write_bytes(result.stdout);(V/'daily_20261122_plan.stderr.log').write_bytes(result.stderr);assert result.returncode==0,result.stderr.decode(errors='replace')[-1800:]
plan=json.loads(result.stdout);policy,_=pub.load_policy(source/'configs/publication/code_only.json');assert plan['HEAD']==head and len(plan['files'])==len(policy['files'])==2670
changes=[r for r in plan['files'] if r['action']!='unchanged'];assert plan['changes']==len(changes)==127
assert set(r['path'] for r in changes)==set(r['path'] for r in review['records'] if r['action']!='unchanged')|{'configs/publication/code_only.json'}
assert [r['path'] for r in changes if r['action']=='update']==['configs/publication/code_only.json']
assert not set(review['exact_task_data_exclusions'])&set(policy['files']);outgoing=pub.audit_outgoing(repo,policy,remote);assert outgoing==[]
semantic={}
def definitions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((source/path).read_bytes()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
for tag in ['bee8bsft','bee8brl']:
    a=definitions('universal_eval/model/adapters/'+tag+'_own_native_bf16_4096_20261116.py');b=definitions('universal_eval/model/adapters/'+tag+'_own_native_bf16_4096_complete_LZMA_20261121.py');assert set(a)==set(b) and [n for n in a if a[n]!=b[n]]==['save_npz']
    old=(source/'scripts'/('verify_'+tag+'_own_native_bf16_4096_20261116.py')).read_text();new=(source/'scripts'/('verify_'+tag+'_own_native_bf16_4096_original_raw_fallback_v2_20261121.py')).read_text();assert new.replace('(choice or raw)','(choice or \'\')')==old
    semantic[tag]='Only full lossless save_npz changes; all model/class math AST exact. Independent verifier only original raw fallback expected metadata expression changes.'
old=(source/'scripts/verify_ovisu1_own_original_understanding_bf16_512_20261116.py').read_text();new=(source/'scripts/verify_ovisu1_completed200_original_raw_prediction_fallback_metadata_v2_20261121.py').read_text();assert new.replace("details[i]['prediction']==(choice or raw)","details[i]['prediction']==choice")==old
semantic['Ovis_NF2']='Exactly original raw fallback prediction metadata expression; numerical functions, score and complete 200 denominator unchanged. Actual prior NF2/184 full target receipts retained.'
for path,digest in [('scripts/A100_alignment_evidence_runner_20261103.py','58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'),('scripts/publish_code_daily.py','e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30')]:assert hashlib.sha256((source/path).read_bytes()).hexdigest()==digest
categories=collections.Counter();files=[]
for row in changes:
    name=row['path'];raw=(source/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['SHA256'];pub.validate_content(name,raw,policy['files'][name])
    if name.startswith('universal_eval/model/'):category='Own original LLaVA-Mini graph, private original cache and full observers, or previously fully verified Bee lossless-only storage'
    elif name.startswith('universal_eval/algorithm/'):category='Existing explicit-choice Accuracy binding, original Greedy prompt bytes preserved'
    elif name.startswith('scripts/verify_'):category='Complete independent source/cast/NPU/KV/head/vision/full-report numerical acceptance; fixed original output schema'
    elif name.startswith('tests/') or name.startswith('scripts/tests/'):category='Actual private-cache/full-value invariants or qualified transaction rejection and real isolated workbook fixtures'
    elif name.startswith('configs/'):category='Actually consumed model/protocol/serial queue/exact retention/qualified publication or publisher configuration'
    elif '/maintenance/' in name:category='Source/protection/SHA/CAS/complete evidence handoff and read-only current process diagnostics; exact accepted correct nonexample retention only after full target audit'
    elif '/continue_' in name:category='Original unique EX serial model/fullNF/capacity/fresh3/fixed200 or qualified dual publication continuation'
    else:category='Explicit source preparation or actual Windows/strict-SSH/CAS/render/normal daily code publisher consumer'
    categories[category]+=1;entry=dict(row,semantic_category=category)
    if name.endswith('.json') and name!='configs/publication/code_only.json':assert 'input_records' not in json.loads(raw) and 'detailed_results' not in json.loads(raw)
    if name.endswith('.py'):entry['definitions']=[n.name for n in ast.parse(raw,filename=name).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    files.append(entry)
approval=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,Code_Day='2026-11-22',reviewed_plan_SHA256=plan['plan_SHA256'],remote_expected_before=remote,reviewed_changes=len(changes),reviewed_explicit_paths=len(plan['files']),source_root=str(source),source_collection_SHA256=review['source_collection_SHA256'],full_diff_SHA256=review['full_diff_SHA256'],full_diff_path=review['full_diff'],full_diff_bytes=Path(review['full_diff']).stat().st_size,all_new_paths_full_content_and_actual_dependencies_reviewed=True,new_JSON_consumers=review['new_JSON_actual_consumers'],exact_task_data_exclusions=review['exact_task_data_exclusions'],all_existing_model_and_frozen_runner_bytes_preserved=True,semantic_AST_and_full_diff_checks=semantic,own_LlavaMini_genuine3_already_launched_but_not_claimed_accepted=True,tests_passed_R_existing=23,tests_passed_R_private_cache_and_observer=3,tests_passed_local=19,MJS_syntax_passed=23,semantic_category_counts=dict(categories),files=files,outgoing_before=outgoing,repository='https://github.com/lyinguo/910b.git',branch='codex/910b-source-layout-20261114',no_merge_force_Gitee_history_weights_data_artifacts_or_credentials=True)
(V/'daily_20261122_plan_review.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in approval.items() if k not in ['files','new_JSON_consumers']},ensure_ascii=False))
