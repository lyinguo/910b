from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,ast,importlib.util,collections
V=Path(__file__).resolve().parent;source=Path('D:/code-lv/.tmp/cd21s2');repo=Path('D:/code-lv/910b')
review=json.loads((V/'daily_20261121_source_content_review_v2.json').read_bytes());local=json.loads((V/'daily_20261121_snapshot_tests_actual.json').read_bytes());remote_tests=json.loads((V/'R_daily21_actual_R_source_unit_checks_20261121.json').read_bytes())
assert review['passed'] and local['passed'] and local['tests_run']==19 and remote_tests['passed'] and remote_tests['tests_run']==23
assert len(review['javascript_actual_syntax_checks'])==21 and {x['path'] for x in review['javascript_actual_syntax_checks']}=={x['path'] for x in review['records'] if x['path'].endswith('.mjs')}
assert hashlib.sha256(Path(review['full_diff']).read_bytes()).hexdigest()==review['full_diff_SHA256']
spec=importlib.util.spec_from_file_location('daily21_reviewed_publisher',source/'scripts/publish_code_daily.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+env['PATH'];env['GIT_CONFIG_COUNT']='1';env['GIT_CONFIG_KEY_0']='http.sslBackend';env['GIT_CONFIG_VALUE_0']='openssl'
def git(*args):
 r=subprocess.run(['git','-c','credential.interactive=never',*args],cwd=repo,env=env,capture_output=True);assert r.returncode==0,r.stderr.decode(errors='replace')[-1800:];return r.stdout
assert git('status','--porcelain')==b'' and git('branch','--show-current').decode().strip()=='codex/910b-source-layout-20261114' and git('remote','get-url','origin').decode().strip()=='https://github.com/lyinguo/910b.git'
head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').decode().split()[0];assert head==remote=='be09d9a8b637a6dd54f9be65db8609e89af92f64'
assert 'Code-Day: 2026-11-21' not in git('log','-200','--format=%B%x00').decode()
r=subprocess.run([os.sys.executable,'-X','utf8','-B',str(source/'scripts/publish_code_daily.py'),'plan','--source-root',str(source),'--repo-root',str(repo),'--policy',str(source/'configs/publication/code_only.json')],cwd=repo,env=env,capture_output=True)
(V/'daily_20261121_plan.stdout.json').write_bytes(r.stdout);(V/'daily_20261121_plan.stderr.log').write_bytes(r.stderr);assert r.returncode==0,r.stderr.decode(errors='replace')[-2000:]
plan=json.loads(r.stdout);assert plan['HEAD']==head and plan['changes']==206
policy,_=pub.load_policy(source/'configs/publication/code_only.json');assert len(plan['files'])==len(policy['files'])==2544
changes=[v for v in plan['files'] if v['action']!='unchanged'];assert set(v['path'] for v in changes)==set(v['path'] for v in review['records'] if v['action']!='unchanged')|{'configs/publication/code_only.json'}
assert [v['path'] for v in changes if v['action']=='update']==['configs/publication/code_only.json'];assert not set(review['exact_task_data_exclusions'])&set(policy['files'])
outgoing=pub.audit_outgoing(repo,policy,remote);assert outgoing==[]
categories=collections.Counter();files=[]
for row in changes:
 path=row['path'];raw=(source/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['SHA256'];pub.validate_content(path,raw,policy['files'][path])
 if path.startswith('universal_eval/model/'):category='Own GLM41/Meta native graph and full value observations; Ovis complete lossless storage-only save'
 elif path.startswith('scripts/verify_'):category='Independent complete numerical verification, exact BF16 reference and original invalid metadata expectation'
 elif path.startswith('scripts/maintenance/'):category='Quiet/original EX guarded physical byte dedup or exact accepted correct nonexample retention, SHA/CAS archive and read-only diagnostics'
 elif '/continue_' in path or '/reserve_' in path:category='Own-model serial source/protection/FLOCK/full shape budget continuation and qualified publication priority'
 elif path.startswith('configs/'):category='Actual parameter/literal consumed model, protocol metadata, retention, queue or publication configuration'
 elif path.startswith(('scripts/windows_control/','scripts/code_publication_windows/','scripts/windows_controls/')):category='Explicit local or strict SSH read/prepare/CAS/render source, no embedded opaque deployment payload'
 else:category='Own source preparation, reviewed narrow publication or explicit required dependency'
 categories[category]+=1;entry=dict(row,semantic_category=category)
 if path.endswith('.json') and path!='configs/publication/code_only.json':assert 'input_records' not in json.loads(raw) and 'detailed_results' not in json.loads(raw)
 if path.endswith('.py'):
  tree=ast.parse(raw,filename=path);entry['definitions']=[n.name for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
 files.append(entry)
for path,digest in [('scripts/A100_alignment_evidence_runner_20261103.py','58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'),('scripts/publish_code_daily.py','e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30')]:assert hashlib.sha256((source/path).read_bytes()).hexdigest()==digest
approval=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,Code_Day='2026-11-21',reviewed_plan_SHA256=plan['plan_SHA256'],remote_expected_before=remote,reviewed_changes=len(changes),reviewed_explicit_paths=len(plan['files']),source_root=str(source),source_collection_SHA256=review['source_collection_SHA256'],full_diff_SHA256=review['full_diff_SHA256'],full_diff_path=review['full_diff'],all_new_paths_content_and_dependencies_reviewed=True,new_JSON_consumers=review['new_JSON_actual_consumers'],exact_task_data_exclusions=review['exact_task_data_exclusions'],actual_parameter_consumed_original_failed_config=review['actual_parameter_consumed_prior_failed_publication_config_included'],all_unchanged_existing_model_and_frozen_runner_bytes_preserved=True,only_new_Ovis_save_AST_and_Meta_NF4_expected_metadata_expression_changed=True,original_NF4_preNN_SHA_failure_preserved_and_new_consumer_actual_file_SHA_guard_passed=True,tests_passed_R=23,tests_passed_local=19,MJS_syntax_passed=21,semantic_category_counts=dict(categories),files=files,outgoing_before=outgoing,repository='https://github.com/lyinguo/910b.git',branch='codex/910b-source-layout-20261114',no_merge_force_Gitee_history_weights_data_artifacts_or_secrets=True)
(V/'daily_20261121_plan_review.json').write_bytes((json.dumps(approval,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in approval.items() if k not in ['files','new_JSON_consumers']},ensure_ascii=False))
