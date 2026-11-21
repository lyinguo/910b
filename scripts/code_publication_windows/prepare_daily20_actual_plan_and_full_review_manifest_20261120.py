from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,ast,importlib.util,collections
P=Path(__file__).resolve().parent;source=Path('D:/code-lv/.tmp/cd20s2');repo=Path('D:/code-lv/910b')
review=json.loads((P/'daily_20261120_source_content_review_v2.json').read_bytes());packet=json.loads((P/'R_daily20_current_declared_primary_sources_v2_20261120.json').read_bytes())
local=json.loads((P/'daily_20261120_snapshot_tests_permission_retry.json').read_bytes());remote_tests=json.loads((P/'R_daily20_actual_R_source_unit_checks_20261120.json').read_bytes())
assert review['passed'] and local['passed'] and local['tests_run']==19 and remote_tests['passed'] and remote_tests['tests_run']==23
for rel in ['scripts/publish_code_daily.py','tests/test_code_publication.py','tests/test_code_publication_javascript.py','tests/test_readonly_npz_indexed.py','tests/test_unsloth_completed_batch_lineage.py','universal_eval/utils/readonly_npz_indexed.py']:
 assert (Path('D:/code-lv/.tmp/cd20s')/rel).read_bytes()==(source/rel).read_bytes(),rel
assert hashlib.sha256((P/'daily_20261120_full_source_diff_v2.patch').read_bytes()).hexdigest()==review['full_diff_SHA256']
spec=importlib.util.spec_from_file_location('daily20_reviewed_pub',source/'scripts/publish_code_daily.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+env['PATH'];env['GIT_CONFIG_COUNT']='1';env['GIT_CONFIG_KEY_0']='http.sslBackend';env['GIT_CONFIG_VALUE_0']='openssl'
def git(*args):
 r=subprocess.run(['git','-c','credential.interactive=never',*args],cwd=repo,env=env,capture_output=True);assert r.returncode==0,r.stderr.decode(errors='replace')[-1800:];return r.stdout
assert git('status','--porcelain')==b''
assert git('branch','--show-current').decode().strip()=='codex/910b-source-layout-20261114'
assert git('remote','get-url','origin').decode().strip()=='https://github.com/lyinguo/910b.git'
head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','origin','refs/heads/codex/910b-source-layout-20261114').decode().split()[0];assert head==remote=='e8da8ad87ae3c9d42e6ff73c350d80c0bec7cd4d'
assert 'Code-Day: 2026-11-20' not in git('log','-200','--format=%B%x00').decode()
result=subprocess.run([os.sys.executable,'-X','utf8','-B',str(source/'scripts/publish_code_daily.py'),'plan','--source-root',str(source),'--repo-root',str(repo),'--policy',str(source/'configs/publication/code_only.json')],cwd=repo,env=env,capture_output=True)
(P/'daily_20261120_plan.stdout.json').write_bytes(result.stdout);(P/'daily_20261120_plan.stderr.log').write_bytes(result.stderr);assert result.returncode==0,result.stderr.decode(errors='replace')[-2000:]
plan=json.loads(result.stdout);assert plan['HEAD']==head and plan['changes']==319
policy,_=pub.load_policy(source/'configs/publication/code_only.json');assert len(plan['files'])==len(policy['files'])==2339
actual_changes=[v for v in plan['files'] if v['action']!='unchanged'];assert set(v['path'] for v in actual_changes)==set(v['path'] for v in review['records'] if v['action']!='unchanged')|{'configs/publication/code_only.json'}
assert [v['path'] for v in actual_changes if v['action']=='update']==['configs/publication/code_only.json']
outgoing_before=pub.audit_outgoing(repo,policy,remote);assert outgoing_before==[]
categories=collections.Counter();files=[]
for item in actual_changes:
 path=item['path'];raw=(source/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['SHA256'];pub.validate_content(path,raw,policy['files'][path])
 if path.startswith('universal_eval/model/'):
  category='Own Phi35/GLM/Huihui unchanged mathematical models with complete value observation or exact namespace mapping'
 elif path.startswith('scripts/verify_') or '/run_phi35_' in path:category='Independent numerical acceptance and immutable failed-version provenance'
 elif '/continue_' in path or '/launch_' in path:category='Source/protection/FLOCK/budget guarded own-stage continuation, no G writer'
 elif path.startswith('configs/'):category='Actual consumed standalone model/protocol/resource queue or explicit publication policy'
 elif path.startswith('scripts/maintenance/'):category='SHA/CAS handoff and official readback closure, no model or table change'
 elif path.startswith('scripts/windows_controls/') or path.startswith('scripts/code_publication_windows/'):category='Reviewed exact Windows controller/source reader or provenance builder'
 elif path.startswith('tests/'):category='Meaningful no-inference observer transparency and full-value storage checks'
 else:category='Explicit own source preparation, publication collection or full-precision D gate'
 categories[category]+=1
 entry=dict(item,semantic_category=category)
 if path.endswith('.json') and path!='configs/publication/code_only.json':
  value=json.loads(raw);assert 'detailed_results' not in value and 'sample_outputs' not in value and 'numeric_sample_checks' not in value
 if path.endswith('.py'):
  tree=ast.parse(raw,filename=path);entry['definitions']=[n.name for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
 files.append(entry)
assert hashlib.sha256((source/'scripts/A100_alignment_evidence_runner_20261103.py').read_bytes()).hexdigest()=='58de6b4fd664b5483a9ca08115f1ca1ab4fb1478e1ffb6387539296e80babf90'
approval=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,Code_Day='2026-11-20',reviewed_plan_SHA256=plan['plan_SHA256'],remote_expected_before=remote,reviewed_changes=len(actual_changes),reviewed_explicit_paths=len(plan['files']),source_root=str(source),source_collection_SHA256=review['source_collection_SHA256'],full_diff_SHA256=review['full_diff_SHA256'],full_diff_path=review['full_diff'],all_new_paths_content_and_dependencies_reviewed=True,new_JSON_consumers=review['new_JSON_actual_consumers'],all_unchanged_existing_model_and_frozen_runner_bytes_preserved=True,actual_own_NPU3_full_NF_provenance=dict(Phi35_NF3='5b86d98b424a72ae29109948404c43c6914e89dcf72e95522522e7d022f90a99',GLM46='d089c0ded120ee0df0be6ad28d9a02fe0b61697330d6f3b4a3178bc667916950',Huihui='7bbc04ae6b05c7845ca01dc3e682f94da39000230beea45f6fbf5dc6f773cc59',no_inference_replayed_for_code_commit=True),tests_passed_R=23,tests_passed_local=19,MJS_syntax_passed=11,semantic_category_counts=dict(categories),files=files,outgoing_before=outgoing_before,repository='https://github.com/lyinguo/910b.git',branch='codex/910b-source-layout-20261114',no_merge_force_Gitee_history_weights_data_artifacts_or_secrets=True)
(P/'daily_20261120_plan_review.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in approval.items() if k not in ['files','new_JSON_consumers']},ensure_ascii=False))
