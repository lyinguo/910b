"""Invoke the existing publisher with exactly the reviewed immutable plan SHA."""
from pathlib import Path
import hashlib,json,os,subprocess
V=Path(__file__).resolve().parent;source=Path('D:/code-lv/.tmp/cd22s1');repo=Path('D:/code-lv/910b')
review=json.loads((V/'daily_20261122_plan_review.json').read_bytes());cas=json.loads((V/'R_daily22_R_source_concurrent_CAS_before_publish_20261122.json').read_bytes())
assert review['passed'] and cas['passed'] and review['reviewed_changes']==127 and review['reviewed_explicit_paths']==2670 and review['tests_passed_R_existing']==23 and review['tests_passed_R_private_cache_and_observer']==3 and review['tests_passed_local']==19 and review['MJS_syntax_passed']==23
assert cas['declared_source_files_fresh_SHA256_checked']==2669 and cas['all_active_LlavaMini_sources_job_and_canonical_unchanged'] and cas['code_publication_policy_not_an_active_model_source']
assert hashlib.sha256((source/'configs/publication/code_only.json').read_bytes()).hexdigest()=='39e53cf773110d35891d79bd38c4e31b1d842da5198cf8b84d50870e838f59e5'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+env['PATH'];env.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.sslBackend',GIT_CONFIG_VALUE_0='openssl')
check=subprocess.run(['git','-c','credential.interactive=never','ls-remote','origin','refs/heads/codex/910b-source-layout-20261114'],cwd=repo,env=env,capture_output=True);assert check.returncode==0 and check.stdout.decode().split()[0]==review['remote_expected_before']
args=[os.sys.executable,'-X','utf8','-B',str(source/'scripts/publish_code_daily.py'),'publish','--source-root',str(source),'--repo-root',str(repo),'--policy',str(source/'configs/publication/code_only.json'),'--expected-plan-sha',review['reviewed_plan_SHA256'],'--message','Preserve verified184 delivery, exact accepted retention and own serial LLaVA-Mini adaptation']
with (V/'daily_20261122_publish.stdout.json').open('xb') as out,(V/'daily_20261122_publish.stderr.log').open('xb') as err:result=subprocess.run(args,cwd=repo,env=env,stdout=out,stderr=err)
print(json.dumps(dict(actual_returncode=result.returncode,stdout=str(V/'daily_20261122_publish.stdout.json'),stderr=str(V/'daily_20261122_publish.stderr.log'),reviewed_plan_SHA256=review['reviewed_plan_SHA256'],no_merge_force_or_extra_commit=True)));raise SystemExit(result.returncode)
