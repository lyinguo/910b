from pathlib import Path
import os,json,subprocess,hashlib
V=Path(__file__).resolve().parent;source=Path('D:/code-lv/.tmp/cd21s2');repo=Path('D:/code-lv/910b')
review=json.loads((V/'daily_20261121_plan_review.json').read_bytes());cas=json.loads((V/'R_daily21_R_source_concurrent_CAS_before_publish_20261121.json').read_bytes())
assert review['passed'] and cas['passed'] and review['reviewed_changes']==206 and review['reviewed_explicit_paths']==2544 and review['tests_passed_R']==23 and review['tests_passed_local']==19 and review['MJS_syntax_passed']==21
assert cas['declared_source_files_fresh_SHA256_checked']==2543 and cas['all_active_Ovis_sources_unchanged'] and cas['code_publication_policy_not_an_active_model_source']
assert hashlib.sha256((source/'configs/publication/code_only.json').read_bytes()).hexdigest()=='c8a705028535eace9acb88fb60fbe45bf6afa3b974a9a3566201d44698786f05'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+env['PATH'];env['GIT_CONFIG_COUNT']='1';env['GIT_CONFIG_KEY_0']='http.sslBackend';env['GIT_CONFIG_VALUE_0']='openssl'
check=subprocess.run(['git','-c','credential.interactive=never','ls-remote','origin','refs/heads/codex/910b-source-layout-20261114'],cwd=repo,env=env,capture_output=True);assert check.returncode==0 and check.stdout.decode().split()[0]==review['remote_expected_before']
args=[os.sys.executable,'-X','utf8','-B',str(source/'scripts/publish_code_daily.py'),'publish','--source-root',str(source),'--repo-root',str(repo),'--policy',str(source/'configs/publication/code_only.json'),'--expected-plan-sha',review['reviewed_plan_SHA256'],'--message','Preserve own-model serial evaluations, exact storage recovery and independent acceptance']
with (V/'daily_20261121_publish.stdout.json').open('xb') as out,(V/'daily_20261121_publish.stderr.log').open('xb') as err:r=subprocess.run(args,cwd=repo,env=env,stdout=out,stderr=err)
print(json.dumps(dict(actual_returncode=r.returncode,stdout=str(V/'daily_20261121_publish.stdout.json'),stderr=str(V/'daily_20261121_publish.stderr.log'),reviewed_plan_SHA256=review['reviewed_plan_SHA256'],no_merge_force_or_extra_commit=True)));raise SystemExit(r.returncode)
