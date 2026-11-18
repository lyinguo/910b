from pathlib import Path
import os,sys,json,subprocess
P=Path(__file__).parent;source=Path('D:/code-lv/.tmp/cd18s');repo=Path('D:/code-lv/910b');mode=sys.argv[1];assert mode in ['plan','publish'];args=[sys.executable,'-X','utf8','-B',str(source/'scripts/publish_code_daily.py'),mode,'--source-root',str(source),'--repo-root',str(repo),'--policy',str(source/'configs/publication/code_only.json')]
if mode=='publish':
 review=json.loads((P/'daily_20261118_plan_review.json').read_bytes());assert review['passed'] and review['tests_passed']==19;args+=['--expected-plan-sha',review['reviewed_plan_SHA256'],'--message','Version own native model adaptations and continuous private-stage evaluation']
env=dict(os.environ);env['PATH']='C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+env['PATH'];env['PYTHONDONTWRITEBYTECODE']='1'
if mode=='publish':
 remote=subprocess.run(['git','-c','credential.interactive=never','ls-remote','origin','refs/heads/codex/910b-source-layout-20261114'],cwd=repo,env=env,capture_output=True)
 assert remote.returncode==0,remote.stderr.decode('utf-8',errors='replace')[-1000:]
 assert remote.stdout.decode().split()[0]==review['remote_expected_before'],'Remote branch changed; stop before mutation'
r=subprocess.run(args,cwd=repo,env=env,capture_output=True);(P/('daily_20261118_'+mode+'.stdout.json')).write_bytes(r.stdout);(P/('daily_20261118_'+mode+'.stderr.log')).write_bytes(r.stderr)
if r.returncode:print(r.stderr.decode('utf-8',errors='replace')[-2500:]);sys.exit(r.returncode)
result=json.loads(r.stdout)
print(json.dumps(result if mode=='publish' else {k:v for k,v in result.items() if k!='files'},ensure_ascii=False))
