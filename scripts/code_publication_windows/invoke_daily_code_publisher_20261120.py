from pathlib import Path
import os, sys, json, subprocess

P = Path(__file__).resolve().parent
source = Path('D:/code-lv/.tmp/cd20s2')
repo = Path('D:/code-lv/910b')
review = json.loads((P / 'daily_20261120_plan_review.json').read_bytes())
cas = json.loads((P / 'R_daily20_R_source_CAS_before_publish_actual_20261120.json').read_bytes())
assert review['passed'] and review['tests_passed_R'] == 23 and review['tests_passed_local'] == 19
assert cas['passed'] and cas['reviewed_plan_SHA256'] == review['reviewed_plan_SHA256']
assert review['reviewed_changes'] == 319 and review['reviewed_explicit_paths'] == 2339
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
env['PATH'] = 'C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd' + os.pathsep + env['PATH']
env['GIT_CONFIG_COUNT'] = '1'; env['GIT_CONFIG_KEY_0'] = 'http.sslBackend'; env['GIT_CONFIG_VALUE_0'] = 'openssl'
remote = subprocess.run(['git', '-c', 'credential.interactive=never', 'ls-remote', 'origin', 'refs/heads/codex/910b-source-layout-20261114'], cwd=repo, env=env, capture_output=True)
assert remote.returncode == 0, remote.stderr.decode(errors='replace')[-1500:]
assert remote.stdout.decode().split()[0] == review['remote_expected_before'], 'Remote changed; refuse before mutation'
args = [sys.executable, '-X', 'utf8', '-B', str(source / 'scripts/publish_code_daily.py'), 'publish', '--source-root', str(source), '--repo-root', str(repo), '--policy', str(source / 'configs/publication/code_only.json'), '--expected-plan-sha', review['reviewed_plan_SHA256'], '--message', 'Preserve own GLM and AutoGLM serial adaptations with complete numerical audits']
result = subprocess.run(args, cwd=repo, env=env, capture_output=True)
(P / 'daily_20261120_publish.stdout.json').write_bytes(result.stdout)
(P / 'daily_20261120_publish.stderr.log').write_bytes(result.stderr)
if result.returncode:
    print(result.stderr.decode('utf-8', errors='replace')[-3500:]); sys.exit(result.returncode)
print(json.dumps(json.loads(result.stdout), ensure_ascii=False))
