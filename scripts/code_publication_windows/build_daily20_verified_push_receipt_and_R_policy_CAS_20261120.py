"""Build a one-use, SHA-bound archival receipt after a verified code-only push."""
from pathlib import Path
from datetime import datetime, timezone
import ast, base64, hashlib, json, os, subprocess

P = Path(__file__).resolve().parent
repo = Path('D:/code-lv/910b')
source = Path('D:/code-lv/.tmp/cd20s2')
git = 'C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
read = lambda name: json.loads((P / name).read_bytes())
published = read('daily_20261120_publish.stdout.json')
review = read('daily_20261120_plan_review.json')
cas = read('R_daily20_R_source_CAS_before_publish_actual_20261120.json')
assert published['status'] == 'pushed' and published['day'] == '2026-11-20'
commit = 'be09d9a8b637a6dd54f9be65db8609e89af92f64'
assert published['commit'] == commit and published['audited_outgoing_commits'] == [commit]
def run(*args):
    result = subprocess.run([git, '-C', str(repo), '-c', 'http.sslBackend=openssl', '-c', 'credential.interactive=never', *args], capture_output=True, check=True)
    return result.stdout
assert run('rev-parse', 'HEAD').decode().strip() == commit
assert run('ls-remote', 'origin', 'refs/heads/codex/910b-source-layout-20261114').decode().split()[0] == commit
assert not run('status', '--porcelain=v1', '-z')
assert run('rev-list', '--count', review['remote_expected_before'] + '..HEAD').strip() == b'1'
assert b'Code-Day: 2026-11-20' in run('show', '-s', '--format=%B', 'HEAD')
new = (source / 'configs/publication/code_only.json').read_bytes()
assert sha(new) == '7c60205d2f555e5502d50594b388154558a81e22d2e8002bc47244f20f09a550'
assert (repo / 'configs/publication/code_only.json').read_bytes() == new
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), passed=True, Code_Day='2026-11-20',
    status='pushed_and_remote_verified', repository='https://github.com/lyinguo/910b.git',
    branch=published['branch'], commit=commit, remote_commit=commit, parent=review['remote_expected_before'],
    explicit_paths=2339, changed_files=319, new_source_or_config_files=318,
    only_existing_file_update='configs/publication/code_only.json', one_new_Code_Day20_commit=True,
    worktree_clean=True, no_merge_force_or_Gitee_history=True,
    only_code_config_necessary_documents=True, no_weights_data_arrays_reports_logs_or_credentials=True,
    tests_R=23, local_publication_tests=19, javascript_actual_node_syntax_checks=11,
    reviewed_plan_SHA256=review['reviewed_plan_SHA256'], full_source_diff_SHA256=sha((P/'daily_20261120_full_source_diff_v2.patch').read_bytes()),
    source_content_review_SHA256=sha((P/'daily_20261120_source_content_review_v2.json').read_bytes()),
    outgoing_full_tree_audited_by_actual_publisher_before_push=True,
    source_CAS_SHA256=sha((P/'R_daily20_R_source_CAS_before_publish_actual_20261120.json').read_bytes()),
    prior_Day19_collector_root_runpy_and_encoding_failures_remain_preserved=True,
    current_Day20_local_test_five_temp_permission_errors_and_successful_same_tests_retry_preserved=True,
    original_source_SyntaxWarnings_preserved_no_source_reformat=True)
raw = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
(P/'daily_20261120_receipt.json').write_bytes(raw)
cas['canonical_SHA256']['/root/projects/comperror/workspace/diagnostics/batch_910b/A100_ALIGNMENT_EVIDENCE_RETENTION_LEDGER.json'] = '06f870990721bdb4e58a409f842980b7b4578eddc23367263d041c052f8e43ce'
data = dict(policy=base64.b64encode(new).decode(), policy_SHA256=sha(new), receipt=base64.b64encode(raw).decode(), receipt_SHA256=sha(raw), canonical=cas['canonical_SHA256'])
remote = 'DATA=' + repr(data) + '\n' + '''import sys
sys.dont_write_bytecode=True
from pathlib import Path
import base64,hashlib,json
R=Path('/root/projects/comperror');C=R/'workspace/diagnostics/batch_910b/code_publication_source_layout_20261114'
sha=lambda b:hashlib.sha256(b).hexdigest()
for path,digest in DATA['canonical'].items():assert sha(Path(path).read_bytes())==digest,path
p=R/'configs/publication/code_only.json';old=p.read_bytes()
assert sha(old)=='b6974eebfc0b50bd7d817eafba8f23cc516960c1db854dd1f6d7421d927c057b'
assert sha((R/'scripts/publish_code_daily.py').read_bytes())=='e67fc04910b201585870593590b92b77fd308d6dc8eb342206c81a01f6a54e30'
new=base64.b64decode(DATA['policy'],validate=True);assert sha(new)==DATA['policy_SHA256']
receipt=base64.b64decode(DATA['receipt'],validate=True);assert sha(receipt)==DATA['receipt_SHA256']
H=C/'daily_20261120_actual_reviewed_commit_remote_receipts';H.mkdir(exist_ok=False)
(H/'previous_reviewed_code_policy.json').write_bytes(old)
(H/'daily_20261120_receipt.json').write_bytes(receipt)
p.write_bytes(new);assert p.read_bytes()==new
for path,digest in DATA['canonical'].items():assert sha(Path(path).read_bytes())==digest,path
v=dict(passed=True,receipt=str(H/'daily_20261120_receipt.json'),receipt_SHA256=sha(receipt),publication_policy_before_SHA256=sha(old),publication_policy_after_SHA256=sha(new),only_R_publication_consumer_policy_updated_no_R_model_or_canonical_or_frozen_job_changes=True)
(H/'actual_R_policy_receipt.json').write_text(json.dumps(v,indent=2)+'\\n',encoding='utf8')
print(json.dumps(v))
'''
ast.parse(remote)
(P/'archive_daily20_actual_code_push_and_reviewed_R_publication_policy_20261120_remote.py').write_text(remote, encoding='utf8', newline='\n')
print(json.dumps(dict(receipt_SHA256=sha(raw), commit=commit, verified_remote=True)))
