import sys
sys.dont_write_bytecode = True
from pathlib import Path
import ast
import base64
import collections
import difflib
import hashlib
import importlib.util
import json
import subprocess

P = Path(__file__).resolve().parent
repo = Path('D:/code-lv/910b')
source = Path('D:/code-lv/.tmp/cd20s')
assert source.resolve().parent == Path('D:/code-lv/.tmp').resolve()
assert not source.exists(), 'Do not reuse a previous candidate snapshot'
source.mkdir()
packet_path = P / 'R_daily20_current_declared_primary_sources_v1_20261120.json'
packet = json.loads(packet_path.read_bytes())
policy = json.loads((repo / 'configs/publication/code_only.json').read_bytes())
assert policy == packet['policy_before']
assert packet['opaque_payload_scripts_excluded_with_exact_reason'] == []
assert packet['unconsumed_JSON_excluded'] == []
raws = {}
for rec in packet['records']:
    raw = base64.b64decode(rec['base64'], validate=True)
    assert len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['SHA256']
    assert rec['path'] not in raws
    raws[rec['path']] = raw
    path = source / rec['path']
    assert path.resolve().is_relative_to(source.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
assert 'run.py' in raws
spec = importlib.util.spec_from_file_location('reviewed_publisher_day20', source / 'scripts/publish_code_daily.py')
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)
records = []
diff = []
new_json = {}
javascript = []
node = Path('C:/Users/lyg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
for rec in packet['records']:
    name = rec['path']
    raw = raws[name]
    pub.validate_path(name, rec['entry']['role'])
    pub.validate_content(name, raw, rec['entry'])
    previous = (repo / name).read_bytes() if (repo / name).is_file() else None
    if name not in policy['files']:
        policy['files'][name] = dict(rec['entry'], initial_SHA256=rec['SHA256'])
    if name.endswith('.json') and name not in packet['policy_before']['files']:
        consumers = packet['new_JSON_consumers'][name]
        assert consumers and all(c in raws for c in consumers)
        new_json[name] = consumers
    if name.endswith('.mjs'):
        result = subprocess.run([str(node), '--check', str(source / name)], capture_output=True)
        assert result.returncode == 0, (name, result.stderr.decode('utf-8', errors='replace'))
        javascript.append(dict(path=name, SHA256=rec['SHA256'], actual_node_syntax_returncode=0))
    action = 'unchanged' if previous == raw else 'add' if previous is None else 'update'
    row = dict(path=name, SHA256=rec['SHA256'], bytes=len(raw), action=action, role=policy['files'][name]['role'])
    if name.endswith('.py'):
        tree = ast.parse(raw.decode('utf-8-sig'), filename=name)
        row['top_level_definitions'] = [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    if action != 'unchanged':
        before = (previous or b'').decode('utf-8-sig').splitlines(keepends=True)
        after = raw.decode('utf-8-sig').splitlines(keepends=True)
        diff.extend(difflib.unified_diff(before, after, fromfile='before/' + name, tofile='after/' + name))
    records.append(row)
for name in ('README.md', '.gitignore', '.gitattributes'):
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((repo / name).read_bytes())
new_policy = (json.dumps(policy, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
path = source / 'configs/publication/code_only.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(new_policy)
pub.load_policy(path)
assert not (set(packet['policy_before']['files']) - set(policy['files']))
diff.extend(difflib.unified_diff((repo / 'configs/publication/code_only.json').read_text().splitlines(keepends=True), new_policy.decode().splitlines(keepends=True), fromfile='before/configs/publication/code_only.json', tofile='after/configs/publication/code_only.json'))
diff_path = P / 'daily_20261120_full_source_diff.patch'
diff_path.write_text(''.join(diff), encoding='utf-8', newline='\n')
review = dict(
    passed=True, source_root=str(source), source_collection_SHA256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
    prior_explicit_paths=len(packet['policy_before']['files']), explicit_paths=len(policy['files']), records=records,
    new_JSON_actual_consumers=new_json, javascript_actual_syntax_checks=javascript,
    action_counts=dict(collections.Counter(r['action'] for r in records)),
    all_path_content_secret_python_AST_payload_and_operator_IR_boundaries_passed=True,
    all_local_maintenance_preserved=True, all_model_and_report_sources_preserved=True,
    no_source_repo_or_R_model_mutation=True,
    full_diff=str(diff_path), full_diff_SHA256=hashlib.sha256(diff_path.read_bytes()).hexdigest(),
    policy_before_SHA256=packet['policy_before_SHA256'], policy_after_SHA256=hashlib.sha256(new_policy).hexdigest(),
)
(P / 'daily_20261120_source_content_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({k: v for k, v in review.items() if k not in ('records', 'new_JSON_actual_consumers', 'javascript_actual_syntax_checks')}, ensure_ascii=False))
print('CHANGED_EXISTING', json.dumps([r for r in records if r['action'] == 'update'], ensure_ascii=False))
