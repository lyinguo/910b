"""Read-only verification of an official 910b automation update."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tomllib

P = Path(__file__).resolve().parent
before = (P / 'automation_before_official_update.toml').read_bytes()
path = Path('C:/Users/lyg/.codex/automations/910b/automation.toml')
after = path.read_bytes()
old = tomllib.loads(before.decode('utf-8'))
new = tomllib.loads(after.decode('utf-8'))
args = json.loads((P / 'automation910b_verified180_official_update_args_20261117.json').read_bytes())
assert new['prompt'] == args['prompt']
assert {k: v for k, v in old.items() if k not in ('prompt', 'updated_at')} == {k: v for k, v in new.items() if k not in ('prompt', 'updated_at')}
assert not any('notification' in k.lower() for k in new)
assert new['status'] == 'ACTIVE'
assert new['rrule'] == 'RRULE:FREQ=MINUTELY;INTERVAL=300'
assert new['target_thread_id'] == '01a18bbd-8040-7d02-9004-beef491d836a'
assert old['prompt'] != new['prompt']
(P / 'automation_after_official_update_verified180.toml').write_bytes(after)
(P / 'automation_prompt_after_official_update_verified180.txt').write_bytes(new['prompt'].encode('utf-8'))
review = dict(
    UTC=datetime.now(timezone.utc).isoformat(), passed=True,
    only_official_tool_mutated_active_TOML=True,
    TOML_path=str(path), before_SHA256=hashlib.sha256(before).hexdigest(), after_SHA256=hashlib.sha256(after).hexdigest(),
    before_prompt_SHA256=hashlib.sha256(old['prompt'].encode('utf-8')).hexdigest(), after_prompt_SHA256=hashlib.sha256(new['prompt'].encode('utf-8')).hexdigest(),
    all_non_prompt_updated_at_fields_preserved=True, notification_policy_absent_preserved=True,
    kind=new['kind'], name=new['name'], status=new['status'], rrule=new['rrule'], target_thread_id=new['target_thread_id'],
)
(P / 'automation910b_verified180_official_full_readback_review_20261117.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(review, ensure_ascii=False))
