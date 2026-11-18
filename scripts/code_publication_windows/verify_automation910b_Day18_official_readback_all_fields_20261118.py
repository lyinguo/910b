"""Read the active TOML without modifying it; verify the official tool result."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tomllib

P=Path(__file__).resolve().parent
before=(P/'automation910b_before_CodeDay18_handoff_update_20261118.toml').read_bytes()
path=Path('C:/Users/lyg/.codex/automations/910b/automation.toml')
after=path.read_bytes()
old=tomllib.loads(before.decode('utf-8'))
new=tomllib.loads(after.decode('utf-8'))
args=json.loads((P/'automation910b_verified180_Day18_official_update_args_20261118.json').read_bytes())
assert args['prompt'].endswith('\n')
assert new['prompt']==args['prompt'][:-1], 'Only the previously observed official removal of one terminal LF is allowed'
assert {k:v for k,v in old.items() if k not in ('prompt','updated_at')}=={k:v for k,v in new.items() if k not in ('prompt','updated_at')}
assert not any('notification' in k.lower() for k in new)
assert new['id']=='910b' and new['status']=='ACTIVE'
assert new['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300'
assert new['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a'
assert new['name']=='跟进910B模型评测与规范检查'
(P/'automation910b_after_Day18_verified_official_update_20261118.toml').write_bytes(after)
(P/'automation910b_after_Day18_verified_official_update_20261118.prompt.txt').write_bytes(new['prompt'].encode('utf-8'))
result=dict(UTC=datetime.now(timezone.utc).isoformat(),passed=True,only_official_tool_mutated_active_TOML=True,
 before_SHA256=hashlib.sha256(before).hexdigest(),after_SHA256=hashlib.sha256(after).hexdigest(),
 before_prompt_SHA256=hashlib.sha256(old['prompt'].encode()).hexdigest(),after_prompt_SHA256=hashlib.sha256(new['prompt'].encode()).hexdigest(),
 exact_terminal_LF_normalization=True,all_non_prompt_updated_at_fields_preserved=True,notification_policy_absent_preserved=True,
 id=new['id'],kind=new['kind'],status=new['status'],name=new['name'],rrule=new['rrule'],target_thread_id=new['target_thread_id'])
out=P/'automation910b_verified180_Day18_official_readback_review_20261118.json'
out.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps(result,ensure_ascii=False))
