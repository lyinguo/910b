import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,tomllib
P=Path(__file__).parent;old=json.loads((P/'910b_verified178_official_update_before_and_expected_20261117.json').read_bytes());raw=Path(old['automation_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==old['automation_before_SHA256'];a=tomllib.loads(raw.decode());prompt=(P/'910b_verified178_expected_official_prompt_20261117.txt').read_text(encoding='utf8');assert hashlib.sha256(prompt.encode()).hexdigest()==old['prompt_expected_SHA256']
assert a['target_thread_id']=='01a18bbd-8040-7d02-9004-beef491d836a' and 'notification_policy' not in a
print(json.dumps(dict(id='910b',mode='update',kind='heartbeat',name=a['name'],prompt=prompt,status=a['status'],rrule=a['rrule'],targetThreadId=a['target_thread_id']),ensure_ascii=False))
