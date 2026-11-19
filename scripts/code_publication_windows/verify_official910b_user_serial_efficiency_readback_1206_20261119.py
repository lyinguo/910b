"""Exact app-managed heartbeat readback, with every non-prompt field preserved."""
from pathlib import Path
import hashlib,json,tomllib
P=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
before_raw=(P/'automation910b_before_user_serial_efficiency_20261119.toml').read_bytes();after_raw=Path('C:/Users/lyg/.codex/automations/910b/automation.toml').read_bytes()
before=tomllib.loads(before_raw.decode());after=tomllib.loads(after_raw.decode());fields=lambda value:{k:v for k,v in value.items() if k not in ['prompt','updated_at']}
requested=json.loads((P/'official910b_user_serial_efficiency_arguments_1203_20261119.json').read_bytes())['prompt']
assert fields(before)==fields(after) and after['id']=='910b' and after['status']=='ACTIVE' and after['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=300'
assert after['prompt']==requested.rstrip('\n')
(P/'automation910b_after_user_serial_efficiency_20261119.toml').write_bytes(after_raw)
(P/'automation910b_after_user_serial_efficiency_prompt_20261119.txt').write_text(after['prompt'],encoding='utf8',newline='\n')
result=dict(passed=True,before_preserved_fields=fields(before),after_preserved_fields=fields(after),before_TOML_SHA256=sha(before_raw),after_TOML_SHA256=sha(after_raw),after_prompt_SHA256=sha(after['prompt'].encode()),requested_prompt_normalized_SHA256=sha(requested.rstrip('\n').encode()),official_only_update=True)
(P/'official910b_user_serial_efficiency_readback_1206_20261119.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
