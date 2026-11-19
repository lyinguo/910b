import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'heartbeat_handoffs/A100_verified180_Day19_pushed_Huihui_GLM_Phi_capacity_blocked_20261118_174952_130723'
b=(H/'handoff.txt').read_bytes();assert hashlib.sha256(b).hexdigest()=='684ebc7449cd9089f11f430ba18088635a423f91db8b43959c94483fb4e715bd'
body=b.decode('utf8');assert '\r' not in body
f=(H/'facts.json').read_bytes();assert hashlib.sha256(f).hexdigest()=='ea7e1d602bce1f66064b39b88daa1f7ebdc691a7059504c4a1495dc6b7a8628c'
facts=json.loads(f);closing=json.loads((H/'closing_addendum.json').read_bytes());pointer=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());late={}
for name in [closing.get('latest_actual_note'),pointer.get('latest_actual_snapshot')]:
 if name:
  p=Path(name);assert p.resolve().is_relative_to((D/'heartbeat_handoffs').resolve());late[name]=json.loads(p.read_bytes())
print(json.dumps(dict(full_required_handoff_text=body,full_facts=facts,closing=closing,current_pointer=pointer,current_pointer_SHA256=hashlib.sha256((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes()).hexdigest(),full_late_actual_notes=late),ensure_ascii=False))
