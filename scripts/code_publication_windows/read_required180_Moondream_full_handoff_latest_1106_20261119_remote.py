import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'heartbeat_handoffs/A100_verified180_Moondream_own_frontend200_private_flex_operator5_no_model_NN_current_20261118_223823_705594'
b=(H/'handoff.txt').read_bytes();assert hashlib.sha256(b).hexdigest()=='d47ffcf7243cc7839eccb84b343b5f9537bae420fc2be1a427d894e5df153465';body=b.decode('utf8');assert '\r' not in body
f=(H/'facts.json').read_bytes();assert hashlib.sha256(f).hexdigest()=='b5692b6140796fc5dbfb3738bbb193ebae6be656a0d3f12c28a6569d9c88f001';facts=json.loads(f)
closing=json.loads((H/'closing_addendum.json').read_bytes());raw=(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes();pointer=json.loads(raw);late={}
for name in [closing.get('latest_actual_note'),pointer.get('latest_actual_snapshot')]:
 if name:
  p=Path(name);assert p.resolve().is_relative_to((D/'heartbeat_handoffs').resolve());late[name]=json.loads(p.read_bytes())
print(json.dumps(dict(full_required_handoff_text=body,full_facts=facts,closing=closing,current_pointer=pointer,current_pointer_SHA256=hashlib.sha256(raw).hexdigest(),full_late_actual_notes=late),ensure_ascii=False))
