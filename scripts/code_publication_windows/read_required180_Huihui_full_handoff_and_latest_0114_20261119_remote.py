import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
H=D/'heartbeat_handoffs/A100_verified180_Huihui_own2048_new_true3_current_resource_preserved_20261118_124318_189908'
b=(H/'handoff.txt').read_bytes();assert hashlib.sha256(b).hexdigest()=='5b2a06f96976c3f0bb9f6500dbbe6b6e9c82a890569921d8eab2511adbb1dc9c'
body=b.decode('utf8');assert '\r' not in body
f=(H/'facts.json').read_bytes();assert hashlib.sha256(f).hexdigest()=='7cf65fecdc0e282e08d5a31fc48308c821e135a3a722d9ec4a14b56fe635e4f8'
facts=json.loads(f);closing=json.loads((H/'closing_addendum.json').read_bytes());pointer=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());late={}
for name in [closing.get('latest_actual_note'),pointer.get('latest_actual_snapshot')]:
 if name:
  p=Path(name);assert p.resolve().is_relative_to((D/'heartbeat_handoffs').resolve());late[name]=json.loads(p.read_bytes())
print(json.dumps(dict(full_required_handoff_text=body,full_facts=facts,closing=closing,current_pointer=pointer,current_pointer_SHA256=hashlib.sha256((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes()).hexdigest(),full_late_actual_notes=late),ensure_ascii=False))
