import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'heartbeat_handoffs/A100_verified180_Prithiv_dual_delivered_Ovis_complete_values_capacity_Phi35_meta_only_20261117_120850_056360'
raw=(H/'handoff.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()=='cc0003dbd0a3b4c603348f262e9705512773e435e1c329379abde38b746caa8d';body=raw.decode('utf8');assert '\r' not in body
f=(H/'facts.json').read_bytes();assert hashlib.sha256(f).hexdigest()=='d0df0cb779d0b20afd3e3819fd5d38f401db18c4ae7812ecce721d4e70c813a3';facts=json.loads(f)
c=json.loads((H/'closing_addendum.json').read_bytes());p=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());late={}
for name in [c.get('latest_actual_note'),p.get('latest_actual_snapshot')]:
 if name:
  path=Path(name);assert path.is_relative_to(D/'heartbeat_handoffs');late[name]=json.loads(path.read_bytes())
print(json.dumps(dict(full_required_handoff_text=body,full_facts=facts,closing=c,current_pointer=p,full_late_actual_notes=late),ensure_ascii=False))
