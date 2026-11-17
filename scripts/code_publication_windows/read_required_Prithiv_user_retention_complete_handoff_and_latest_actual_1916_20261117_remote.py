import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'heartbeat_handoffs/A100_verified179_user_cleanup1138_Prithiv_card2_NEWfresh3_fixed200_original_real_lock_live_20261117_080127_947376'
raw=(H/'handoff.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()=='bd49e008e87b17af2f88745ec2bac092910d2c17eee05eb43b60594375cb3541';body=raw.decode('utf8');assert '\r' not in body
f=(H/'facts.json').read_bytes();assert hashlib.sha256(f).hexdigest()=='bcda5574f67c4d47ad1fe4959993007906c6eda2518feb068892e463fd427012';facts=json.loads(f)
c=json.loads((H/'closing_addendum.json').read_bytes());p=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());late={}
for name in [c.get('latest_actual_notes'),p.get('latest_actual_snapshot')]:
 if name:
  path=Path(name);assert path.is_relative_to(D/'heartbeat_handoffs');late[name]=json.loads(path.read_bytes())
print(json.dumps(dict(full_required_handoff_text=body,full_facts=facts,closing=c,current_pointer=p,full_late_actual_notes=late),ensure_ascii=False))
