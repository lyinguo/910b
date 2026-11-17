import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';H=D/'heartbeat_handoffs/A100_verified179_Merve200_dual_multi16_two_card_pass_capacity_blocked_all_code_preserved_20261117_061053_934942'
raw=(H/'handoff.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()=='69ffe482b4766393f21cc05460f7d84aa7bfbe6c9b62432c09e532ff70c7722f';body=raw.decode('utf8');assert '\r' not in body
f=(H/'facts.json').read_bytes();assert hashlib.sha256(f).hexdigest()=='8c3dafd0f335a324e01ebed467e35086ec98917b86b45236c10748e822cd640d';facts=json.loads(f)
c=json.loads((H/'closing_addendum.json').read_bytes());p=json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes());late={}
for name in [c.get('latest_actual_notes'),p.get('latest_actual_snapshot')]:
 if name:
  path=Path(name);assert path.is_relative_to(H);late[name]=json.loads(path.read_bytes())
print(json.dumps(dict(full_required_handoff_text=body,full_facts=facts,closing=c,current_pointer=p,full_late_actual_notes=late),ensure_ascii=False))
