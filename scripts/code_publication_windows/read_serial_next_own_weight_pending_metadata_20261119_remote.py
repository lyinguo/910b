import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
def js(p):return json.loads(p.read_bytes())
state=js(D/'state.json');reg=js(D/'registry.json');pointer=js(D/'A100_ALIGNMENT_HANDOFF_CURRENT.json')
pending={k:v for k,v in state['models'].items() if v['status']=='needs_adaptation'}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),pending=pending,registry_pending={k:reg[k] for k in pending if k in reg},current_pointer=pointer,no_model_or_source_or_canonical_changes=True),ensure_ascii=False))
