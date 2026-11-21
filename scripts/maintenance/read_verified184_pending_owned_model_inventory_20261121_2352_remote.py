import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
state=json.loads((D/'state.json').read_bytes())
pending=[dict(model_id=m,entry=e) for m,e in state['models'].items() if e['status']=='needs_adaptation']
out=dict(UTC=datetime.now(timezone.utc).isoformat(),pending_count=len(pending),pending=pending,canonical_state_SHA256=hashlib.sha256((D/'state.json').read_bytes()).hexdigest())
for n in ['immediate_disk_recovery_cleanup_watch_20261120/state.json','immediate_disk_recovery_cleanup_watch_20261120/actual_failure.json']:
 p=D/n
 if p.exists():out[n]=json.loads(p.read_bytes())
print(json.dumps(out,ensure_ascii=False))
