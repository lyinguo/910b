import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';state=json.loads((D/'state.json').read_bytes());reg=json.loads((D/'registry.json').read_bytes())
pending={k:v for k,v in state['models'].items() if v['status']=='needs_adaptation'}
keys=set()
for v in pending.values():keys.update(v)
rows=[]
for k,v in pending.items():
 r={n:v[n] for n in ['row','model_path','weight_path','task','task_type','config','active_config','reason','last_error'] if n in v};r['model_id']=k;rows.append(r)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),pending_count=len(pending),entry_keys=sorted(keys),pending=rows,example_entries=dict(list(pending.items())[:2])),ensure_ascii=False))
