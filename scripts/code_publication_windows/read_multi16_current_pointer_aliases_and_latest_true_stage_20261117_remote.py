import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';p=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';v=json.loads(p.read_bytes());J=Path(v['current_same_approved_fresh3_unstarted200_home'])
print(json.dumps(dict(pointer_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),pointer=v,state=json.loads((J/'state.json').read_bytes()),completion=json.loads((J/'completion.json').read_bytes()) if (J/'completion.json').exists() else None),ensure_ascii=False))
