import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';s=json.loads((D/'state.json').read_bytes());out={'canonical_SHA256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'state.json',D/'registry.json',D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json']},'pending':{n:v for n,v in s['models'].items() if v['status']=='needs_adaptation'}}
print(json.dumps(out,ensure_ascii=False))