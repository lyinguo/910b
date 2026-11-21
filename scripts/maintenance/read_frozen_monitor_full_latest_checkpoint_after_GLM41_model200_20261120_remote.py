import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
R=Path('/root/projects/comperror');M=R/'workspace/diagnostics/batch_910b/incremental_monitor'
out={}
for p in [M/'latest.json',M/'checkpoint.json']:
 b=p.read_bytes();json.loads(b);out[str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),complete_text=b.decode('utf-8'))
print(json.dumps(dict(files=out)))
