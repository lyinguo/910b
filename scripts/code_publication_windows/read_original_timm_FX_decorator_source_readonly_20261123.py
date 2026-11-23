"""Read the exact original FX decorators; no model or tensor execution."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
p=Path('/mnt/afs/miniconda3/envs/lvzhiqiang/lib/python3.12/site-packages/timm/layers/_fx.py');b=p.read_bytes()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),path=str(p),SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),complete_source_text=b.decode('utf8'),readonly_no_model_or_array_or_mutation=True),ensure_ascii=False))
