import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');p=R/'universal_eval/model/adapters/llava_more_vlm.py';b=p.read_bytes();print(json.dumps(dict(path=str(p),SHA256=hashlib.sha256(b).hexdigest(),complete_text=b.decode('utf8')),ensure_ascii=False))
