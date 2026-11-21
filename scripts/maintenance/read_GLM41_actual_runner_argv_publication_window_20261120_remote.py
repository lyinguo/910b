import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
out={}
for pid in [1622920,1618883,1624585,1620589]:
 p=Path('/proc')/str(pid)
 if p.exists():
  out[str(pid)]=dict(argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v],cwd=str((p/'cwd').resolve()),stat=(p/'stat').read_text())
print(json.dumps(out))
