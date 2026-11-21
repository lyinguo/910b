import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json,importlib.util
spec=importlib.util.find_spec('vllm');locations=list(spec.submodule_search_locations);assert len(locations)==1
root=Path(locations[0])/'multimodal/media';out={}
for name in ['base.py','image.py']:
 p=root/name;raw=p.read_bytes();out[str(p)]=dict(SHA256=hashlib.sha256(raw).hexdigest(),complete_text=raw.decode())
print(json.dumps(out,ensure_ascii=False))
