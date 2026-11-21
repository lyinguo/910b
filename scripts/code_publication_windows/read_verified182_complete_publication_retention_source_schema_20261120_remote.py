import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';js=lambda p:json.loads(Path(p).read_bytes());out={}
for name in ['ZaiGLM46Flash_original_BF16_thinking4096_LZMA_verified181_blank_G_publication_20261119','AutoGLMPhone9B_original_BF16_2048_LZMA_verified182_blank_G_publication_20261120']:
 H=B/name
 for p in H.glob('*audit*.json'):
  x=js(p);out[str(p)]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),keys=list(x),value={k:v for k,v in x.items() if not isinstance(v,(dict,list))})
print(json.dumps(out,ensure_ascii=False))
