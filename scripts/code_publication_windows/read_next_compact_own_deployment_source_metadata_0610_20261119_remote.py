"""Read explicit candidate deployment/assets and safetensors headers, never arrays."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct,collections
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';state=json.loads((D/'state.json').read_bytes())['models']
ids=['thwri/CogFlorence-2-Large-Freeze','thwri/CogFlorence-2.1-Large','thwri/CogFlorence-2.2-Large','PerceptronAI/Isaac-0.1','moondream/moondream3-preview','gokaygokay/Florence-2-Flux','gokaygokay/Florence-2-Flux-Large','HuggingFaceM4/Florence-2-DocVQA']
o=dict(UTC=datetime.now(timezone.utc).isoformat(),models={})
for mid in ids:
 v=state[mid];W=Path(v['weight_paths'][0]);item=dict(state=v,weight_path=str(W),currently_readable=W.is_dir(),assets={},headers={})
 if W.is_dir():
  item['files']=[dict(name=p.name,bytes=p.stat().st_size) for p in W.iterdir() if p.is_file()]
  for name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json','modeling_florence2.py','configuration_florence2.py']:
   p=W/name
   if p.is_file() and p.stat().st_size<1500000:
    b=p.read_bytes();item['assets'][name]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode('utf8'))
  for p in W.glob('*.safetensors'):
   with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
   h={k:v for k,v in h.items() if k!='__metadata__'}
   item['headers'][p.name]=dict(count=len(h),dtype_counts=dict(collections.Counter(v['dtype'] for v in h.values())),own_heads={k:v for k,v in h.items() if any(x in k for x in ['lm_head','shared','embed_tokens'])})
 for arch in (R/'configs').iterdir():
  p=arch/(mid.replace('/','--')+'.yaml')
  if p.is_file():
   b=p.read_bytes();item.setdefault('saved_deployments',{})[str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),text=b.decode('utf8'))
 o['models'][mid]=item
print(json.dumps(o,ensure_ascii=False))
