import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct,importlib.util
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';s=json.loads((D/'state.json').read_bytes())['models'];mid='microsoft/Phi-3.5-vision-instruct';W=Path(s[mid]['weight_paths'][0]);o={'UTC':datetime.now(timezone.utc).isoformat(),'model_id':mid,'sources':{},'headers':{},'own_assets':{}}
spec=importlib.util.find_spec('vllm');base=Path(spec.origin).parent
paths=[R/'universal_eval/model/adapters/vllm_adapter.py',base/'model_executor/models/phi3v.py',base/'model_executor/models/phi3.py',base/'model_executor/models/phi3_small.py',base/'model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.py',W/'modeling_phi3_v.py',W/'processing_phi3_v.py',W/'configuration_phi3_v.py']
for p in paths:
 if p.exists():o['sources'][str(p)]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
for p in W.iterdir():
 if p.name.endswith('.json') or p.suffix in ['.jinja','.model']:
  if p.is_file():o['own_assets'][str(p)]={'bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
for p in W.glob('*.safetensors'):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];v=json.loads(f.read(n))
 o['headers'][str(p)]={'bytes':p.stat().st_size,'SHA256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'tensors':{k:x for k,x in v.items() if k!='__metadata__'}}
print(json.dumps(o,ensure_ascii=False))
