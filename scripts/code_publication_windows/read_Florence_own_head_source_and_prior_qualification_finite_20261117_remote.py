import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';s=json.loads((D/'state.json').read_bytes())['models']
ids=['microsoft/Florence-2-large-ft','yifeihu/TF-ID-base','yifeihu/TFT-ID-1.0'];o={'UTC':datetime.now(timezone.utc).isoformat(),'models':{},'sources':{},'prior_small':{}}
for mid in ids:
 W=Path(s[mid]['weight_paths'][0]);header={}
 for f in W.glob('*.safetensors'):
  with f.open('rb') as inp:n=struct.unpack('<Q',inp.read(8))[0];h=json.loads(inp.read(n))
  header[f.name]={k:v for k,v in h.items() if k!='__metadata__' and any(t in k for t in ['embed','lm_head','shared','logits'])}
 o['models'][mid]={'head_source_headers':header}
 for n in ['modeling_florence2.py','configuration_florence2.py','processing_florence2.py']:
  f=W/n
  if f.exists():o['sources'][str(f)]={'SHA256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_text()}
for n in ['universal_eval/model/adapters/high_nine_florence_caption.py','universal_eval/model/adapters/florence2_adapter.py']:
 f=R/n
 if f.exists():o['sources'][str(f)]={'SHA256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_text()}
for p in D.iterdir():
 if p.is_dir() and 'florence' in p.name.lower():o['prior_small'][str(p)]=[str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f.suffix in ['.json','.txt'] and f.stat().st_size<1024*1024][:100]
prior=B/'next_Fara_Florence_actual_A100_qualification_0311_20261104_20261104_041246_069722'
if prior.exists():o['prior_small'][str(prior)]=[str(f.relative_to(prior)) for f in prior.rglob('*') if f.is_file() and f.suffix in ['.json','.txt'] and f.stat().st_size<1024*1024][:150]
print(json.dumps(o,ensure_ascii=False))
