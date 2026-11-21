import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,yaml,struct
R=Path('/root/projects/comperror');out={'files':{},'SF_headers':{}}
for p in (R/'configs').glob('*/google--gemma-4-12B-it.yaml'):
 b=p.read_bytes();out['files'][str(p)]={'SHA256':hashlib.sha256(b).hexdigest(),'complete_text':b.decode()};q=yaml.safe_load(b);W=Path(q['model']['model_path']);out['own_original_weight_path']=str(W);out['readable']=W.is_dir()
 if W.is_dir():
  for n in ['config.json','generation_config.json','model.safetensors.index.json','preprocessor_config.json','tokenizer_config.json']:
   a=W/n
   if a.is_file():
    z=a.read_bytes();out['files'][str(a)]={'SHA256':hashlib.sha256(z).hexdigest(),'complete_text':z.decode()}
  idx=W/'model.safetensors.index.json'
  if idx.is_file():
   for n in sorted(set(json.loads(idx.read_bytes())['weight_map'].values())):
    a=W/n
    if not a.is_file():out['SF_headers'][n]={'missing':True};continue
    with a.open('rb') as f:s=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(s))
    out['SF_headers'][n]={'bytes':a.stat().st_size,'tensor_count':len(h)-('__metadata__' in h),'dtypes':sorted({x['dtype'] for k,x in h.items() if k!='__metadata__'}),'headers':h}
print(json.dumps(out,ensure_ascii=False))