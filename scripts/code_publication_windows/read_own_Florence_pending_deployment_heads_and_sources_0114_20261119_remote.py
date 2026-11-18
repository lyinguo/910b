import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct
R=Path('/root/projects/comperror');W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Florence2ForConditionalGeneration');rows=[]
for mid in ['microsoft/Florence-2-large-ft','F16/florence2-large-ft-gufeng_v3','AskUI/PTA-1']:
 root=W/mid;cfg=R/'configs/Florence2ForConditionalGeneration'/(mid.replace('/','--')+'.yaml');row=dict(model_id=mid,own_root=str(root),root_available=root.is_dir(),files={},own_safetensor_headers={})
 paths=[cfg]
 if root.is_dir():
  paths += [root/n for n in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer_config.json','special_tokens_map.json']]
  paths += list(root.glob('*.py'))
  row['own_file_inventory']=[dict(name=p.name,bytes=p.stat().st_size) for p in root.iterdir() if p.is_file()]
 for p in paths:
  if p.is_file():
   b=p.read_bytes();row['files'][str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode('utf8'))
 if root.is_dir():
  for p in root.glob('*.safetensors'):
   with p.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];assert n<32*1024**2;header=json.loads(f.read(n))
   row['own_safetensor_headers'][p.name]=dict(bytes=p.stat().st_size,headers={k:v for k,v in header.items() if k!='__metadata__'})
 rows.append(row)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),only_own_metadata_original_source_and_SF_headers=True,no_model_forward_no_weight_array_reread_no_external_writes=True,candidates=rows),ensure_ascii=False))
