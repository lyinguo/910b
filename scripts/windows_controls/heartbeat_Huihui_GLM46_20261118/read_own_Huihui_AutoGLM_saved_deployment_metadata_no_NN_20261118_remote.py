import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct
R=Path('/root/projects/comperror')
root=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration')
models=['huihui-ai/Huihui-GLM-4.6V-Flash-abliterated','zai-org/AutoGLM-Phone-9B','zai-org/AutoGLM-Phone-9B-Multilingual']
out=[]
for mid in models:
 w=root/mid;c=R/'configs/Glm4vForConditionalGeneration'/(mid.replace('/','--')+'.yaml')
 row=dict(model_id=mid,root=str(w),root_readable=w.is_dir(),saved_config=str(c),files={},safetensor_headers={})
 for p in [c]+[w/n for n in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','chat_template.jinja','model.safetensors.index.json']]:
  if p.is_file():
   b=p.read_bytes();row['files'][str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode('utf8'))
 if w.is_dir():
  row['own_small_file_inventory']=[dict(name=p.name,bytes=p.stat().st_size) for p in w.iterdir() if p.is_file()]
  for p in sorted(w.glob('*.safetensors')):
   with p.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];assert n<32*1024*1024;h=json.loads(f.read(n))
   row['safetensor_headers'][p.name]=dict(bytes=p.stat().st_size,tensors={k:v for k,v in h.items() if k!='__metadata__'})
 out.append(row)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),only_own_saved_metadata_and_headers=True,no_weight_arrays_no_learned_forward_no_dispatch=True,candidates=out),ensure_ascii=False))
