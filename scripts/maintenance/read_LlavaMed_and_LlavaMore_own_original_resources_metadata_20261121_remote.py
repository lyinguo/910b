import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,yaml,struct
R=Path('/root/projects/comperror');out={'files':{},'headers':{}}
for mid in ['microsoft--llava-med-v1.5-mistral-7b.yaml','aimagelab--LLaVA_MORE-llama_3_1-8B-finetuning.yaml']:
 for p in (R/'configs').glob('*/'+mid):
  b=p.read_bytes();out['files'][str(p)]={'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()};c=yaml.safe_load(b);W=Path(c['model']['model_path']);out[str(W)]={'exists':W.is_dir()}
  if not W.is_dir():continue
  for n in ['config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','model.safetensors.index.json']:
   f=W/n
   if f.is_file():
    b=f.read_bytes();out['files'][str(f)]={'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
print(json.dumps(out,ensure_ascii=False))