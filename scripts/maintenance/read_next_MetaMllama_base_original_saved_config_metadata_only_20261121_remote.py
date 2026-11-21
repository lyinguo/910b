import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,os
R=Path('/root/projects/comperror');out={'files':{},'matching_config_names':[]}
for p in (R/'configs').glob('*/*Llama-3.2-11B-Vision*'):
 if 'Instruct' in p.name or 'bnb' in p.name or 'mlx' in p.name or 'unsloth' in p.name or 'alpindale' in p.name:continue
 b=p.read_bytes();out['files'][str(p)]={'SHA256':hashlib.sha256(b).hexdigest(),'complete_text':b.decode()}
for p in (R/'scripts').glob('*mllama*'):
 out['matching_config_names'].append(str(p.relative_to(R)))
print(json.dumps(out,ensure_ascii=False))