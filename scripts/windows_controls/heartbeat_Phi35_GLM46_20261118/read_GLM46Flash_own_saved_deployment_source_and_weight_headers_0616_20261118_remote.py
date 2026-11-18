import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct
R=Path('/root/projects/comperror');W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration/zai-org/GLM-4.6V-Flash')
files={};headers={}
for name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','chat_template.jinja','configuration_glm4v.py','modeling_glm4v.py']:
 p=W/name
 if p.is_file():files[str(p)]=dict(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,text=p.read_text())
cfg=R/'configs/Glm4vForConditionalGeneration/zai-org--GLM-4.6V-Flash.yaml'
if cfg.exists():files[str(cfg)]=dict(SHA256=hashlib.sha256(cfg.read_bytes()).hexdigest(),text=cfg.read_text())
for p in W.glob('*.safetensors'):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
 headers[p.name]=dict(bytes=p.stat().st_size,source_header={k:v for k,v in h.items() if k!='__metadata__'})
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),model_id='zai-org/GLM-4.6V-Flash',own_only_no_weight_arrays_or_model_forward=True,files=files,headers=headers),ensure_ascii=False))
