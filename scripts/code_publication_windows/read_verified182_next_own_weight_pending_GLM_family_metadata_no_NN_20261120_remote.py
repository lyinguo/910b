import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';state=json.loads((D/'state.json').read_bytes());out=[]
for mid,r in state['models'].items():
 if r['status']!='needs_adaptation' or not any(x in mid.lower() for x in ['glm','cogvlm','minicpm','qwen2.5-vl-3b']):continue
 cfg=R/'configs/Glm4vForConditionalGeneration'/(mid.replace('/','--')+'.yaml')
 out.append(dict(model_id=mid,state=r,own_saved_GLM_config=str(cfg),own_saved_GLM_config_text=cfg.read_text() if cfg.exists() else None))
print(json.dumps(out,ensure_ascii=False))
