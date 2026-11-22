import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';state=json.loads((D/'state.json').read_bytes());reg=json.loads((D/'registry.json').read_bytes());print(json.dumps(dict(state_SHA256=hashlib.sha256((D/'state.json').read_bytes()).hexdigest(),reg_SHA256=hashlib.sha256((D/'registry.json').read_bytes()).hexdigest(),state_keys=list(state),reg_keys=list(reg),pending_own_Llava_entries={k:v for k,v in state['models'].items() if any(x in k.lower() for x in ['llava','llama3','llama-3','moondream']) and v.get('status')!='done'},registry_sample={k:v for k,v in list(reg.items())[:2]}),ensure_ascii=False))
