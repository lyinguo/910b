import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,yaml,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';MID='zai-org/AutoGLM-Phone-9B';p=R/'configs/Glm4vForConditionalGeneration/zai-org--AutoGLM-Phone-9B.yaml';raw=p.read_bytes();cfg=yaml.safe_load(raw);entry=json.loads((D/'state.json').read_bytes())['models'][MID];m=cfg['model'];names=list(dict.fromkeys(entry.get('weight_paths',[])+[m.get('model_path',m.get('kwargs',{}).get('model_path',''))]));roots={}
for n in names:
 if not n:continue
 q=Path(n)
 try:roots[n]=dict(is_dir=q.is_dir(),files=[dict(name=z.name,bytes=z.stat().st_size) for z in q.iterdir() if z.is_file()])
 except OSError as e:roots[n]=dict(temporarily_unreadable_or_missing_now=True,error=str(e))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,original_saved_config=cfg,original_saved_config_SHA256=hashlib.sha256(raw).hexdigest(),original_state_entry=entry,own_inventory_roots=roots,no_model_NN_or_G_or_permanent_missing_classification_change=True),ensure_ascii=False))
