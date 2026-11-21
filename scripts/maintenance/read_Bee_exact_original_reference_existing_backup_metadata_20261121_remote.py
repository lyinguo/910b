import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';h='d39447b7054fd381bb8c1801ab33ef141f17753488080249e0f50ed47bfc4a63';found=[];count=0
for p in B.glob('*/backup_manifest.json'):
 count+=1
 try:
  a=json.loads(p.read_bytes())
  for n,v in a.items():
   if isinstance(v,dict) and (v.get('sha256') or v.get('SHA256'))==h:found.append({'manifest':str(p),'record':v})
 except Exception:pass
print(json.dumps({'manifest_count':count,'matching_old_Bee_reference':found}))