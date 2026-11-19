import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
R=Path('/root/projects/comperror');H=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119';p=json.loads((H/'plan.json').read_bytes());out=[]
for scope in p['scopes']:
 names={v['path'] for v in scope['remove']};rows={k:v for k,v in p['foreign_or_open_reference_reasons_preserved'].items() if k in names}
 if rows:
  reasons=sorted(set(z for values in rows.values() for z in values));out.append(dict(model_id=scope['model_id'],proof=scope['proof'],candidate_files=len(names),referenced_files=len(rows),all_distinct_reference_reasons=reasons,first_reference=next(iter(rows.items()))))
print(json.dumps(out,ensure_ascii=False))
