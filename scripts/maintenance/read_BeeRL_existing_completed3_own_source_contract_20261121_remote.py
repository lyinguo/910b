import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
out={'existing_BeeRL_homes':[],'sources':[]}
for p in B.glob('*Bee*'):
 if 'rl' not in p.name.lower():continue
 row={'home':str(p),'files':[x.name for x in p.iterdir() if x.is_file()]}
 for n in ['job.json','completion.json','actual_completion.json','capacity_review.json']:
  f=p/n
  if f.exists():row[n]=json.loads(f.read_bytes())
 out['existing_BeeRL_homes'].append(row)
for root in [R/'scripts',R/'universal_eval/model/adapters']:
 for p in root.glob('*bee*'):
  if 'rl' in p.name.lower() and p.suffix=='.py':out['sources'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()})
print(json.dumps(out,ensure_ascii=False))
