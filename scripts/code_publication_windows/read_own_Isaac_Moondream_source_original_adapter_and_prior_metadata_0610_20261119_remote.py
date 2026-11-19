import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';s=json.loads((D/'state.json').read_bytes())['models']
o=dict(UTC=datetime.now(timezone.utc).isoformat(),sources={},prior_named_stage_metadata={})
files=[R/'universal_eval/model/adapters/transformers_vlm.py']
for mid in ['PerceptronAI/Isaac-0.1','moondream/moondream3-preview']:
 W=Path(s[mid]['weight_paths'][0]);files += [p for p in W.iterdir() if p.is_file() and p.suffix=='.py']
for p in files:
 b=p.read_bytes();assert len(b)<1500000;o['sources'][str(p)]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),text=b.decode('utf8'))
for home in B.iterdir():
 if not home.is_dir() or not any(k in home.name.lower() for k in ['isaac','moondream']):continue
 rows={}
 for p in home.iterdir():
  if p.is_file() and p.suffix=='.json' and p.stat().st_size<1000000:
   b=p.read_bytes()
   try:rows[p.name]=dict(SHA256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
   except ValueError:pass
 o['prior_named_stage_metadata'][str(home)]=rows
print(json.dumps(o,ensure_ascii=False))
