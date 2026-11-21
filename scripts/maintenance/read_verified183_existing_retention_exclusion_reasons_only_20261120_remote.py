import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'verified183_current_accepted_correct_nonexample_metadata_reaudit_continue_GLM41Thinking_20261120'
out={}
for p in sorted(H.glob('*.json')):
 if any(s in p.name for s in ['discovery','foreign','metadata']):
  raw=p.read_bytes();x=json.loads(raw);out[p.name]=dict(SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),keys=list(x) if isinstance(x,dict) else None)
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,list) and len(v)>30:
     out[p.name][k+'_count']=len(v);matches=[row for row in v if isinstance(row,dict) and any('prithiv' in str(z).lower() or 'mllama' in str(z).lower() or 'granite' in str(z).lower() or 'merve' in str(z).lower() for z in row.values() if isinstance(z,str))];out[p.name][k+'_selected_scope_records']=matches
    elif isinstance(v,(str,int,float,bool)) or v is None:out[p.name][k]=v
    elif k in ['eligible_scopes','accepted_scope_reviews','scopes','candidate_scopes','excluded_scopes']:out[p.name][k]=v
print(json.dumps(out,ensure_ascii=False))
