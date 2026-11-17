import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';state=json.loads((D/'state.json').read_bytes());o={'UTC':datetime.now(timezone.utc).isoformat(),'actual_free_bytes':shutil.disk_usage(D).free,'recent_verified_runs':[],'Prithiv_metadata':{},'retention_documents':{}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for mid,e in state['models'].items():
 if e['status']!='done' or not e.get('original_protocol_final_qualification_complete') or str(e.get('completed_at',''))<'2026-11-16':continue
 f=Path(e['report_dir'])/'operator_lineage.json'
 if not f.exists():continue
 line=json.loads(f.read_bytes());actual=line.get('actual_execution',{});proofpaths=[p for p in e.get('attempts',[]) if p.get('count')==200]
 record=dict(model_id=mid,completed_at=e['completed_at'],report_dir=e['report_dir'],actual_execution_keys=list(actual),proof=proofpaths[-1] if proofpaths else None,homes={},lineage_paths={k:v for k,v in actual.items() if isinstance(v,str) and v.startswith('/')})
 homes=set()
 for k,v in actual.items():
  if isinstance(v,str) and v.startswith(str(R/'workspace/diagnostics/model_runs')):
   p=Path(v);h=p if p.is_dir() else p.parent;homes.add(h)
 # Only finite diagnostic homes explicitly referenced by current accepted reports.
 for h in homes:
  files=[p for p in h.rglob('*.npz') if p.is_file()];record['homes'][str(h)]={'npz_count':len(files),'npz_bytes':sum(p.stat().st_size for p in files),'files':[dict(path=str(p),bytes=p.stat().st_size) for p in files]}
 o['recent_verified_runs'].append(record)
for home in B.iterdir():
 if home.is_dir() and ('prithiv' in home.name.lower() or home.name=='DonutInvoice_v6_existing200_full_numeric_original_empty_schema_review_20261117'):
  small={}
  for f in home.iterdir():
   if f.is_file() and f.suffix=='.json' and f.stat().st_size<2*1024**2:small[f.name]={'SHA256':sha(f),'value':json.loads(f.read_bytes())}
  o['Prithiv_metadata'][str(home)]=small
for f in D.iterdir():
 if f.is_file() and f.suffix=='.json' and any(t in f.name.lower() for t in ['retention','matrix']):o['retention_documents'][str(f)]={'SHA256':sha(f),'value':json.loads(f.read_bytes())}
print(json.dumps(o,ensure_ascii=False))
