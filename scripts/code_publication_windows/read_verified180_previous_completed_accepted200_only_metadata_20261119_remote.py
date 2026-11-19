import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119'
def js(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
state=js(D/'state.json');jobs=[];alternates=[];existing=js(H/'discovery.json');current={r['proof'] for r in existing['all_selected_proof_metadata']}
for jp in B.glob('*/job.json'):
 cp=jp.parent/'completion.json'
 if not cp.is_file():continue
 c=js(cp)
 if c.get('actual_returncode')!=0:continue
 j=js(jp)
 for acc in c.get('accepted',[]):
  if acc.get('count',acc.get('sample_count'))!=200:continue
  name=acc.get('proof')
  if not isinstance(name,str) or name in current or not Path(name).is_file():continue
  digest=acc.get('proof_sha256')
  if digest and sha(name)!=digest:continue
  p=js(name);mid=j.get('model_id',p.get('model_id'));e=state['models'].get(mid)
  if not e or e['status']!='done' or not p.get('passed') or p.get('sample_count')!=200 or p.get('historical_record_reuse') or p.get('error_indices'):continue
  report=Path(acc['report_dir']);series=[]
  for path,h in p.get('external_evidence_sha256',{}).items():
   f=Path(path)
   if f.suffix=='.npz' and f.is_file() and f.is_relative_to(R/'workspace/diagnostics/model_runs'):
    s=f.stat();series.append(dict(path=path,bytes=s.st_size,nlink=s.st_nlink,SHA256=h))
  if series:jobs.append(dict(model_id=mid,job=str(jp),job_SHA256=sha(jp),completion=str(cp),completion_SHA256=sha(cp),proof=name,proof_SHA256=sha(name),report_dir=str(report),current_report=e['report_dir'],proof_keys=list(p),count=len(series),bytes=sum(v['bytes'] for v in series),files=series,accepted_entry=acc))
for mid,e in state['models'].items():
 if e['status']!='done':continue
 for at in e.get('attempts',[]):
  proof=at.get('independent_verification')
  if at.get('count')==200 and isinstance(proof,str) and proof not in current and Path(proof).is_file():
   p=js(proof)
   if p.get('passed') and p.get('sample_count')==200:alternates.append(dict(model_id=mid,attempt=at,proof=proof,proof_SHA256=sha(proof),keys=list(p)))
out=dict(UTC=datetime.now(timezone.utc).isoformat(),noncurrent_completed_accepted_job_candidates=sorted(jobs,key=lambda r:r['bytes'],reverse=True),noncurrent_attempt_proof_candidates=alternates,read_only_no_candidate_deletion=True)
raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode();p=H/'historical_complete_accepted200_metadata_read_only.json';p.write_bytes(raw);print(json.dumps(dict(**out,evidence_path=str(p),evidence_SHA256=hashlib.sha256(raw).hexdigest()),ensure_ascii=False))
