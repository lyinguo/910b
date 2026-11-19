import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'authorized_verified180_current_accepted_nonexample_reaudit_user_serial_continue_20261119';a=json.loads((H/'discovery.json').read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
print_policy=js(D/'A100_ALIGNMENT_EVIDENCE_RETENTION_POLICY.json')
jobs={}
for h in [B/'HuihuiGLM46Flash_own_original_BF16_2048_true3_v1_real_FLOCK_20261118',B/'GLM46Flash_own_original_BF16_thinking4096_true3_v2_actual_D965_real_FLOCK_1653_20261118']:
 j=js(h/'job.json');jobs[str(h)]=dict(protected_canonical_hashes=j.get('protected_canonical_hashes'),spec_static_keys=list(j['spec']))
details=[]
for row in a['skipped']:
 if row['reason'] not in ['Full original logical0:199 mapping not closed','Unique own numerical home not closed']:continue
 pf=Path(row['proof']);proof=js(pf);report=Path(proof.get('report_dir',next(r['report_dir'] for r in a['all_selected_proof_metadata'] if r['model_id']==row['model_id'])));line=js(report/'operator_lineage.json').get('actual_execution',{});npzs=[]
 for key,digest in proof.get('external_evidence_sha256',proof.get('external_evidence_SHA256',{})).items():
  f=Path(key)
  if f.suffix.lower()=='.npz' and f.is_file():npzs.append(dict(path=key,bytes=f.stat().st_size,SHA256=digest))
 trace_samples={}
 for key in ['sample_trace','per_sample_evidence','full_numeric_group_index','final_trace_manifest','runtime_evidence','actual_batch_manifest','execution_trace','trace_path']:
  if isinstance(line.get(key),str) and Path(line[key]).is_file():
   p=Path(line[key]);raw=p.read_text()
   try:
    rows=[json.loads(x) for x in raw.splitlines()] if p.suffix=='.jsonl' else js(p)
    if isinstance(rows,list):sample=rows[:1]
    elif isinstance(rows,dict):sample={k:(v[:1] if isinstance(v,list) else v) for k,v in rows.items() if k not in ['source_sha256','source_SHA256','external_evidence_sha256','external_evidence_SHA256']}
    trace_samples[key]=dict(path=str(p),sample=sample,records=len(rows))
   except Exception as e:trace_samples[key]=dict(path=str(p),error=repr(e))
 details.append(dict(model_id=row['model_id'],skip_reason=row['reason'],proof=str(pf),report_dir=str(report),lineage=line,existing_NPZ_count=len(npzs),existing_NPZ_bytes=sum(x['bytes'] for x in npzs),sample_NPZ=npzs[:2],trace_samples=trace_samples))
out=dict(UTC=datetime.now(timezone.utc).isoformat(),retention_policy=print_policy,existing_waiter_parent_job_protections=jobs,model_details=sorted(details,key=lambda r:r['existing_NPZ_bytes'],reverse=True),actual_free_bytes=shutil.disk_usage(R).free)
p=H/'unmapped_current_accepted_trace_formats_read_only.json';raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode();p.write_bytes(raw);print(json.dumps(dict(**out,evidence_path=str(p),evidence_SHA256=hashlib.sha256(raw).hexdigest()),ensure_ascii=False))
