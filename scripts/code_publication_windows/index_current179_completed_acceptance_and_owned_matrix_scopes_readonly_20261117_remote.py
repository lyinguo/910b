import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';S=R/'workspace/diagnostics/model_runs';js=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();state=js(D/'state.json');reg=js(D/'registry.json');rows=[]
for mid,e in state['models'].items():
 if e['status']!='done':continue
 rr=reg[mid];folder=Path(e['report_dir']);line=js(folder/'operator_lineage.json');report=js(folder/'eval_report.json');candidates=[]
 for v in [rr.get('accepted_formal200_proof'),e.get('accepted_formal200_proof'),str(Path(e['active_attempt'])/'verification_stdout.json') if e.get('active_attempt') else None]:
  if v and Path(v).is_file() and v not in candidates:candidates.append(v)
 proofs=[]
 for name in candidates:
  p=Path(name);raw=p.read_bytes()
  try:v=json.loads(raw)
  except Exception:continue
  proofs.append(dict(path=name,SHA256=sha(p),value={k:x for k,x in v.items() if k not in ['external_evidence_sha256','numeric_sample_checks','numeric_review','numeric_checks','exact_array_checks','per_sample_numeric']},external_count=len(v.get('external_evidence_sha256',{})),owned_matrix_files=[dict(path=n,SHA256=h) for n,h in v.get('external_evidence_sha256',{}).items() if Path(n).suffix in ['.npz','.npy'] and Path(n).is_relative_to(S)]))
 execution=line.get('actual_execution',{})
 rows.append(dict(model_id=mid,score=e.get('score'),report_dir=str(folder),current_state_keys=list(e),registry_keys=list(rr),accepted_proofs=proofs,registry_proof=rr.get('accepted_formal200_proof'),registry_proof_SHA256=rr.get('accepted_formal200_proof_sha256'),lineage_top_keys=list(line),lineage_actual_execution=execution,report_sample_count=report.get('statistics',{}).get('total_samples'),report_summary=report.get('summary'),relevant_state={k:v for k,v in e.items() if any(n in k.lower() for n in ['proof','verif','accepted','attempt'])}))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),canonical_done_count=len(rows),rows=rows,no_array_value_or_weight_read_no_mutation=True),ensure_ascii=False))
