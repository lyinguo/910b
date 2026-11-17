import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';state=json.loads((D/'state.json').read_bytes());reg=json.loads((D/'registry.json').read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[];examples={};select={'OpenGVLab/HoVLE-HD','huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated','unsloth/Llama-3.2-11B-Vision-Instruct','MathLLMs/FigCodifier','opendatalab/MinerU2.5-Pro-2604-1.2B','HuggingFaceM4/idefics2-8b','InternScience/StructTable-base','nvidia/EGM-8B'}
for mid,e in state['models'].items():
 if e['status']!='done':continue
 matches=[]
 for a in e.get('attempts',[]):
  if a.get('count')!=200 or a.get('report_dir')!=e['report_dir']:continue
  v=a.get('independent_verification');p=Path(v) if isinstance(v,str) else None
  if not p or not p.is_file():continue
  raw=p.read_bytes()
  try:q=json.loads(raw)
  except Exception:continue
  matches.append(dict(path=str(p),SHA256=sha(p),top_keys=list(q),scalar_fields={k:v for k,v in q.items() if not isinstance(v,(dict,list))},attempt_returncode=a.get('returncode'),attempt_completed_at=a.get('completed_at')))
  if mid in select:
   folder=Path(e['report_dir']);line=json.loads((folder/'operator_lineage.json').read_bytes());ex=line.get('actual_execution',{});rr=reg[mid]
   examples[mid]=dict(accepted_verification=q,accepted_path=str(p),accepted_SHA256=sha(p),registry_verifier=rr.get('verifier'),registry_verifier_SHA256=rr.get('source_sha256',{}).get(rr.get('verifier')),full_five_SHA256={n:sha(folder/n) for n in ['eval_report.json','sample_inputs.jsonl','sample_outputs.jsonl','run.log','operator_lineage.json']},execution_scalars={k:v for k,v in ex.items() if isinstance(v,(str,int,float,bool))},execution_object_keys={k:list(v)[:20] if isinstance(v,dict) else len(v) for k,v in ex.items() if isinstance(v,(dict,list))})
 rows.append(dict(model_id=mid,report_dir=e['report_dir'],canonical_registry_proof=reg[mid].get('accepted_formal200_proof'),legacy_current_accepted_verification=matches))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),rows=rows,selected_legacy_full_receipts=examples,no_CPU200_or_array_reverification=True),ensure_ascii=False))
