import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
report=R/'workspace/reports/microsoft_Phi-3.5-vision-instruct/image-text-to-text/20261118_054425'
line=json.loads((report/'operator_lineage.json').read_bytes())['actual_execution'];home=Path(line['evidence_directory'])
assert home.is_relative_to(R/'workspace/diagnostics/model_runs')
files={}
for p in home.iterdir():
 if p.is_file():files[p.name]=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,value=json.loads(p.read_bytes()) if p.suffix=='.json' and p.stat().st_size<100000 else None)
proc={}
for pid in [1515122,1515134,1515420]:
 p=Path('/proc')/str(pid)
 if p.exists():
  proc[str(pid)]={n:(p/n).read_text() for n in ['stat','wchan','io','status']}
H=B/'Phi35_own_true3_full_shape_capacity_approved_fresh3_fixed200_continuation_v1_0616_20261118';continuation={}
for name in ['state.json','actual_completion.json','NOTIFY_error.json','full_shape_complete_numeric_storage_capacity_review.json']:
 p=H/name
 if p.exists():continuation[name]=json.loads(p.read_bytes())
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),report_files=[p.name for p in report.iterdir()],evidence_directory=str(home),files=files,proc=proc,continuation=continuation),ensure_ascii=False))
