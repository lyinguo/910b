import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json
R=Path('/root/projects/comperror');F=R/'workspace/reports/microsoft_Phi-3.5-vision-instruct/image-text-to-text/20261118_054425'
line=json.loads((F/'operator_lineage.json').read_bytes())['actual_execution'];H=Path(line['evidence_directory']);files=[];csvs={}
for p in H.rglob('*'):
 if not p.is_file():continue
 files.append(dict(relative=str(p.relative_to(H)),bytes=p.stat().st_size))
 if p.suffix=='.csv':
  with p.open(encoding='utf-8-sig') as f:
   reader=csv.DictReader(f);rows=list(reader)
  csvs[str(p)]=dict(columns=reader.fieldnames,rows=len(rows),first=rows[:2],last=rows[-1:],SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
idx=json.loads(Path(line['full_numeric_group_index']).read_bytes());groups=[]
for r in idx['groups']:
 p=Path(r['path']);g=json.loads(p.read_bytes());groups.append(dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_native_forward_calls=g['actual_native_forward_calls'],frames=len(g['frames']),archive=g['archive'],archive_bytes=Path(g['archive']).stat().st_size,manifest_bytes=p.stat().st_size,physical_unique_arrays=g['lossless_shared_tensor_storage']['physical_unique_arrays']))
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),actual_lineage=line,all_own_evidence_file_stats=files,all_CSV_headers_real_rows=csvs,groups=groups),ensure_ascii=False))
