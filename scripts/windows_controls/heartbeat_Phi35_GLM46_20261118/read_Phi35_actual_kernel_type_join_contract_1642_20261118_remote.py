import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv,json,hashlib
H=Path('/root/projects/comperror/workspace/diagnostics/model_runs/Phi35_own_original_BF16_native_full_values_c_zj1opm')
p,=H.rglob('task_time.csv');q=p.with_name('kernel_details.csv')
def rows(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
t,k=rows(p),rows(q);tkeys=[(r['Device_id'],r['stream_id'],r['task_id']) for r in t];kkeys=[(r['Device_id'],r['Stream ID'],r['Task ID']) for r in k]
matched=[r for r,key in zip(t,tkeys) if key in set(kkeys)]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),task_path=str(p),kernel_path=str(q),SHA256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p,q]},all_task_types=dict(Counter(r['kernel_type'] for r in t)),all_actual_kernel_cores=dict(Counter(r['Accelerator Core'] for r in k)),kernel_details_rows=len(k),kernel_details_unique_task_keys=len(set(kkeys)),matched_task_time_rows=len(matched),all_detail_keys_in_task_time=set(kkeys).issubset(set(tkeys)),matched_types=dict(Counter(r['kernel_type'] for r in matched)),matched_name_values=list(dict.fromkeys(r['kernel_name'] for r in matched))[:20],actual_new_NPU_forwards=0,no_existing_evidence_changed=True),ensure_ascii=False))
