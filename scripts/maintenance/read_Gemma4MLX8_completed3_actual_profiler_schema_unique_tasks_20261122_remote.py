import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,collections,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
J=B/'Gemma4E4BMLX8_own_original_Transformers_BF16_true3_v1_real_FLOCK_20261122'
done=json.loads((J/'completion.json').read_bytes());report=Path(done['report_dir']);line=json.loads((report/'operator_lineage.json').read_bytes())['actual_execution'];final=json.loads(Path(line['final_trace_manifest']).read_bytes());out=[]
for path,digest in final['first_real_vision_prefill_all_profiler_files_SHA256'].items():
 p=Path(path)
 if 'task_time' not in p.name and 'kernel_details' not in p.name:continue
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 out.append(dict(path=path,SHA256=digest,fieldnames=list(rows[0]) if rows else [],count=len(rows),kernel_types=dict(collections.Counter(r.get('kernel_type',r.get('Accelerator Core')) for r in rows)),devices=dict(collections.Counter(r.get('Device_id') for r in rows)),rows=rows))
print(json.dumps(dict(original_completion=done,profiler=out),ensure_ascii=False))
