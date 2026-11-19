import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,os,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'after_Huihui_real_dual181_then_GLM46_own_completed3_component_capacity_serial_v1_20261119';I=B/'after_Huihui_dual181_then_GLM46_serial_queue_actual_launch_20261119'
def maybe(p):
 if not p.exists():return None
 try:return json.loads(p.read_bytes())
 except ValueError:return dict(partial_JSON=True,bytes=p.stat().st_size)
files={n:maybe(H/n) for n in ['state.json','actual_completion.json','NOTIFY_error.json','NOTIFY_capacity_blocked.json','first200_measurement_approval.json','current_Huihui_retention_actual.json']};launch=maybe(I/'actual_launch.json');pid=launch['PID'];p=Path('/proc')/str(pid);actual=dict(exists=p.exists())
if p.exists():actual.update(argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),cwd=os.readlink(p/'cwd'),stat=(p/'stat').read_text())
logs={x.name:x.read_text(errors='replace')[-5000:] for x in I.glob('*.log') if x.is_file()}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),launch=launch,actual_process=actual,files=files,logs=logs),ensure_ascii=False))
