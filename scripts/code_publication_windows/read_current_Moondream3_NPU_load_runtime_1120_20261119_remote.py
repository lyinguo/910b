import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,os
R=Path('/root/projects/comperror'); B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'; H=B/'Moondream3_own_actual_NPU_loading_all662_runtime_buffers_no_forward_1120_20261119'
a={n:json.loads((H/n).read_bytes()) for n in ['state.json','actual_completion.json','review.json'] if (H/n).exists()}
if 'state.json' in a:
 pid=a['state.json']['PID']; a['actual_proc']={n:Path(f'/proc/{pid}/{n}').read_bytes().decode(errors='replace').replace(chr(0),' ') for n in ['stat','cmdline','io'] if Path(f'/proc/{pid}/{n}').exists()}
print(json.dumps(a,ensure_ascii=False))
