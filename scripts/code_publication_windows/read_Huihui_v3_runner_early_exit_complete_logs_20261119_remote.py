import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');B=R/'workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'HuihuiGLM46_completed_true3_verified4972_cleanup_lossless_LZMA_serial_resumer_v3_20261119';J=B/'HuihuiGLM46_completed_true3_lossless_LZMA_approved_NEWfresh3_fixed200_v3_real_FLOCK_20261119'
out={}
for p in H.iterdir():
 if p.suffix=='.log':out[str(p)]=dict(bytes=p.stat().st_size,text=p.read_text(errors='replace'))
out['job']=json.loads((J/'job.json').read_bytes());print(json.dumps(out,ensure_ascii=False))
