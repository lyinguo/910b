import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';J=B/'Bee8BSFT_own_original_BF16_4096_true3_v1_real_FLOCK_20261116';H=B/'Bee8BSFT_own_true3_then_complete_shape_capacity_fresh3_fixed200_continuation_v1_20261116';out={}
def get(p):
 p=Path(p)
 if p.is_file():
  b=p.read_bytes();q=json.loads(b);out[str(p)]={'SHA256':hashlib.sha256(b).hexdigest(),'JSON':q};return q
j=get(J/'job.json');c=get(J/'completion.json');get(H/'capacity_review.json');get(H/'actual_completion.json')
if c:
 for a in c.get('accepted',[]):get(a['proof'])
if j:
 cfg=get(R/j['spec']['config']);get(cfg['model']['kwargs']['protocol_file'])
print(json.dumps(out,ensure_ascii=False))