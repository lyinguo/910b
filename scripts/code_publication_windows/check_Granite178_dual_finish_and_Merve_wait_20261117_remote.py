import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pub=B/'Granite41_original_HF_BF16_verified178_blank_G_publication_20261117'
inp=B/'Granite41_verified178_publication_review_inputs_v1_20261117'
q=B/'MervePali_own603_wait_Granite_delivery_release_then_new_numeric3_capacity_fresh3_fixed200_continuation_20261117'
o={'UTC':datetime.now(timezone.utc).isoformat(),'files':{},'publication_files':[p.name for p in pub.iterdir()]}
for p in [inp/'publication_actual_completion.json',inp/'publication_stdout.log',inp/'publication_stderr.log',q/'state.json',q/'actual_completion.json',q/'NOTIFY_error.json']:
 if p.exists():
  raw=p.read_bytes()
  try:v=json.loads(raw)
  except Exception:v=raw.decode(errors='replace')
  o['files'][str(p)]={'SHA256':sha(p),'value':v}
print(json.dumps(o,ensure_ascii=False))
