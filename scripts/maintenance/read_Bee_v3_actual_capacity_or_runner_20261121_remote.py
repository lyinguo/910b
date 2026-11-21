import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct';H=B/'BeeSFT_completed_storage_NF_capacity_v3_actual_saved_patch_axis_serial_20261121';o={'UTC':datetime.now(timezone.utc).isoformat(),'files':{}}
for n in ['state.json','actual_completion.json','capacity_review.json','first200_measurement_approval.json','completed_storage_proof_reuse.json']:
 p=H/n
 if p.is_file():
  b=p.read_bytes();q=json.loads(b);o['files'][n]={'SHA256':hashlib.sha256(b).hexdigest(),'JSON':q}
print(json.dumps(o))