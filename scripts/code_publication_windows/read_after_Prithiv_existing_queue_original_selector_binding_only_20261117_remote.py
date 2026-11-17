import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,os
os.chdir('/root');R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';p=R/'scripts/code_publication_windows/A100_select_available_original_queue_1245_remote.py';rows=[]
for f in [p,R/'scripts/A100_select_available_original_queue_1245_remote.py']:
 if f.exists():rows.append(dict(path=str(f),SHA256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
J=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/Prithiv_completed_card2_true3_verified179_capacity_approved_NEWfresh3_fixed200_original_real_FLOCK_20261117'
print(json.dumps(dict(selector_files=rows,Prithiv_state=json.loads((J/'state.json').read_bytes()))))
