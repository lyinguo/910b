import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
p=Path('/root/projects/comperror/workspace/diagnostics/batch_910b/A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/verified182_GLM_accepted_correct_nonexamples_user_serial_Single_capacity_recovery_20261120/plan.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='fceeb6a183684448cfccdd8d5a7cf863837dbf9981d9e836f7c0514e45df6ab7'
print(p.read_text())
